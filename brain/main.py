"""Integration entry point: webcam hand gesture -> context decision -> NAO action.

Run (from the project root, inside ``.venv311``)::

    python -m brain.main --mock                      # no robot, simulated sonars
    python -m brain.main --robot zyra                # real robot (supervised!)
    python -m brain.main --robot nyx --ip 127.0.0.1 --port <port>   # Choregraphe

Per frame: MediaPipe Hands (one hand) -> ``GesturePredictor`` (majority vote).
When a NEW stable gesture (not ``NONE``) appears and the cooldown has elapsed,
the robot is sensed (sonars, motion, posture), the context dict is built
(exactly ``brain.context.FEATURE_COLUMNS``), ``ContextDecider`` picks the action
(learned model, then the hard safety layer) and the ``Executor`` sends it to the
bridge together with the eye colour and the spoken reason. Every decision is
appended to a CSV log.

Low confidence -> ``ASK_CONFIRM``: showing the same gesture again with
confidence >= ``MIN_CONF`` within ``--confirm-window`` seconds evaluates it
normally (the cooldown does not block that confirmation); otherwise the pending
confirmation expires and the eyes go back to blue.

If ``models/context_model.joblib`` does not exist the intent rules + the safety
layer are used (``ContextDecider(None)``), with a warning.

Pure logic (``build_context``, ``Executor``, ``SpeechGate``,
``ConfirmationTracker``, ``Pilot``, ``MockScenario``, ``overlay_lines``,
``shutdown``) lives apart from the camera I/O in ``camera_loop`` so it can be
tested without a webcam or a robot. cv2/mediapipe are imported lazily.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import queue
import sys
import time
import types
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime

from . import vocab
from .config import DEFAULT_ENV_PATH, Config, load_config, parse_env_file
from .context import FEATURE_COLUMNS
from .context.safety import MIN_CONF, Decision, validate
from .gestures.features import hand_size
from .music import MusicPlayer
from .voice.keywords import keyword_intent

NONE_LABEL = "NONE"
STANDING_POSTURES = ("Stand", "StandInit", "StandZero")
DEFAULT_COOLDOWN = 2.5  # s between two robot commands
DEFAULT_CONFIRM_WINDOW = 4.0  # s to repeat a gesture after ASK_CONFIRM
SPEECH_GAP = 3.0  # s: the same sentence is not spoken twice within this time
SENSE_INTERVAL = 0.5  # s between sonar refreshes for the overlay
# The predictor must let low-confidence gestures through so the safety layer
# can answer ASK_CONFIRM; below this it reports NONE.
PREDICTOR_MIN_CONF = 0.4
ERROR_SPEECH = "No pude ejecutar la orden."

MOCK_CLEAR = 1.5  # m, simulated free path
MOCK_NEAR = 0.40  # m, front obstacle inside the graded-advance band
MOCK_BLOCKED = 0.20  # m, front/side obstacle that blocks movement
MOCK_SIT, MOCK_STAND = "Sit", "StandInit"

LOG_COLUMNS = (
    "timestamp", "gesture", "confidence", "sonar_left", "sonar_right", "hand_size",
    "moving", "standing", "requested_intent", "action", "params", "reason", "eye_color",
    "overridden", "bridge_ok", "simulated", "latency_ms", "mode", "input", "bridge_error",
)
AMBIGUOUS_GESTURES = ("LOVE_YOU", "VICTORY")  # S9: confusable pair, optional confirmation
BLINK_INTERVAL = 1.0  # s, minimum gap between two "gesture detected" eye blinks
DANCE_INTERVAL = 4.0  # s per dance-loop step (each loop_x animation lasts ~4 s)
MODE_EYES = {vocab.MODE_MOVEMENT: "blue", vocab.MODE_MUSIC: "purple"}
MODE_SPEECH = {vocab.MODE_MOVEMENT: "Modo movimiento", vocab.MODE_MUSIC: "Modo música"}
INPUT_SPEECH = {vocab.INPUT_VOICE: "Te escucho por voz", vocab.INPUT_GESTURES: "Volvemos a los gestos"}
MOVEMENT_INTENTS = (vocab.ADVANCE, vocab.TURN_LEFT, vocab.TURN_RIGHT, vocab.SIT, vocab.STAND,
                    vocab.TOGGLE_POSTURE, vocab.WAVE)
EYE_BGR = {  # overlay swatch colours (OpenCV uses BGR)
    "green": (0, 200, 0), "yellow": (0, 220, 255), "red": (0, 0, 230),
    "blue": (230, 100, 0), "white": (255, 255, 255),
    "magenta": (255, 0, 255), "cyan": (255, 255, 0), "orange": (0, 165, 255),
    "purple": (128, 0, 128), "pink": (180, 105, 255),
}


# ------------------------------------------------------------------ pure logic
def is_standing(posture) -> bool:
    return posture in STANDING_POSTURES


def build_context(gesture: str, confidence: float, sense: dict, size: float) -> dict:
    """Context dict; keys are exactly ``FEATURE_COLUMNS`` (in that order).

    A failed ``sense`` yields NaN sonars (the safety layer treats them as
    unsafe), not moving and not standing.
    """
    nan = float("nan")
    ok = bool(sense.get("ok", True))
    ctx = {
        "gesture": gesture,
        "gesture_confidence": float(confidence),
        "sonar_left": float(sense.get("sonar_left", nan)) if ok else nan,
        "sonar_right": float(sense.get("sonar_right", nan)) if ok else nan,
        "hand_size": float(size),
        "moving": int(bool(sense.get("moving", False))) if ok else 0,
        "standing": int(is_standing(sense.get("posture"))) if ok else 0,
    }
    return {key: ctx[key] for key in FEATURE_COLUMNS}


def ascii_text(text: str) -> str:
    """Strip accents and non-ASCII characters (OpenCV fonts cannot draw them)."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


class SpeechGate:
    """Suppresses the identical sentence if it was spoken less than ``min_gap`` s ago."""

    def __init__(self, min_gap: float = SPEECH_GAP):
        self.min_gap = min_gap
        self._text: str | None = None
        self._t: float | None = None

    def allow(self, text: str, now: float) -> bool:
        if text == self._text and self._t is not None and now - self._t < self.min_gap:
            return False
        self._text, self._t = text, now
        return True


@dataclass
class ExecResult:
    ok: bool = True
    errors: list = field(default_factory=list)
    acks: list = field(default_factory=list)
    motion_ack_t: float | None = None  # clock value when the motion command was acknowledged
    simulated: bool = False


class Executor:
    """Maps a ``Decision`` to bridge commands (pure mapping, no camera).

    Order: safety ``stop`` first (STOP, SIT, STAND, or a WARN_* while moving),
    eye colour, the motion itself, then the spoken reason. A refusal from the
    bridge (``ok: false``) turns the eyes red and speaks a short error instead.
    """

    def __init__(self, client, speech: SpeechGate | None = None, clock=time.perf_counter,
                 music: MusicPlayer | None = None):
        self.client = client
        self.speech = speech or SpeechGate()
        self.clock = clock
        self.music = music or MusicPlayer()

    def execute(self, decision, moving: bool, now: float | None = None) -> ExecResult:
        now = self.clock() if now is None else now
        result = ExecResult()

        def send(ack: dict) -> dict:
            result.acks.append(ack)
            result.simulated = result.simulated or bool(ack.get("simulated"))
            if not ack.get("ok"):
                result.errors.append(str(ack.get("error", "unknown error")))
            return ack

        action, params = decision.action, decision.params
        warn = action.startswith("WARN_")
        if action in ("STOP", "SIT", "STAND") or (warn and moving):
            send(self.client.stop())
            self.music.stop()
        send(self.client.eyes(decision.eye_color))
        if action in ("ADVANCE", "ADVANCE_SHORT"):
            send(self.client.move(float(params["x"]), 0.0))
        elif action in ("TURN_LEFT", "TURN_RIGHT"):
            send(self.client.move(0.0, float(params["theta"])))
        elif action in ("WAVE", "DANCE"):
            self.music.play_random()
            if hasattr(self.client, "dance"):
                send(self.client.dance())
            else:
                send(self.client.wave())
        elif action == "GREET":
            if hasattr(self.client, "greet"):
                send(self.client.greet())
            else:
                send(self.client.wave())
        elif action == "SIT":
            send(self.client.posture("Sit"))
        elif action == "STAND":
            send(self.client.posture("StandInit"))
        # ASK_CONFIRM / WAIT / WARN_*: no motion
        result.motion_ack_t = self.clock()

        if result.errors:
            result.ok = False
            print(f"WARNING: bridge refused a command: {'; '.join(result.errors)}", file=sys.stderr)
            self.client.eyes("red")
            text = ERROR_SPEECH
        else:
            text = decision.reason
        if text and self.speech.allow(text, now):
            self.client.say(text)
        return result


class ConfirmationTracker:
    """Pending ``ASK_CONFIRM`` for one gesture, valid for ``window`` seconds."""

    def __init__(self, window: float = DEFAULT_CONFIRM_WINDOW):
        self.window = window
        self.pending_gesture: str | None = None
        self._deadline = 0.0

    def ask(self, gesture: str, now: float) -> None:
        self.pending_gesture = gesture
        self._deadline = now + self.window

    def clear(self) -> None:
        self.pending_gesture = None

    def remaining(self, now: float) -> float:
        if self.pending_gesture is None:
            return 0.0
        return max(0.0, self._deadline - now)

    def matches(self, gesture: str, confidence: float, now: float) -> bool:
        return (self.pending_gesture is not None and gesture == self.pending_gesture
                and confidence >= MIN_CONF and now <= self._deadline)

    def expire(self, now: float) -> bool:
        """Drop the pending confirmation if its window is over; True exactly once."""
        if self.pending_gesture is not None and now > self._deadline:
            self.pending_gesture = None
            return True
        return False


@dataclass
class DecisionRecord:
    timestamp: str
    gesture: str
    confidence: float
    context: dict
    requested_intent: str
    decision: object
    bridge_ok: bool
    simulated: bool
    latency_ms: float
    explain: str = ""
    mode: str = vocab.MODE_MOVEMENT
    input_mode: str = vocab.INPUT_GESTURES
    bridge_error: str = ""

    def row(self) -> list:
        ctx, dec = self.context, self.decision
        return [
            self.timestamp, self.gesture, f"{self.confidence:.3f}",
            f"{ctx['sonar_left']:.3f}", f"{ctx['sonar_right']:.3f}", f"{ctx['hand_size']:.3f}",
            ctx["moving"], ctx["standing"], self.requested_intent, dec.action,
            json.dumps(dec.params, sort_keys=True), dec.reason, dec.eye_color,
            int(bool(dec.overridden)), int(self.bridge_ok), int(self.simulated),
            f"{self.latency_ms:.1f}", self.mode, self.input_mode, self.bridge_error,
        ]


def default_log_path() -> str:
    return os.path.join("runs", datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv")


class DecisionLogger:
    """Writes one CSV row per decision (flushed immediately)."""

    def __init__(self, path: str):
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        self.path = path
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(LOG_COLUMNS)
        self._fh.flush()

    def log(self, record: DecisionRecord) -> None:
        self._writer.writerow(record.row())
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


def _call(client, name: str, *args):
    """Call an optional client method; a missing method or a failure is reported, never raised."""
    fn = getattr(client, name, None)
    if fn is None:
        return None
    try:
        return fn(*args)
    except Exception as exc:  # the bridge may be flaky; feedback must not crash the loop
        print(f"WARNING: {name} failed: {exc}", file=sys.stderr)
        return None


class DanceLoop:
    """Cycles ``dance(loop_a..loop_d)`` while active; driven by ``tick(now)``, never sleeps."""

    STYLES = ("loop_a", "loop_b", "loop_c", "loop_d")

    def __init__(self, client, interval: float = DANCE_INTERVAL, styles=STYLES):
        self.client = client
        self.interval = interval
        self.styles = tuple(styles)
        self.active = False
        self._index = 0
        self._next_at = 0.0

    def start(self, now: float) -> None:
        if self.active:
            return
        self.active = True
        self._index = 0
        _call(self.client, "dance", self.styles[0])
        self._next_at = now + self.interval

    def tick(self, now: float) -> None:
        if not self.active or now < self._next_at:
            return
        self._index = (self._index + 1) % len(self.styles)
        _call(self.client, "dance", self.styles[self._index])
        self._next_at = now + self.interval

    def stop(self, send_stop: bool = True) -> None:
        was_active, self.active = self.active, False
        if was_active and send_stop:
            _call(self.client, "stop")


class Pilot:
    """Gesture/voice -> decision -> robot state machine (no camera, no I/O besides the client).

    Two axes: ``mode`` (MOVEMENT / MUSIC) and ``input`` (GESTURES / VOICE).

    A gesture command fires when the stable gesture is not ``NONE``, differs from
    the gesture that last produced a command (reset whenever ``NONE`` is seen)
    and ``cooldown`` seconds have passed since the previous command. A gesture
    that is held fires once; one suppressed by the cooldown fires as soon as it
    ends if it is still held. A pending ``ASK_CONFIRM`` bypasses edge and
    cooldown when the same gesture comes back with confidence >= ``MIN_CONF`` in
    time. ``OPEN_PALM`` (e-stop) always fires on its rising edge, ignoring the
    cooldown and the input source. While input is VOICE only ``VICTORY`` and
    ``OPEN_PALM`` are honoured from gestures; voice intents are ignored while
    input is GESTURES.

    Movement intents from gestures go through the ``decider`` (model + safety);
    voice movement intents go through ``safety.validate`` directly. LOVE_YOU
    and music intents are not sonar gated. ``step`` also services the dance
    loop and the voice queue, so one call per camera frame drives everything.

    ``latency_ms`` = stable gesture (command triggered) -> bridge ack of the
    motion command: sense + context + decide + act.
    """

    def __init__(self, client, predictor, decider, cooldown: float = DEFAULT_COOLDOWN,
                 confirm_window: float = DEFAULT_CONFIRM_WINDOW, clock=time.perf_counter,
                 logger: DecisionLogger | None = None,
                 sense_interval: float | None = SENSE_INTERVAL,
                 sonar_override: "VirtualSonar | None" = None,
                 music: MusicPlayer | None = None, dj=None, predictor2=None, hold=None,
                 voice=None, confirm_ambiguous: bool = False,
                 blink_interval: float = BLINK_INTERVAL, dance_interval: float = DANCE_INTERVAL):
        self.client = client
        self.sonar_override = sonar_override
        self.predictor = predictor
        self.predictor2 = predictor2  # second hand; enables the two-hands mode toggle
        if hold is None and predictor2 is not None:
            from .gestures.two_hands import TwoHandsHold
            hold = TwoHandsHold("THUMBS_UP", clock=clock)
        self.hold = hold
        self.decider = decider
        self.cooldown = cooldown
        self.clock = clock
        self.logger = logger
        self.sense_interval = sense_interval
        self.music = music or MusicPlayer()
        self.dj = dj
        self.dance = DanceLoop(client, dance_interval)
        self.voice = voice
        self.confirm_ambiguous = confirm_ambiguous
        self.blink_interval = blink_interval
        self.executor = Executor(client, clock=clock, music=self.music)
        self.confirm = ConfirmationTracker(confirm_window)
        self.mode = vocab.MODE_MOVEMENT
        self.input_mode = vocab.INPUT_GESTURES
        self.mic_open = False
        self.voice_status = ""
        self.voice_queue: queue.Queue = queue.Queue()
        self.gesture = NONE_LABEL
        self.confidence = 0.0
        self.sense: dict | None = None  # last good sense reading (overlay)
        self.last_record: DecisionRecord | None = None
        self.last_explain = ""
        self._acted: str | None = None
        self._last_fire: float | None = None
        self._last_sense_t: float | None = None
        self._prev_label = NONE_LABEL
        self._last_blink: float | None = None

    # ---------------------------------------------------------------- helpers
    @property
    def eye_color(self) -> str:
        return MODE_EYES[self.mode]

    def cooldown_remaining(self, now: float | None = None) -> float:
        if self._last_fire is None:
            return 0.0
        now = self.clock() if now is None else now
        return max(0.0, self.cooldown - (now - self._last_fire))

    def _read_sense(self, now: float) -> dict:
        reading = self.client.sense()
        if reading.get("ok") and self.sonar_override is not None:
            reading = self.sonar_override.patch(reading)
        if reading.get("ok"):
            self.sense = reading
            self._last_sense_t = now
        return reading

    def _refresh_sense(self, now: float) -> None:
        if self.sense_interval is None:
            return
        if self._last_sense_t is None or now - self._last_sense_t >= self.sense_interval:
            self._read_sense(now)

    def _bridge_error(self) -> str:
        error = getattr(self.client, "last_error", None)
        return error if isinstance(error, str) else ""

    def _say(self, text: str, now: float) -> None:
        if text and self.executor.speech.allow(text, now):
            _call(self.client, "say", text)

    # ------------------------------------------------------------ state axes
    def announce_state(self) -> None:
        """Set eyes and ear LEDs to match the current state (no speech)."""
        _call(self.client, "eyes", self.eye_color)
        _call(self.client, "ears", self.input_mode == vocab.INPUT_VOICE)

    def set_mode(self, mode: str, now: float) -> str:
        """Switch MOVEMENT/MUSIC with feedback; leaving MUSIC stops music and the dance loop."""
        if mode == self.mode:
            return MODE_SPEECH[mode]
        self.mode = mode
        if mode == vocab.MODE_MOVEMENT:
            self.stop_media()
        self.confirm.clear()
        _call(self.client, "eyes", self.eye_color)
        self._say(MODE_SPEECH[mode], now)
        return MODE_SPEECH[mode]

    def set_input(self, input_mode: str, now: float) -> str:
        """Switch GESTURES/VOICE with feedback (ear LEDs on while listening)."""
        if input_mode == self.input_mode:
            return INPUT_SPEECH[input_mode]
        self.input_mode = input_mode
        if input_mode != vocab.INPUT_VOICE and self.mic_open:
            self.mic_open = False
            if self.voice is not None:
                try:
                    self.voice.end()
                except Exception:
                    pass
        _call(self.client, "ears", input_mode == vocab.INPUT_VOICE)
        _call(self.client, "eyes", self.eye_color)
        self._say(INPUT_SPEECH[input_mode], now)
        return INPUT_SPEECH[input_mode]

    # ------------------------------------------------------------ media / stop
    def _halt_media(self) -> None:
        """Pause the music and stop the dance loop (no robot command)."""
        if self.dj is not None and self.dj.is_playing:
            self.dj.pause()
        self.dance.stop(send_stop=False)

    def emergency_stop(self) -> None:
        """Instant e-stop: robot stop (motion, dance, speech), music pause, dance loop off."""
        self.client.stop()
        self._halt_media()

    def stop_media(self) -> None:
        """Stop the music and the dance loop completely (also used on shutdown)."""
        self.music.stop()
        if self.dj is not None:
            self.dj.stop()
        self.dance.stop()

    # ------------------------------------------------------------------ voice
    def post_voice(self, intent: str, text: str = "") -> None:
        """Thread-safe: queue an intent from the voice thread for the main loop."""
        self.voice_queue.put((intent, text))

    def post_voice_text(self, text: str) -> None:
        """Queue the keyword-fallback intent of an already transcribed sentence."""
        self.post_voice(keyword_intent(text), text)

    def toggle_mic(self) -> bool:
        """Push-to-talk toggle (key ``v``): first call starts recording, second sends it."""
        if self.input_mode != vocab.INPUT_VOICE:
            return False
        if self.voice is None:
            self.voice_status = "Voz no disponible"
            return False
        try:
            if not self.mic_open:
                if getattr(self.voice, "busy", False):
                    self.voice_status = "Espera: procesando la voz anterior"
                    return False
                self.voice.begin()
                self.mic_open = True
                self.voice_status = "Grabando... (v para enviar)"
            else:
                self.voice.end()
                self.mic_open = False
                self.voice_status = "Procesando..."
        except Exception as exc:  # no mic / no sounddevice: report on the overlay, keep running
            self.mic_open = False
            self.voice_status = f"Error de microfono: {exc}"
            return False
        return True

    def _drain_voice(self, now: float) -> None:
        while True:
            try:
                intent, text = self.voice_queue.get_nowait()
            except queue.Empty:
                return
            if self.input_mode != vocab.INPUT_VOICE:
                continue
            self.voice_status = f"Oido: '{text}' -> {intent}"
            if intent in vocab.VOICE_INTENTS and intent != vocab.UNKNOWN:
                self.handle_intent(intent, now)

    def tick(self, now: float) -> None:
        """Service the dance loop and the voice queue (call once per frame)."""
        self.dance.tick(now)
        if self.dance.active and self.dj is not None and not self.dj.is_playing:
            self.dance.stop()  # the music ended or was paused elsewhere
        self._drain_voice(now)

    # ----------------------------------------------------------------- intents
    def handle_intent(self, intent: str, now: float | None = None, conf: float = 1.0,
                      label: str = "VOICE", size: float = 0.0) -> DecisionRecord | None:
        """Run one intent (voice, or a non-decider gesture) and log it."""
        now = self.clock() if now is None else now
        t0 = self.clock()
        outcome = self._apply_intent(intent, now, conf, label, size)
        if outcome is None:
            return None
        decision, ctx, ok, requested = outcome
        return self._record(label, conf, ctx, requested, decision, ok, False,
                            (self.clock() - t0) * 1000.0)

    def _apply_intent(self, intent: str, now: float, conf: float, label: str, size: float):
        """-> (decision, ctx, ok, requested) or None when the intent is not actionable."""
        if intent in MOVEMENT_INTENTS:
            sense = self._read_sense(now)
            ctx = build_context(label, conf, sense, size)
            action = intent
            if intent == vocab.TOGGLE_POSTURE:
                action = vocab.SIT if ctx["standing"] else vocab.STAND
            decision = validate(action, ctx)
            result = self.executor.execute(decision, moving=bool(ctx["moving"]), now=now)
            return decision, ctx, bool(sense.get("ok")) and result.ok, action
        ctx = build_context(label, conf, self.sense or {"ok": False}, size)
        spoken = False
        if intent == vocab.STOP:
            self.emergency_stop()
            text = "Me detengo."
        elif intent == vocab.CELEBRATE:
            text, spoken = self._celebrate(now), True
        elif intent == vocab.MUSIC_PLAY:
            text, spoken = self._music_play(now), True
        elif intent == vocab.MUSIC_PAUSE:
            if self.dj is not None:
                self.dj.pause()
            self.dance.stop()
            text, spoken = "Música en pausa.", True
        elif intent in (vocab.MUSIC_NEXT, vocab.MUSIC_PREV):
            ok = False
            if self.dj is not None:
                ok = self.dj.next() if intent == vocab.MUSIC_NEXT else self.dj.previous()
            if not ok:
                text = "No hay canciones."
            else:
                text = "Siguiente canción." if intent == vocab.MUSIC_NEXT else "Canción anterior."
                if self.dj.is_playing:
                    self.dance.start(now)  # no-op when the loop is already running
            spoken = True
        elif intent == vocab.MUSIC_STOP:
            if self.dj is not None:
                self.dj.stop()
            self.dance.stop()
            text, spoken = "Música detenida.", True
        elif intent == vocab.TOGGLE_MODE:
            text = self.set_mode(vocab.MODE_MUSIC if self.mode == vocab.MODE_MOVEMENT
                                 else vocab.MODE_MOVEMENT, now)
        elif intent == vocab.SET_MODE_MUSIC:
            text = self.set_mode(vocab.MODE_MUSIC, now)
        elif intent == vocab.SET_MODE_MOVEMENT:
            text = self.set_mode(vocab.MODE_MOVEMENT, now)
        elif intent == vocab.TOGGLE_INPUT:
            text = self.set_input(vocab.INPUT_GESTURES if self.input_mode == vocab.INPUT_VOICE
                                  else vocab.INPUT_VOICE, now)
        elif intent == vocab.SET_INPUT_VOICE:
            text = self.set_input(vocab.INPUT_VOICE, now)
        elif intent == vocab.SET_INPUT_GESTURES:
            text = self.set_input(vocab.INPUT_GESTURES, now)
        else:
            return None
        if spoken:
            self._say(text, now)
        return Decision(intent, {}, text, self.eye_color), ctx, True, intent

    def _music_play(self, now: float) -> str:
        if self.dj is None or not self.dj.play():
            return "No hay canciones para reproducir."
        self.dance.start(now)
        return "¡Música!"

    def _celebrate(self, now: float) -> str:
        """We Are the Champions + special dance; the dance loop gives way to it."""
        self.dance.stop()  # sends stop so the champions dance does not overlap a loop step
        if self.dj is not None:
            self.dj.play_champions()
        _call(self.client, "dance", "champions")
        return "¡Somos los campeones!"

    # -------------------------------------------------------------------- step
    def step(self, landmarks, handedness=None, now: float | None = None,
             second=None) -> DecisionRecord | None:
        """Process one frame (``landmarks`` is a (21, 3) array or None).

        ``second`` is an optional ``(landmarks, handedness)`` of a second hand.
        """
        now = self.clock() if now is None else now
        label, conf = self.predictor.update(landmarks, handedness)
        label2 = None
        if self.predictor2 is not None:
            label2 = self.predictor2.update(*(second if second is not None else (None, None)))[0]
        self.gesture, self.confidence = label, conf
        size = hand_size(landmarks) if landmarks is not None else 0.0

        if self.confirm.expire(now):
            _call(self.client, "eyes", self.eye_color)  # unanswered question: back to the mode colour
        self._refresh_sense(now)
        self.tick(now)

        if label != NONE_LABEL and self._prev_label == NONE_LABEL and (
                self._last_blink is None or now - self._last_blink >= self.blink_interval):
            self._last_blink = now
            _call(self.client, "blink")
        self._prev_label = label

        if self.hold is not None:
            labels = ([label] if landmarks is not None else []) + (
                [label2] if second is not None else [])
            if self.hold.update(labels, now):
                self._acted, self._last_fire = label, now
                return self.handle_intent(vocab.TOGGLE_MODE, now, conf, label, size)
            if len(labels) >= 2 and all(item == self.hold.label for item in labels):
                self._acted = label  # both hands up: the single-hand pipeline stays silent
                return None

        if label == NONE_LABEL:
            self._acted = None
            return None
        confirming = self.confirm.matches(label, conf, now)
        palm = label == "OPEN_PALM" and label != self._acted
        voice_only = self.input_mode == vocab.INPUT_VOICE
        if voice_only and label not in vocab.ALWAYS_ACTIVE_GESTURES:
            return None
        if palm and (self.mode == vocab.MODE_MUSIC or voice_only):
            self._acted, self._last_fire = label, now
            return self.handle_intent(vocab.STOP, now, conf, label, size)
        if not confirming and not palm and (label == self._acted
                                            or self.cooldown_remaining(now) > 0.0):
            return None
        return self._fire(label, conf, size, now, confirming)

    def _special_intent(self, label: str) -> str | None:
        """Intent of a gesture handled outside the decider (None = decider)."""
        if label in ("VICTORY", "LOVE_YOU"):
            return vocab.MOVEMENT_GESTURES[label]
        if self.mode == vocab.MODE_MUSIC:
            return vocab.MUSIC_GESTURES.get(label, vocab.UNKNOWN)
        return None

    def _fire(self, label: str, conf: float, size: float, now: float, confirming: bool):
        intent = self._special_intent(label)
        if intent is not None:
            return self._fire_special(intent, label, conf, size, now, confirming)
        if label == "OPEN_PALM":
            self._halt_media()
        t0 = self.clock()
        sense = self._read_sense(now)
        ctx = build_context(label, conf, sense, size)
        decision = self.decider.decide(ctx)
        requested = (self.decider.requested_action(ctx)
                     if hasattr(self.decider, "requested_action") else "")
        result = self.executor.execute(decision, moving=bool(ctx["moving"]), now=now)
        end = result.motion_ack_t if result.motion_ack_t is not None else self.clock()
        latency_ms = (end - t0) * 1000.0

        self.last_explain = ""
        if getattr(self.decider, "has_model", False):
            try:
                self.last_explain = self.decider.explain(ctx)
            except Exception as exc:  # explanation is optional, never fatal
                self.last_explain = f"Explicacion no disponible: {exc}"

        self._acted = label
        self._last_fire = now
        if decision.action == "ASK_CONFIRM" and not confirming:
            self.confirm.ask(label, now)
        else:
            self.confirm.clear()
        return self._record(
            label, conf, ctx, requested or "", decision,
            bool(sense.get("ok")) and result.ok,
            bool(sense.get("simulated")) or bool(sense.get("virtual_sonar")) or result.simulated,
            latency_ms, self.last_explain)

    def _fire_special(self, intent: str, label: str, conf: float, size: float, now: float,
                      confirming: bool):
        t0 = self.clock()
        self._acted = label
        if intent == vocab.UNKNOWN:  # a movement gesture with no meaning in MUSIC mode
            return None
        self._last_fire = now
        ask = conf < MIN_CONF or (self.confirm_ambiguous and label in AMBIGUOUS_GESTURES
                                  and not confirming)
        if ask:
            ctx = build_context(label, conf, self.sense or {"ok": False}, size)
            decision = validate("ASK_CONFIRM", ctx)
            result = self.executor.execute(decision, moving=False, now=now)
            self.confirm.ask(label, now)
            return self._record(label, conf, ctx, intent, decision, result.ok, result.simulated,
                                (self.clock() - t0) * 1000.0)
        self.confirm.clear()
        return self.handle_intent(intent, now, conf, label, size)

    def _record(self, label, conf, ctx, requested, decision, bridge_ok, simulated, latency_ms,
                explain: str = "") -> DecisionRecord:
        record = DecisionRecord(
            timestamp=datetime.now().isoformat(timespec="milliseconds"),
            gesture=label, confidence=float(conf), context=ctx, requested_intent=requested,
            decision=decision, bridge_ok=bool(bridge_ok), simulated=bool(simulated),
            latency_ms=latency_ms, explain=explain, mode=self.mode, input_mode=self.input_mode,
            bridge_error=self._bridge_error(),
        )
        self.last_record = record
        if self.logger is not None:
            self.logger.log(record)
        print(f"[{record.timestamp}] {label} ({conf:.2f}) -> {decision.action}  "
              f"[{decision.reason}]  {latency_ms:.0f} ms", flush=True)
        return record


class VirtualSonar:
    """Simulated sonars for a robot WITHOUT real ones (Choregraphe's virtual NAO).

    The virtual robot reports 0.0 m on both sonars, which the safety layer
    correctly treats as unreadable, so it never walks or turns. With
    ``--virtual-sonar`` the readings are replaced by these simulated values
    (keys 1/2/3/l/r place simulated obstacles). Posture and motion still come
    from the robot. Refused on the real robot: real sensors are never overridden.
    Every decision taken with simulated sonars is logged as ``simulated``.
    """

    KEYS = "123lr"
    TAG = "[sonar virtual]"
    KEYS_HELP = "Teclas: 1 libre | 2 40cm | 3 20cm | l/r lado | q salir"

    def __init__(self):
        self.front = MOCK_CLEAR
        self.left_blocked = False
        self.right_blocked = False

    def handle_key(self, key: str) -> bool:
        """Apply an obstacle key; returns False when ``key`` is not one of them."""
        if key == "1":
            self.front = MOCK_CLEAR
        elif key == "2":
            self.front = MOCK_NEAR
        elif key == "3":
            self.front = MOCK_BLOCKED
        elif key == "l":
            self.left_blocked = not self.left_blocked
        elif key == "r":
            self.right_blocked = not self.right_blocked
        else:
            return False
        return True

    def patch(self, reading: dict) -> dict:
        """Copy of ``reading`` with the simulated sonar values."""
        patched = dict(reading)
        patched["sonar_left"] = MOCK_BLOCKED if self.left_blocked else self.front
        patched["sonar_right"] = MOCK_BLOCKED if self.right_blocked else self.front
        patched["virtual_sonar"] = True
        return patched

    def describe(self) -> str:
        left = "BLOQ" if self.left_blocked else "libre"
        right = "BLOQ" if self.right_blocked else "libre"
        return f"Sonar virtual: frente {self.front * 100:.0f} cm | izq {left} | der {right}"


class MockScenario:
    """Simulated environment for ``--mock`` (keys 1/2/3/l/r/s), applied via ``set_mock``."""

    KEYS = "123lrs"
    TAG = "[mock]"
    KEYS_HELP = "Teclas: 1 libre | 2 40cm | 3 20cm | l/r lado | s sentar | q salir"

    def __init__(self, client):
        self.client = client
        self.front = MOCK_CLEAR
        self.left_blocked = False
        self.right_blocked = False
        self.sitting = False

    @property
    def posture(self) -> str:
        return MOCK_SIT if self.sitting else MOCK_STAND

    def apply(self) -> None:
        self.client.set_mock(
            sonar_left=MOCK_BLOCKED if self.left_blocked else self.front,
            sonar_right=MOCK_BLOCKED if self.right_blocked else self.front,
            posture=self.posture,
        )

    def handle_key(self, key: str) -> bool:
        """Apply a scenario key; returns False when ``key`` is not one of them."""
        if key == "1":
            self.front = MOCK_CLEAR
        elif key == "2":
            self.front = MOCK_NEAR
        elif key == "3":
            self.front = MOCK_BLOCKED
        elif key == "l":
            self.left_blocked = not self.left_blocked
        elif key == "r":
            self.right_blocked = not self.right_blocked
        elif key == "s":
            self.sitting = not self.sitting
        else:
            return False
        self.apply()
        return True

    def describe(self) -> str:
        left = "BLOQ" if self.left_blocked else "libre"
        right = "BLOQ" if self.right_blocked else "libre"
        return (f"Escenario: frente {self.front * 100:.0f} cm | izq {left} | der {right} "
                f"| postura {self.posture}")


def shutdown(client, music: MusicPlayer | None = None) -> None:
    """Always try to stop the robot, stop music, turn the eyes blue and close the bridge. Never raises.

    Stiffness is never released.
    """
    if music is not None:
        try:
            music.stop()
        except Exception:
            pass
    for label, call in (("stop", lambda: client.stop()), ("eyes", lambda: client.eyes("blue"))):
        try:
            call()
        except Exception as exc:  # the bridge may already be dead
            print(f"WARNING: could not send {label} on shutdown: {exc}", file=sys.stderr)
    try:
        client.close()
    except Exception as exc:
        print(f"WARNING: error closing the bridge: {exc}", file=sys.stderr)


def run_session(client, loop, music: MusicPlayer | None = None):
    """Run ``loop()``; whatever happens, stop the robot, blue eyes and close the bridge."""
    try:
        return loop()
    finally:
        shutdown(client, music)


def load_context_decider(model_path: str):
    """``ContextDecider`` from the model file, or the intent rules + safety with a warning."""
    from .context.decider import ContextDecider
    if os.path.isfile(model_path):
        return ContextDecider(model_path)
    print(f"WARNING: context model '{model_path}' not found. Using the intent rules + safety "
          "layer (collect scenarios and run brain.context.train to use the learned model).",
          file=sys.stderr)
    return ContextDecider(None)


# --------------------------------------------------------------------- overlay
def _cm(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "--"
    return "--" if math.isnan(value) else f"{value * 100:.0f}"


def overlay_lines(pilot: Pilot, scenario: MockScenario | None = None,
                  now: float | None = None) -> list[str]:
    """On-screen text (Spanish, ASCII only because OpenCV fonts lack accents)."""
    now = pilot.clock() if now is None else now
    sense = pilot.sense or {}
    lines = [
        f"Gesto: {pilot.gesture} ({pilot.confidence:.2f})",
        f"Modo: {pilot.mode} | Entrada: {pilot.input_mode}",
        "Musica: " + (ascii_text(pilot.dj.current_track or "--") if pilot.dj is not None else "--")
        + (" | baile" if pilot.dance.active else ""),
        f"Sonar I/D: {_cm(sense.get('sonar_left'))} / {_cm(sense.get('sonar_right'))} cm",
        f"Postura: {sense.get('posture', '--')} | Moviendo: "
        + ("SI" if sense.get("moving") else "NO"),
    ]
    record = pilot.last_record
    if record is not None:
        lines.append(f"Ultima accion: {record.decision.action}")
        lines.append("Motivo: " + ascii_text(record.decision.reason))
    pending = pilot.confirm.pending_gesture
    if pending is not None:
        lines.append(f"Confirmar: repite {pending} ({pilot.confirm.remaining(now):.1f}s)")
    if pilot.input_mode == vocab.INPUT_VOICE:
        lines.append("Voz: " + ascii_text(pilot.voice_status or "pulsa 'v' para grabar/enviar"))
    elif pilot.voice_status:
        lines.append("Voz: " + ascii_text(pilot.voice_status))
    remaining = pilot.cooldown_remaining(now)
    lines.append(f"Espera: {remaining:.1f}s" if remaining > 0 else "Listo para ordenes")
    if scenario is not None:
        lines.append(f"{scenario.TAG} {scenario.describe()}")
        lines.append(scenario.KEYS_HELP)
    return lines


def explain_lines(pilot: Pilot, max_lines: int = 4) -> list[str]:
    """First lines of the decision-tree explanation (empty without a model)."""
    if not pilot.last_explain:
        return []
    return [ascii_text(line) for line in pilot.last_explain.splitlines()[:max_lines]]


# ------------------------------------------------------------------ camera I/O
def draw_overlay(cv2, display, pilot: Pilot, scenario: MockScenario | None, now: float) -> None:
    for i, text in enumerate(overlay_lines(pilot, scenario, now)):
        cv2.putText(display, text, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 0), 2)
    height, width = display.shape[:2]
    for i, text in enumerate(explain_lines(pilot)):
        cv2.putText(display, text, (10, height - 50 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1)
    color = pilot.last_record.decision.eye_color if pilot.last_record else pilot.eye_color
    cv2.circle(display, (width - 40, 40), 22, EYE_BGR.get(color, EYE_BGR["blue"]), -1)
    cv2.putText(display, "Ojos", (width - 62, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    fraction = (pilot.cooldown_remaining(now) / pilot.cooldown) if pilot.cooldown > 0 else 0.0
    cv2.rectangle(display, (10, height - 12), (210, height - 4), (80, 80, 80), 1)
    cv2.rectangle(display, (10, height - 12), (10 + int(200 * fraction), height - 4),
                  (0, 165, 255), -1)


def camera_loop(args, pilot: Pilot, scenario: MockScenario | None) -> int:
    # Heavy imports are deferred so --help and the pure logic stay light.
    import cv2
    import mediapipe as mp
    import numpy as np

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: no se puede abrir la camara {args.camera}", file=sys.stderr)
        return 1
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    window = "NAO gestos de mano"
    hint = "Ctrl+C para salir" if args.no_video else "'q' sale"
    if scenario is not None and not args.no_video:
        hint += f", teclas {'/'.join(scenario.KEYS)} simulan el entorno"
    if not args.no_video:
        hint += ", 'v' graba/envia la voz (entrada VOZ)"
    print(f"Listo. {hint}.", flush=True)
    try:
        with mp_hands.Hands(max_num_hands=2, model_complexity=1, min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("ERROR: no se pudo leer de la camara", file=sys.stderr)
                    return 1
                result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                hands_found = []  # up to two (landmarks, handedness)
                for i, hand in enumerate(result.multi_hand_landmarks or []):
                    points = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=float)
                    side = result.multi_handedness[i].classification[0].label
                    hands_found.append((points, side))
                    if not args.no_video:
                        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                landmarks, handedness = hands_found[0] if hands_found else (None, None)
                pilot.step(landmarks, handedness, second=hands_found[1] if len(hands_found) > 1
                           else None)
                if args.no_video:
                    continue
                display = cv2.flip(frame, 1)  # mirrored preview (model input stays raw)
                draw_overlay(cv2, display, pilot, scenario, pilot.clock())
                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    return 0
                if key == ord("v"):
                    pilot.toggle_mic()
                    continue
                if scenario is not None and key != 255 and scenario.handle_key(chr(key)):
                    print(scenario.describe(), flush=True)
    finally:
        cap.release()
        if not args.no_video:
            cv2.destroyAllWindows()


# ------------------------------------------------------------------------ CLI
def configured_endpoint(config: Config, environ: dict) -> tuple:
    """(ip, port) from NAO_IP / NAO_PORT only when they are really set (env or ``.env``).

    The Config defaults (127.0.0.1) must never override a robot profile's own IP.
    """
    return (config.nao_ip if "NAO_IP" in environ else None,
            config.nao_port if "NAO_PORT" in environ else None)


def build_parser(config: Config | None = None) -> argparse.ArgumentParser:
    config = config or Config()
    p = argparse.ArgumentParser(
        prog="python -m brain.main",
        description="NAO controlled by hand gestures or voice: webcam -> gesture -> context "
                    "decision (sonars, posture, motion) -> robot. Modes: MOVEMENT (default) and "
                    "MUSIC; input: GESTURES (default) or VOICE. Two thumbs up held ~1 s toggle the "
                    "mode, VICTORY toggles the input, OPEN_PALM is always an emergency stop; in "
                    "VOICE input press 'v' in the preview to start/stop recording. "
                    "NAO_IP/NAO_PORT/MUSIC_DIR/CHAMPIONS_SONG/GROQ_API_KEY come from .env.",
    )
    p.add_argument("--robot", choices=("zyra", "nyx", "sim"), default=None,
                   help="robot profile for the bridge (required unless --mock)")
    p.add_argument("--mock", action="store_true",
                   help="bridge without naoqi/robot; keys 1/2/3/l/r/s simulate the environment")
    p.add_argument("--ip", default=None, help="override the robot IP (e.g. 127.0.0.1 for Choregraphe)")
    p.add_argument("--port", type=int, default=None, help="override the robot port")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--gesture-model", default=os.path.join("models", "gesture_model.joblib"))
    p.add_argument("--context-model", default=os.path.join("models", "context_model.joblib"),
                   help="if missing, the intent rules + safety layer are used (with a warning)")
    p.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN,
                   help="seconds between robot commands (default %(default)s)")
    p.add_argument("--confirm-window", type=float, default=DEFAULT_CONFIRM_WINDOW,
                   help="seconds to repeat a gesture after a confirmation request "
                        "(default %(default)s)")
    p.add_argument("--virtual-sonar", action="store_true",
                   help="ONLY for Choregraphe's virtual robot (no real sonars): replace its 0.0 m "
                        "sonar readings with simulated ones; keys 1/2/3/l/r place obstacles. "
                        "Refused with --robot zyra and with --mock")
    p.add_argument("--music-dir", default=config.music_dir,
                   help="folder with the songs for MUSIC mode (default: MUSIC_DIR from .env, "
                        "else %(default)s)")
    p.add_argument("--champions-song", default=config.champions_song or None,
                   help="audio file played by the celebrate gesture/intent (default: "
                        "CHAMPIONS_SONG from .env)")
    p.add_argument("--confirm-ambiguous", action="store_true",
                   help="ask for a repeat before acting on LOVE_YOU or VICTORY (often confused "
                        "with each other; default off)")
    p.add_argument("--no-voice", action="store_true",
                   help="do not build the push-to-talk voice listener (GROQ_API_KEY; without a "
                        "key only the keyword fallback exists)")
    p.add_argument("--log", default=None, help="decision CSV path (default runs/<timestamp>.csv)")
    p.add_argument("--no-video", action="store_true", help="do not open the preview window")
    return p


# Default bridge profile IPs (same values as ROBOTS in bridge/nao_bridge.py),
# used only to keep --virtual-sonar away from physical robots.
PROFILE_IPS = {"nyx": "127.0.0.1", "zyra": "192.168.108.90", "sim": "10.190.103.36"}
LOCAL_HOSTS = {"127.0.0.1", "localhost"}


def resolved_robot_ip(args) -> str:
    """IP the bridge will connect to: ``--ip`` if given, else the profile's IP."""
    return args.ip or PROFILE_IPS.get(args.robot or "", "")


def build_voice(pilot: Pilot, config: Config):
    """Push-to-talk listener posting intents on the pilot's queue; None if it cannot be built."""
    try:
        from .voice.groq_client import GroqClient
        from .voice.listener import VoiceListener
        groq = GroqClient(config.groq_api_key) if config.has_groq else None
        if groq is None:
            print("AVISO: sin GROQ_API_KEY la voz no puede transcribir (solo el respaldo por "
                  "palabras clave sobre texto).", file=sys.stderr)
        return VoiceListener(pilot.post_voice, client=groq)
    except Exception as exc:
        print(f"AVISO: voz no disponible: {exc}", file=sys.stderr)
        return None


def main(argv: list[str] | None = None, config: Config | None = None) -> int:
    config = config or load_config()
    parser = build_parser(config)
    args = parser.parse_args(argv)
    environ = {**parse_env_file(DEFAULT_ENV_PATH), **os.environ}
    cfg_ip, cfg_port = configured_endpoint(config, environ)
    args.ip = args.ip or cfg_ip
    args.port = args.port or cfg_port
    if args.robot is None and not args.mock:
        parser.error("--robot is required unless --mock is given")
    if args.virtual_sonar and args.mock:
        parser.error("--virtual-sonar is not needed with --mock (mock already simulates the sonars)")
    if args.virtual_sonar and args.robot == "zyra":
        parser.error("--virtual-sonar is refused on the real robot: its real sonars are never overridden")
    if args.virtual_sonar and resolved_robot_ip(args) not in LOCAL_HOSTS:
        parser.error("--virtual-sonar only works with a virtual robot on this computer "
                     "(127.0.0.1, e.g. Choregraphe); got %s" % resolved_robot_ip(args))
    for stream in (sys.stdout, sys.stderr):  # Spanish text must not crash legacy consoles
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    if not os.path.isfile(args.gesture_model):
        print(f"ERROR: no se encontro el modelo de gestos '{args.gesture_model}'. Entrenalo primero: "
              "python -m brain.gestures.train --data data/gestures --out models/gesture_model.joblib",
              file=sys.stderr)
        return 2

    from .gestures.predictor import GesturePredictor
    from .music import DJPlayer
    from .nao_client import NaoBridgeError, NaoClient

    predictor = GesturePredictor(args.gesture_model, min_conf=PREDICTOR_MIN_CONF)
    predictor2 = GesturePredictor(args.gesture_model, min_conf=PREDICTOR_MIN_CONF)
    decider = load_context_decider(args.context_model)
    logger = DecisionLogger(args.log or default_log_path())
    print(f"Registro de decisiones: {logger.path}")
    if args.virtual_sonar:
        print("SONAR VIRTUAL: los sonares son SIMULADOS (robot virtual de Choregraphe). "
              "Teclas 1/2/3/l/r ponen obstaculos. No usar con un robot fisico.")
    elif not args.mock:
        print(f"ROBOT REAL ({args.robot}): despeja el piso y deja a alguien junto al NAO.")

    try:
        client = NaoClient(robot=args.robot or "sim", mock=args.mock, ip=args.ip, port=args.port)
    except NaoBridgeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        logger.close()
        return 1

    virtual = VirtualSonar() if args.virtual_sonar else None
    dj = DJPlayer(args.music_dir, args.champions_song)
    pilot = Pilot(client, predictor, decider, cooldown=args.cooldown,
                  confirm_window=args.confirm_window, logger=logger, sonar_override=virtual,
                  dj=dj, predictor2=predictor2, confirm_ambiguous=args.confirm_ambiguous)
    if not args.no_voice:
        pilot.voice = build_voice(pilot, config)
    scenario = MockScenario(client) if args.mock else virtual

    def loop() -> int:
        ping = client.ping()
        print(f"Puente listo (mock={ping.get('mock')}).")
        pilot.announce_state()
        return camera_loop(args, pilot, scenario)

    try:
        return run_session(client, loop, types.SimpleNamespace(stop=pilot.stop_media))
    except KeyboardInterrupt:
        print("Interrumpido.")
        return 0
    except NaoBridgeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        logger.close()


if __name__ == "__main__":
    sys.exit(main())
