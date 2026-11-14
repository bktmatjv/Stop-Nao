"""Mode x input state machine, music mode, dance loop, voice queue (no camera, no robot)."""
import csv
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from brain import main as brain_main  # noqa: E402
from brain import vocab  # noqa: E402
from brain.config import Config  # noqa: E402
from brain.context.decider import ContextDecider  # noqa: E402
from brain.main import DanceLoop, DecisionLogger, Pilot, overlay_lines  # noqa: E402

HAND = np.zeros((21, 3))
HAND[9] = (0.1, 0.0, 0.0)


class Client:
    def __init__(self, sonar=1.5):
        self.calls = []
        self.last_error = None
        self.reading = {"ok": True, "simulated": True, "sonar_left": sonar, "sonar_right": sonar,
                        "moving": False, "posture": "StandInit"}

    def _ack(self, *call):
        self.calls.append(call)
        return {"ok": True, "simulated": True}

    def sense(self):
        return dict(self.reading)

    def move(self, x, theta):
        return self._ack("move", x, theta)

    def stop(self):
        return self._ack("stop")

    def wave(self):
        return self._ack("wave")

    def dance(self, style="default"):
        return self._ack("dance", style)

    def blink(self):
        return self._ack("blink")

    def ears(self, on):
        return self._ack("ears", on)

    def posture(self, name):
        return self._ack("posture", name)

    def say(self, text):
        return self._ack("say", text)

    def eyes(self, color):
        return self._ack("eyes", color)

    def close(self):
        pass

    def names(self):
        return [c[0] for c in self.calls]

    def dances(self):
        return [c[1] for c in self.calls if c[0] == "dance"]


class DJ:
    def __init__(self, has_tracks=True):
        self.has_tracks = has_tracks
        self.playing = False
        self.paused = False
        self.log = []
        self.track = "a.mp3"

    def play(self):
        self.log.append("play")
        if not self.has_tracks:
            return False
        self.playing, self.paused = True, False
        return True

    def pause(self):
        self.log.append("pause")
        was = self.playing
        self.playing, self.paused = False, was
        return was

    def resume(self):
        return self.play()

    def next(self):
        self.log.append("next")
        return True

    def previous(self):
        self.log.append("previous")
        return True

    def stop(self):
        self.log.append("stop")
        self.playing = self.paused = False

    def play_champions(self):
        self.log.append("champions")
        self.playing, self.paused = True, False
        self.track = "champions.mp3"
        return True

    @property
    def is_playing(self):
        return self.playing

    @property
    def is_paused(self):
        return self.paused

    @property
    def current_track(self):
        return self.track if (self.playing or self.paused) else None


class Script:
    def __init__(self, script):
        self.script, self.i = list(script), 0

    def update(self, landmarks, handedness):
        item = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return item


class FakeVoice:
    def __init__(self, fail=False):
        self.fail, self.events, self.busy = fail, [], False

    def begin(self):
        if self.fail:
            raise RuntimeError("sin microfono")
        self.events.append("begin")

    def end(self):
        self.events.append("end")


def make(script, dj=None, script2=None, **kw):
    client = kw.pop("client", None) or Client()
    pilot = Pilot(client, Script(script), ContextDecider(None), sense_interval=None, dj=dj,
                  predictor2=Script(script2) if script2 else None, cooldown=2.5, **kw)
    return client, pilot


def feed(pilot, label_times, hand=HAND):
    rec = None
    for t in label_times:
        rec = pilot.step(hand, "Right", now=t) or rec
    return rec


# ----------------------------------------------------------------- input axis
def test_defaults():
    _, p = make([("NONE", 0.0)])
    assert (p.mode, p.input_mode) == (vocab.MODE_MOVEMENT, vocab.INPUT_GESTURES)


def test_victory_toggles_input_with_feedback_and_back():
    c, p = make([("VICTORY", 0.9), ("NONE", 0.0), ("VICTORY", 0.9)])
    p.step(HAND, "Right", now=0.0)
    assert p.input_mode == vocab.INPUT_VOICE
    assert ("ears", True) in c.calls and ("say", "Te escucho por voz") in c.calls
    assert ("eyes", "blue") in c.calls
    p.step(None, None, now=1.0)
    p.step(HAND, "Right", now=3.0)
    assert p.input_mode == vocab.INPUT_GESTURES
    assert ("ears", False) in c.calls and ("say", "Volvemos a los gestos") in c.calls
    assert "wave" not in c.names() and "dance" not in c.names()


def test_voice_input_ignores_other_gestures_but_not_estop():
    c, p = make([("VICTORY", 0.9), ("NONE", 0.0), ("THUMBS_UP", 0.95), ("NONE", 0.0),
                 ("OPEN_PALM", 0.95)], dj=DJ())
    p.step(HAND, "Right", now=0.0)
    p.step(None, None, now=0.1)
    c.calls.clear()
    p.step(HAND, "Right", now=3.0)  # THUMBS_UP ignored
    assert "move" not in c.names()
    p.step(None, None, now=3.1)
    p.step(HAND, "Right", now=3.2)  # OPEN_PALM inside the cooldown still stops instantly
    assert "stop" in c.names()


def test_estop_bypasses_cooldown_and_halts_dance_and_music():
    dj = DJ()
    c, p = make([("THUMBS_UP", 0.95), ("NONE", 0.0), ("OPEN_PALM", 0.95)], dj=dj)
    p.set_mode(vocab.MODE_MUSIC, 0.0)
    p.step(HAND, "Right", now=0.1)  # play
    assert p.dance.active and dj.playing
    p.step(None, None, now=0.2)
    c.calls.clear()
    p.step(HAND, "Right", now=0.3)  # palm, cooldown running
    assert "stop" in c.names()
    assert not p.dance.active and not dj.playing and dj.paused
    n = len(c.dances())
    p.tick(100.0)
    assert len(c.dances()) == n


def test_voice_intents_ignored_with_gesture_input_and_honoured_with_voice():
    c, p = make([("NONE", 0.0)])
    p.post_voice(vocab.WAVE, "saluda")
    p.step(None, None, now=0.0)
    assert "wave" not in c.names() and "dance" not in c.names()
    p.set_input(vocab.INPUT_VOICE, 0.1)
    p.post_voice(vocab.WAVE, "saluda")
    p.step(None, None, now=0.2)
    assert "wave" in c.names() or "dance" in c.names()


def test_voice_advance_still_goes_through_the_safety_layer():
    c, p = make([("NONE", 0.0)], client=Client(sonar=0.2))
    p.set_input(vocab.INPUT_VOICE, 0.0)
    p.post_voice(vocab.ADVANCE, "avanza")
    p.step(None, None, now=0.1)
    assert "move" not in c.names()
    assert p.last_record.decision.action == "WARN_OBSTACLE"


def test_voice_stop_is_an_estop_and_unknown_is_ignored():
    dj = DJ()
    c, p = make([("NONE", 0.0)], dj=dj)
    p.set_input(vocab.INPUT_VOICE, 0.0)
    c.calls.clear()
    p.post_voice(vocab.UNKNOWN, "blah")
    p.step(None, None, now=0.1)
    assert c.calls == []
    p.post_voice(vocab.STOP, "para")
    p.step(None, None, now=0.2)
    assert "stop" in c.names()


def test_voice_can_switch_mode_and_input_back():
    c, p = make([("NONE", 0.0)])
    p.set_input(vocab.INPUT_VOICE, 0.0)
    p.post_voice(vocab.SET_MODE_MUSIC, "modo musica")
    p.post_voice(vocab.SET_INPUT_GESTURES, "gestos")
    p.step(None, None, now=0.1)
    assert (p.mode, p.input_mode) == (vocab.MODE_MUSIC, vocab.INPUT_GESTURES)


# ------------------------------------------------------------------ mode axis
def test_two_hands_thumbs_up_toggles_mode_and_does_not_advance():
    c, p = make([("THUMBS_UP", 0.95)], script2=[("THUMBS_UP", 0.95)])
    for t in (0.0, 0.4, 0.8):
        p.step(HAND, "Right", now=t, second=(HAND, "Left"))
    assert p.mode == vocab.MODE_MOVEMENT and "move" not in c.names()
    p.step(HAND, "Right", now=1.1, second=(HAND, "Left"))
    assert p.mode == vocab.MODE_MUSIC
    assert ("eyes", "purple") in c.calls and ("say", "Modo música") in c.calls
    assert "move" not in c.names()
    # one hand keeps showing thumbs up right after: nothing fires (cooldown + edge)
    p.step(HAND, "Right", now=1.5)
    assert "move" not in c.names()


def test_two_hands_toggle_back_to_movement_stops_music_and_dance():
    dj = DJ()
    c, p = make([("THUMBS_UP", 0.95)], dj=dj, script2=[("THUMBS_UP", 0.95)])
    p.set_mode(vocab.MODE_MUSIC, 0.0)
    p.handle_intent(vocab.MUSIC_PLAY, 0.1)
    for t in (5.0, 5.5, 6.1):
        p.step(HAND, "Right", now=t, second=(HAND, "Left"))
    assert p.mode == vocab.MODE_MOVEMENT
    assert ("eyes", "blue") in c.calls and ("say", "Modo movimiento") in c.calls
    assert not p.dance.active and not dj.playing


def test_single_thumbs_up_without_second_hand_still_advances():
    c, p = make([("THUMBS_UP", 0.95)], script2=[("NONE", 0.0)])
    rec = p.step(HAND, "Right", now=0.0)
    assert rec.decision.action == "ADVANCE" and "move" in c.names()


# ------------------------------------------------------------------ music mode
def test_music_gesture_table_and_dance_loop_driven_by_ticks():
    dj = DJ()
    c, p = make([("THUMBS_UP", 0.95), ("NONE", 0.0), ("POINT_RIGHT", 0.95), ("NONE", 0.0),
                 ("POINT_LEFT", 0.95), ("NONE", 0.0), ("FIST", 0.95)], dj=dj)
    p.set_mode(vocab.MODE_MUSIC, 0.0)
    p.step(HAND, "Right", now=10.0)
    assert dj.log[-1] == "play" and c.dances() == ["loop_a"]
    p.tick(12.0)
    assert c.dances() == ["loop_a"]  # no overlap before ~4 s
    p.tick(14.1)
    p.tick(18.2)
    p.tick(22.3)
    p.tick(26.4)
    assert c.dances() == ["loop_a", "loop_b", "loop_c", "loop_d", "loop_a"]
    p.step(None, None, now=30.0)
    p.step(HAND, "Right", now=31.0)
    assert dj.log[-1] == "next" and p.dance.active  # dance continues
    p.step(None, None, now=32.0)
    p.step(HAND, "Right", now=34.0)
    assert dj.log[-1] == "previous"
    p.step(None, None, now=35.0)
    c.calls.clear()
    p.step(HAND, "Right", now=38.0)  # FIST
    assert dj.log[-1] == "stop" and not p.dance.active and "stop" in c.names()


def test_music_pause_stops_dance_and_play_resumes_it():
    dj = DJ()
    c, p = make([("THUMBS_UP", 0.95), ("NONE", 0.0)], dj=dj)
    p.set_mode(vocab.MODE_MUSIC, 0.0)
    p.handle_intent(vocab.MUSIC_PLAY, 1.0)
    p.handle_intent(vocab.MUSIC_PAUSE, 2.0)
    assert dj.paused and not p.dance.active and "stop" in c.names()
    p.handle_intent(vocab.MUSIC_PLAY, 3.0)
    assert dj.playing and p.dance.active


def test_dance_loop_stops_by_itself_when_music_is_not_playing():
    dj = DJ()
    c, p = make([("NONE", 0.0)], dj=dj)
    p.handle_intent(vocab.MUSIC_PLAY, 0.0)
    dj.playing = False
    p.tick(5.0)
    assert not p.dance.active


def test_play_without_tracks_does_not_dance():
    c, p = make([("NONE", 0.0)], dj=DJ(has_tracks=False))
    p.handle_intent(vocab.MUSIC_PLAY, 0.0)
    assert c.dances() == [] and not p.dance.active


def test_gesture_without_meaning_in_music_mode_never_moves_the_robot():
    """A label the MUSIC table does not map resolves to UNKNOWN: no robot command at all."""
    c, p = make([("UNMAPPED", 0.95)], dj=DJ())
    p.set_mode(vocab.MODE_MUSIC, 0.0)
    c.calls.clear()
    p.step(HAND, "Right", now=5.0)
    assert not set(c.names()) & {"move", "wave", "dance", "posture"}


@pytest.mark.parametrize("mode", [vocab.MODE_MOVEMENT, vocab.MODE_MUSIC])
def test_love_you_celebrates_in_both_modes(mode):
    dj = DJ()
    c, p = make([("LOVE_YOU", 0.95)], dj=dj)
    p.set_mode(mode, 0.0)
    p.handle_intent(vocab.MUSIC_PLAY, 0.5)
    c.calls.clear()
    p.step(HAND, "Right", now=1.0)
    assert dj.log[-1] == "champions"
    assert c.dances() == ["champions"] and not p.dance.active


# ------------------------------------------------------- S9 confirm ambiguous
@pytest.mark.parametrize("label", ["LOVE_YOU", "VICTORY"])
def test_confirm_ambiguous_on_asks_then_executes_on_repeat(label):
    dj = DJ()
    c, p = make([(label, 0.95)], dj=dj, confirm_ambiguous=True)
    rec = p.step(HAND, "Right", now=0.0)
    assert rec.decision.action == "ASK_CONFIRM"
    assert "champions" not in dj.log and p.input_mode == vocab.INPUT_GESTURES
    rec = p.step(HAND, "Right", now=1.0)
    assert rec.decision.action == label.replace("LOVE_YOU", "CELEBRATE").replace(
        "VICTORY", "TOGGLE_INPUT")


def test_confirm_ambiguous_off_acts_immediately():
    dj = DJ()
    _, p = make([("LOVE_YOU", 0.95)], dj=dj, confirm_ambiguous=False)
    assert p.step(HAND, "Right", now=0.0).decision.action == "CELEBRATE"


def test_confirm_ambiguous_is_on_by_default_in_the_cli():
    """LOVE_YOU <-> VICTORY confusion is measured high, so the CLI asks for a repeat."""
    assert brain_main.build_parser().parse_args(["--mock"]).confirm_ambiguous is True
    assert brain_main.build_parser().parse_args(
        ["--mock", "--no-confirm-ambiguous"]).confirm_ambiguous is False


def test_confirm_ambiguous_does_not_ask_for_other_gestures():
    _, p = make([("THUMBS_UP", 0.95)], confirm_ambiguous=True)
    assert p.step(HAND, "Right", now=0.0).decision.action == "ADVANCE"


# --------------------------------------------------------------------- blink
def test_blink_on_first_detection_is_rate_limited():
    c, p = make([("NONE", 0.0), ("THUMBS_UP", 0.95), ("NONE", 0.0), ("THUMBS_UP", 0.95)])
    for t in (0.0, 0.1, 0.2, 0.3):
        p.step(HAND, "Right", now=t)
    assert c.names().count("blink") == 1


# ---------------------------------------------------------------- mic / voice
def test_mic_toggle_only_in_voice_input_and_errors_are_reported():
    _, p = make([("NONE", 0.0)])
    p.voice = FakeVoice()
    assert p.toggle_mic() is False and p.voice.events == []
    p.set_input(vocab.INPUT_VOICE, 0.0)
    assert p.toggle_mic() and p.toggle_mic()
    assert p.voice.events == ["begin", "end"]
    p.voice = FakeVoice(fail=True)
    assert p.toggle_mic() is False
    assert "sin microfono" in p.voice_status
    assert any("sin microfono" in line for line in overlay_lines(p, None, now=0.0))


def test_post_voice_text_uses_keyword_fallback():
    c, p = make([("NONE", 0.0)])
    p.set_input(vocab.INPUT_VOICE, 0.0)
    p.post_voice_text("saluda")
    p.step(None, None, now=0.1)
    assert "wave" in c.names() or "dance" in c.names()


# ------------------------------------------------------------- log / overlay
def test_log_columns_and_bridge_error(tmp_path):
    path = tmp_path / "x.csv"
    logger = DecisionLogger(str(path))
    c, p = make([("VICTORY", 0.9)], logger=logger)
    c.last_error = "dance: boom"
    p.step(HAND, "Right", now=0.0)
    logger.close()
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert list(rows[0])[-3:] == ["mode", "input", "bridge_error"]
    assert rows[0]["mode"] == "MOVEMENT" and rows[0]["input"] == "VOICE"
    assert rows[0]["bridge_error"] == "dance: boom"


def test_overlay_shows_mode_input_and_track():
    dj = DJ()
    _, p = make([("NONE", 0.0)], dj=dj)
    p.set_mode(vocab.MODE_MUSIC, 0.0)
    p.handle_intent(vocab.MUSIC_PLAY, 0.1)
    text = "\n".join(overlay_lines(p, None, now=1.0))
    assert text.isascii()
    for needle in ("MUSIC", "GESTURES", "a.mp3"):
        assert needle in text


# ----------------------------------------------------------------- dance loop
def test_dance_loop_unit():
    c = Client()
    loop = DanceLoop(c, interval=4.0)
    loop.tick(0.0)
    assert c.dances() == []
    loop.start(0.0)
    loop.start(1.0)  # idempotent
    loop.tick(3.9)
    loop.tick(4.0)
    assert c.dances() == ["loop_a", "loop_b"]
    loop.stop()
    assert c.names()[-1] == "stop"
    loop.tick(99.0)
    assert c.dances() == ["loop_a", "loop_b"]


# ------------------------------------------------------------------------ CLI
def test_help_lists_new_flags(capsys):
    with pytest.raises(SystemExit):
        brain_main.main(["--help"])
    out = capsys.readouterr().out
    for flag in ("--confirm-ambiguous", "--music-dir", "--champions-song", "--no-voice"):
        assert flag in out


def test_cli_defaults_come_from_config_and_cli_wins():
    cfg = Config(nao_ip="10.0.0.5", nao_port=1234, music_dir="mysongs", champions_song="c.mp3")
    args = brain_main.build_parser(cfg).parse_args(["--mock"])
    assert (args.music_dir, args.champions_song) == ("mysongs", "c.mp3")
    args = brain_main.build_parser(cfg).parse_args(["--mock", "--music-dir", "x"])
    assert args.music_dir == "x"
    assert brain_main.configured_endpoint(cfg, {"NAO_IP": "10.0.0.5"}) == ("10.0.0.5", None)
    assert brain_main.configured_endpoint(cfg, {}) == (None, None)


# ------------------------------------------------ dance whenever a DJ track plays
@pytest.mark.parametrize("intent", [vocab.MUSIC_NEXT, vocab.MUSIC_PREV])
def test_next_and_previous_keep_the_loop_while_playing(intent):
    dj = DJ()
    c, p = make([("NONE", 0.0)], dj=dj)
    p.handle_intent(vocab.MUSIC_PLAY, 0.0)
    c.calls.clear()
    p.handle_intent(intent, 1.0)
    assert dj.playing and p.dance.active
    assert c.dances() == []  # the running loop is kept, not restarted on top of itself


@pytest.mark.parametrize("intent", [vocab.MUSIC_NEXT, vocab.MUSIC_PREV])
def test_next_and_previous_while_paused_stay_paused_without_dancing(intent):
    dj = DJ()
    c, p = make([("NONE", 0.0)], dj=dj)
    p.handle_intent(vocab.MUSIC_PLAY, 0.0)
    p.handle_intent(vocab.MUSIC_PAUSE, 1.0)
    c.calls.clear()
    p.handle_intent(intent, 2.0)
    assert dj.paused and not dj.playing
    assert not p.dance.active and c.dances() == []


@pytest.mark.parametrize("intent", [vocab.MUSIC_NEXT, vocab.MUSIC_PREV])
def test_next_and_previous_start_the_loop_if_a_track_is_playing_without_one(intent):
    dj = DJ()
    c, p = make([("NONE", 0.0)], dj=dj)
    dj.playing = True  # e.g. right after the champions song, whose dance is not a loop
    p.handle_intent(intent, 1.0)
    assert p.dance.active and c.dances() == ["loop_a"]


def test_celebrate_dances_champions_once_and_never_starts_the_loop():
    dj = DJ()
    c, p = make([("NONE", 0.0)], dj=dj)
    p.handle_intent(vocab.MUSIC_PLAY, 0.0)
    assert p.dance.active
    c.calls.clear()
    p.handle_intent(vocab.CELEBRATE, 1.0)
    assert c.dances() == ["champions"] and not p.dance.active
    p.tick(100.0)
    assert c.dances() == ["champions"]  # still no loop step afterwards


def test_pause_stop_and_estop_still_stop_the_loop():
    for intent in (vocab.MUSIC_PAUSE, vocab.MUSIC_STOP, vocab.STOP):
        c, p = make([("NONE", 0.0)], dj=DJ())
        p.handle_intent(vocab.MUSIC_PLAY, 0.0)
        p.handle_intent(intent, 1.0)
        assert not p.dance.active, intent
