"""Integration entry point: webcam gesture -> context decision -> NAO action.

Run (from the project root, inside ``.venv311``)::

    python -m brain.main --robot zyra
    python -m brain.main --robot sim --mock          # no robot, no naoqi

Per frame: MediaPipe Pose -> ``GesturePredictor`` (majority vote). When a NEW
stable gesture (not ``NONE``) appears and the cooldown has elapsed, the robot is
sensed (sonars + motion state), the context feature vector is built, the
``ContextDecider`` (learned model + hard safety layer) picks the action and the
action is sent to the bridge. Every decision is appended to a CSV log.

If ``models/context_model.joblib`` does not exist, a documented rule-based
policy (``RuleBasedDecider``) plus the same safety layer is used instead, with a
warning, so gestures can be demoed before the context data is collected.

Pure logic (``build_features``, ``RuleBasedDecider``, ``Pilot``,
``DecisionLogger``, ``safe_shutdown``) lives apart from the camera I/O in
``camera_loop`` so it can be tested without a webcam or a robot.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime

from .context import FEATURE_COLUMNS
from .context.safety import apply_safety
from .gestures.features import person_distance

NONE_LABEL = "NONE"
DEFAULT_COOLDOWN = 2.5  # seconds between two robot commands
FAR_PERSON = 0.08  # shoulder width / frame width below this -> person too far
SENSE_INTERVAL = 0.5  # seconds between sonar refreshes for the overlay
MOCK_BLOCKED = (0.2, 0.2)  # simulated obstacle (m)
MOCK_CLEAR = (1.5, 1.5)  # simulated free path (m)
LOG_COLUMNS = (
    "timestamp", "gesture", "confidence", "sonar_left", "sonar_right",
    "person_distance", "moving", "action", "reason", "latency_ms",
)


# ------------------------------------------------------------------ pure logic
def build_features(gesture: str, confidence: float, sense: dict, distance: float) -> dict:
    """Context feature dict; keys are exactly ``brain.context.FEATURE_COLUMNS``."""
    return {
        "gesture": gesture,
        "gesture_confidence": float(confidence),
        "sonar_left": float(sense["sonar_left"]),
        "sonar_right": float(sense["sonar_right"]),
        "person_distance": float(distance),
        "moving": int(bool(sense["moving"])),
    }


class RuleBasedDecider:
    """Fallback policy used when no trained context model is available.

    Rules, in order:
      1. gesture ``NONE``                      -> ``WAIT``
      2. ``person_distance`` < 0.08 (too far)  -> ``ASK_CLOSER``
      3. otherwise the gesture's own action (ADVANCE / STOP / TURN_LEFT / TURN_RIGHT)
      4. the hard safety layer (ADVANCE with an obstacle < 0.30 m -> WARN_OBSTACLE)
    """

    name = "rule-based fallback"

    def decide_with_reason(self, features: dict) -> tuple[str, str]:
        gesture = features["gesture"]
        if gesture == NONE_LABEL:
            return "WAIT", f"{self.name}: no gesture"
        if features["person_distance"] < FAR_PERSON:
            return "ASK_CLOSER", f"{self.name}: person too far"
        proposed = gesture  # gesture labels double as action names
        final = apply_safety(proposed, features["sonar_left"], features["sonar_right"])
        if final != proposed:
            return final, f"safety override: {self.name} said {proposed} but obstacle/invalid sonar"
        return final, f"{self.name}: gesture {gesture}"


def load_decider(model_path: str):
    """``ContextDecider`` if the model file exists, else the rule-based fallback."""
    if os.path.isfile(model_path):
        from .context.decider import ContextDecider
        return ContextDecider(model_path)
    print(
        f"WARNING: context model '{model_path}' not found. Using the rule-based "
        "fallback policy + safety layer (collect scenarios and run "
        "brain.context.train to use the learned model).",
        file=sys.stderr,
    )
    return RuleBasedDecider()


@dataclass
class Decision:
    timestamp: str
    gesture: str
    confidence: float
    sonar_left: float
    sonar_right: float
    person_distance: float
    moving: int
    action: str
    reason: str
    latency_ms: float
    ack: dict = field(default_factory=dict)

    def row(self) -> list:
        return [
            self.timestamp, self.gesture, f"{self.confidence:.3f}",
            f"{self.sonar_left:.3f}", f"{self.sonar_right:.3f}",
            f"{self.person_distance:.3f}", self.moving, self.action,
            self.reason, f"{self.latency_ms:.1f}",
        ]


def default_log_path() -> str:
    return os.path.join("runs", datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv")


class DecisionLogger:
    """Appends one CSV row per decision (flushed immediately)."""

    def __init__(self, path: str):
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        self.path = path
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(LOG_COLUMNS)
        self._fh.flush()

    def log(self, decision: Decision) -> None:
        self._writer.writerow(decision.row())
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


class Pilot:
    """Gesture-to-action state machine (no camera, no I/O besides the client).

    A command fires when the stable gesture is not ``NONE``, differs from the
    gesture that last produced a command (it resets whenever ``NONE`` is seen),
    and ``cooldown`` seconds have passed since the previous command. A gesture
    that is held keeps firing only once; one suppressed by the cooldown fires as
    soon as the cooldown ends if it is still being held.

    ``latency_ms`` measures the decision pipeline from the moment the command
    is triggered (stable gesture, cooldown over) to the bridge's act ack:
    sense + feature build + decide + act.
    """

    def __init__(self, client, predictor, decider, cooldown: float = DEFAULT_COOLDOWN,
                 clock=time.perf_counter, logger: DecisionLogger | None = None,
                 sense_interval: float | None = SENSE_INTERVAL):
        self.client = client
        self.predictor = predictor
        self.decider = decider
        self.cooldown = cooldown
        self.clock = clock
        self.logger = logger
        self.sense_interval = sense_interval
        self.gesture = NONE_LABEL
        self.confidence = 0.0
        self.sense: dict | None = None
        self.last_decision: Decision | None = None
        self._acted: str | None = None
        self._last_fire: float | None = None
        self._last_sense_t: float | None = None

    def cooldown_remaining(self, now: float | None = None) -> float:
        if self._last_fire is None:
            return 0.0
        now = self.clock() if now is None else now
        return max(0.0, self.cooldown - (now - self._last_fire))

    def _do_sense(self, now: float) -> dict:
        reading = self.client.sense()
        if not reading.get("ok"):
            from .nao_client import NaoBridgeError
            raise NaoBridgeError(f"sense failed: {reading.get('error')}")
        self.sense = reading
        self._last_sense_t = now
        return reading

    def _refresh_sense(self, now: float) -> None:
        if self.sense_interval is None:
            return
        if self._last_sense_t is None or now - self._last_sense_t >= self.sense_interval:
            self._do_sense(now)

    def step(self, landmarks, now: float | None = None) -> Decision | None:
        """Process one frame (``landmarks`` is a (33, 4) array or None)."""
        now = self.clock() if now is None else now
        if landmarks is None:
            if hasattr(self.predictor, "reset"):
                self.predictor.reset()
            label, conf, distance = NONE_LABEL, 0.0, 0.0
        else:
            label, conf = self.predictor.update(landmarks)
            distance = person_distance(landmarks)
        self.gesture, self.confidence = label, conf
        self._refresh_sense(now)

        if label == NONE_LABEL:
            self._acted = None
            return None
        if label == self._acted or self.cooldown_remaining(now) > 0.0:
            return None
        return self._fire(label, conf, distance, now)

    def _fire(self, label: str, conf: float, distance: float, now: float) -> Decision:
        t0 = self.clock()
        sense = self._do_sense(now)
        features = build_features(label, conf, sense, distance)
        action, reason = self.decider.decide_with_reason(features)
        ack = self.client.act(action)
        latency_ms = (self.clock() - t0) * 1000.0
        if not ack.get("ok"):
            reason += f" | act failed: {ack.get('error')}"
        decision = Decision(
            timestamp=datetime.now().isoformat(timespec="milliseconds"),
            gesture=label, confidence=float(conf),
            sonar_left=features["sonar_left"], sonar_right=features["sonar_right"],
            person_distance=features["person_distance"], moving=features["moving"],
            action=action, reason=reason, latency_ms=latency_ms, ack=ack,
        )
        self._acted = label
        self._last_fire = now
        self.last_decision = decision
        if self.logger is not None:
            self.logger.log(decision)
        print(f"[{decision.timestamp}] {label} ({conf:.2f}) -> {action}  [{reason}]  "
              f"{latency_ms:.0f} ms")
        return decision


class MockObstacle:
    """Toggles a simulated obstacle in the mock bridge (``m`` key)."""

    def __init__(self, client):
        self.client = client
        self.blocked = False

    def toggle(self) -> bool:
        self.blocked = not self.blocked
        left, right = MOCK_BLOCKED if self.blocked else MOCK_CLEAR
        self.client.set_mock(left, right)
        return self.blocked


def safe_shutdown(client) -> None:
    """Always try to STOP the robot, then close the bridge. Never raises."""
    try:
        client.act("STOP")
    except Exception as exc:  # the bridge may already be dead
        print(f"WARNING: could not send STOP on shutdown: {exc}", file=sys.stderr)
    try:
        client.close()
    except Exception as exc:
        print(f"WARNING: error closing the bridge: {exc}", file=sys.stderr)


def run_session(client, loop):
    """Run ``loop()``; whatever happens, STOP the robot and close the bridge."""
    try:
        return loop()
    finally:
        safe_shutdown(client)


# ------------------------------------------------------------------ camera I/O
def overlay_lines(pilot: Pilot, mock_obstacle: MockObstacle | None = None,
                  now: float | None = None) -> list[str]:
    """On-screen text (Spanish, ASCII only because OpenCV fonts lack accents)."""
    sense = pilot.sense
    if sense:
        sonar = f"Sonar I/D: {sense['sonar_left']:.2f} / {sense['sonar_right']:.2f} m"
    else:
        sonar = "Sonar I/D: -- / -- m"
    lines = [f"Gesto: {pilot.gesture} ({pilot.confidence:.2f})", sonar]
    last = pilot.last_decision
    if last is not None:
        lines.append(f"Ultima accion: {last.action}")
        lines.append(f"Motivo: {last.reason}")
    remaining = pilot.cooldown_remaining(now)
    lines.append(f"Espera: {remaining:.1f}s" if remaining > 0 else "Listo para ordenes")
    if mock_obstacle is not None:
        lines.append("[m] obstaculo simulado: " + ("SI" if mock_obstacle.blocked else "NO"))
    return lines


def landmarks_to_array(pose_landmarks):
    import numpy as np
    return np.array([[p.x, p.y, p.z, p.visibility] for p in pose_landmarks.landmark], dtype=float)


def camera_loop(args, pilot: Pilot, mock_obstacle: MockObstacle | None) -> int:
    # Heavy imports are deferred so --help and the pure logic stay light.
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {args.camera}", file=sys.stderr)
        return 1
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    window = "NAO gesture pilot"
    print("Running. 'q' quits" + (", 'm' toggles a simulated obstacle." if mock_obstacle
                                  else ".") + (" (no video: use Ctrl+C to quit)" if args.no_video else ""))
    try:
        with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as pose:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("ERROR: failed to read from camera", file=sys.stderr)
                    return 1
                result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                landmarks = (landmarks_to_array(result.pose_landmarks)
                             if result.pose_landmarks is not None else None)
                pilot.step(landmarks)
                if args.no_video:
                    continue
                if result.pose_landmarks is not None:
                    mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                display = cv2.flip(frame, 1)  # mirrored preview
                for i, text in enumerate(overlay_lines(pilot, mock_obstacle)):
                    cv2.putText(display, text, (10, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
                cv2.imshow(window, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    return 0
                if key == ord("m") and mock_obstacle is not None:
                    blocked = mock_obstacle.toggle()
                    print("Simulated obstacle:", "ON (0.2 m)" if blocked else "OFF (1.5 m)")
    finally:
        cap.release()
        if not args.no_video:
            cv2.destroyAllWindows()


# ------------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m brain.main",
        description="Gesture-controlled NAO: webcam -> gesture -> context decision -> robot.",
    )
    p.add_argument("--robot", choices=("zyra", "nyx", "sim"), default=None,
                   help="robot profile for the bridge (required unless --mock)")
    p.add_argument("--mock", action="store_true",
                   help="bridge without naoqi/robot; 'm' toggles a simulated obstacle")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    p.add_argument("--gesture-model", default=os.path.join("models", "gesture_model.joblib"))
    p.add_argument("--context-model", default=os.path.join("models", "context_model.joblib"),
                   help="if missing, a rule-based fallback policy is used (with a warning)")
    p.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN,
                   help="seconds between robot commands (default %(default)s)")
    p.add_argument("--log", default=None,
                   help="decision CSV path (default runs/<timestamp>.csv)")
    p.add_argument("--no-video", action="store_true", help="do not open the preview window")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.robot is None and not args.mock:
        parser.error("--robot is required unless --mock is given")
    robot = args.robot or "sim"

    if not os.path.isfile(args.gesture_model):
        print(f"ERROR: gesture model '{args.gesture_model}' not found. Train it first: "
              "python -m brain.gestures.train", file=sys.stderr)
        return 2

    from .gestures.predictor import GesturePredictor
    from .nao_client import NaoBridgeError, NaoClient

    predictor = GesturePredictor(args.gesture_model)
    decider = load_decider(args.context_model)
    logger = DecisionLogger(args.log or default_log_path())
    print(f"Decision log: {logger.path}")
    if not args.mock:
        print(f"REAL ROBOT mode ({robot}): clear the floor and keep someone next to NAO.")

    try:
        client = NaoClient(robot=robot, mock=args.mock)
    except NaoBridgeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        logger.close()
        return 1

    pilot = Pilot(client, predictor, decider, cooldown=args.cooldown, logger=logger)
    mock_obstacle = MockObstacle(client) if args.mock else None

    def loop() -> int:
        ping = client.ping()
        print(f"Bridge ready (mock={ping.get('mock')}).")
        return camera_loop(args, pilot, mock_obstacle)

    try:
        return run_session(client, loop)
    except KeyboardInterrupt:
        print("Interrupted.")
        return 0
    except NaoBridgeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        logger.close()


if __name__ == "__main__":
    sys.exit(main())
