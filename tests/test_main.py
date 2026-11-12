"""Tests for the integration loop (brain.main). No webcam and no real robot.

Pure logic runs against a recording ``FakeClient`` and a scripted predictor.
The scenario tests (star case, graded advance, turns, posture toggle) go
through ``NaoClient(mock=True)``: the real Python 2.7 bridge in --mock mode,
no naoqi and no robot.
"""
import csv
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from brain import main as brain_main  # noqa: E402
from brain.context import FEATURE_COLUMNS  # noqa: E402
from brain.context.decider import ContextDecider  # noqa: E402
from brain.context.safety import Decision  # noqa: E402
from brain.main import (  # noqa: E402
    ConfirmationTracker,
    DecisionLogger,
    Executor,
    MockScenario,
    Pilot,
    SpeechGate,
    build_context,
    overlay_lines,
    run_session,
    shutdown,
)
from brain.nao_client import NaoBridgeError, NaoClient  # noqa: E402

PYTHON27 = r"C:\Python27\python.exe"
needs_py27 = pytest.mark.skipif(not os.path.isfile(PYTHON27), reason="Python 2.7 not installed")

HAND = np.zeros((21, 3))
HAND[9] = (0.1, 0.0, 0.0)  # middle MCP -> hand_size 0.1 (a comfortable distance)
CLOSE_HAND = np.zeros((21, 3))
CLOSE_HAND[9] = (0.4, 0.0, 0.0)  # hand_size 0.4 -> person too close to wave


# --------------------------------------------------------------------- stubs
class ScriptedPredictor:
    """Returns the scripted (label, confidence) pairs, repeating the last one."""

    def __init__(self, script):
        self.script = list(script)
        self.i = 0
        self.seen = []

    def update(self, landmarks, handedness):
        self.seen.append((landmarks, handedness))
        item = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return item


class FakeClient:
    """Records every command as a tuple; optionally refuses the named commands."""

    def __init__(self, sonar_left=1.5, sonar_right=1.5, moving=False, posture="StandInit",
                 fail=(), sense_ok=True):
        self.sense_reading = {"ok": True, "simulated": True, "sonar_left": sonar_left,
                              "sonar_right": sonar_right, "moving": moving, "posture": posture}
        self.sense_ok = sense_ok
        self.fail = set(fail)
        self.calls = []
        self.closed = False

    def _ack(self, *call):
        self.calls.append(call)
        if call[0] in self.fail:
            return {"ok": False, "error": f"{call[0]} refused", "simulated": True}
        return {"ok": True, "simulated": True}

    def sense(self):
        self.calls.append(("sense",))
        if not self.sense_ok:
            return {"ok": False, "error": "sense failed"}
        return dict(self.sense_reading)

    def move(self, x, theta):
        return self._ack("move", x, theta)

    def stop(self):
        return self._ack("stop")

    def wave(self):
        return self._ack("wave")

    def posture(self, name):
        return self._ack("posture", name)

    def say(self, text):
        return self._ack("say", text)

    def eyes(self, color):
        return self._ack("eyes", color)

    def set_mock(self, **kwargs):
        return self._ack("set_mock", kwargs)

    def close(self):
        self.closed = True
        self.calls.append(("close",))

    def names(self):
        return [c[0] for c in self.calls]


class Recorder:
    """Proxy that records and delegates every client call."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = []

    def __getattr__(self, name):
        fn = getattr(self.inner, name)

        def wrapper(*args, **kwargs):
            self.calls.append((name, kwargs) if kwargs else (name,) + args)
            return fn(*args, **kwargs)

        return wrapper

    def names(self):
        return [c[0] for c in self.calls]


def make_pilot(client, script, decider=None, cooldown=2.5, confirm_window=4.0, logger=None):
    return Pilot(client, ScriptedPredictor(script), decider or ContextDecider(None),
                 cooldown=cooldown, confirm_window=confirm_window, logger=logger,
                 sense_interval=None)


def run_script(pilot, items):
    """Step the pilot through ``(time, landmarks)`` items; returns the records (None = idle)."""
    return [pilot.step(lm, "Right", now=t) for t, lm in items]


@pytest.fixture
def mock_client():
    client = NaoClient(mock=True)
    yield client
    client.close()


def decision(action="ADVANCE", params=None, reason="Razon.", eye="green"):
    return Decision(action, dict(params or {}), reason, eye)


# --------------------------------------------------------------- context dict
def test_context_keys_are_exactly_feature_columns():
    sense = {"ok": True, "sonar_left": 0.5, "sonar_right": 0.7, "moving": True, "posture": "Stand"}
    ctx = build_context("THUMBS_UP", 0.9, sense, 0.12)
    assert tuple(ctx) == tuple(FEATURE_COLUMNS)
    assert ctx["gesture"] == "THUMBS_UP"
    assert ctx["gesture_confidence"] == pytest.approx(0.9)
    assert ctx["sonar_left"] == pytest.approx(0.5) and ctx["sonar_right"] == pytest.approx(0.7)
    assert ctx["hand_size"] == pytest.approx(0.12)
    assert ctx["moving"] == 1 and ctx["standing"] == 1


@pytest.mark.parametrize("posture,standing", [
    ("Stand", 1), ("StandInit", 1), ("StandZero", 1), ("Sit", 0), ("Crouch", 0), ("Unknown", 0),
])
def test_standing_comes_from_posture(posture, standing):
    sense = {"ok": True, "sonar_left": 1.0, "sonar_right": 1.0, "moving": False, "posture": posture}
    assert build_context("FIST", 0.9, sense, 0.1)["standing"] == standing


def test_failed_sense_gives_unsafe_context():
    ctx = build_context("THUMBS_UP", 0.9, {"ok": False, "error": "x"}, 0.1)
    assert tuple(ctx) == tuple(FEATURE_COLUMNS)
    assert math.isnan(ctx["sonar_left"]) and math.isnan(ctx["sonar_right"])
    assert ctx["standing"] == 0


# ------------------------------------------------------------ scenarios (mock)
@needs_py27
def test_star_case_obstacle_blocks_advance(mock_client, tmp_path):
    rec = Recorder(mock_client)
    mock_client.set_mock(sonar_left=0.20, sonar_right=0.20)
    log_path = tmp_path / "log.csv"
    logger = DecisionLogger(str(log_path))
    pilot = make_pilot(rec, [("THUMBS_UP", 0.9)], logger=logger)
    record = pilot.step(HAND, "Right", now=0.0)
    logger.close()

    assert record.decision.action == "WARN_OBSTACLE"
    assert record.decision.eye_color == "red"
    assert record.decision.overridden is True
    assert "move" not in rec.names()  # nothing was sent that could move the robot
    assert ("eyes", "red") in rec.calls
    assert ("say", record.decision.reason) in rec.calls
    assert mock_client.sense()["moving"] is False

    rows = list(csv.DictReader(open(log_path, encoding="utf-8")))
    assert len(rows) == 1
    row = rows[0]
    assert row["gesture"] == "THUMBS_UP"
    assert row["requested_intent"] == "ADVANCE"
    assert row["action"] == "WARN_OBSTACLE"
    assert row["overridden"] == "1"
    assert row["eye_color"] == "red"
    assert row["bridge_ok"] == "1" and row["simulated"] == "1"
    assert float(row["sonar_left"]) == pytest.approx(0.20)
    assert float(row["latency_ms"]) >= 0.0


@needs_py27
def test_graded_advance_with_near_obstacle(mock_client):
    rec = Recorder(mock_client)
    mock_client.set_mock(sonar_left=0.40, sonar_right=0.40)
    record = make_pilot(rec, [("THUMBS_UP", 0.9)]).step(HAND, "Right", now=0.0)
    assert record.decision.action == "ADVANCE_SHORT"
    assert record.decision.eye_color == "yellow"
    move = [c for c in rec.calls if c[0] == "move"]
    assert len(move) == 1
    assert move[0][1] == pytest.approx(0.10, abs=1e-6) and move[0][2] == 0.0
    assert ("eyes", "yellow") in rec.calls


@needs_py27
def test_clear_path_advances_full_step(mock_client):
    rec = Recorder(mock_client)
    mock_client.set_mock(sonar_left=1.5, sonar_right=1.5)
    record = make_pilot(rec, [("THUMBS_UP", 0.9)]).step(HAND, "Right", now=0.0)
    assert record.decision.action == "ADVANCE"
    move = [c for c in rec.calls if c[0] == "move"]
    assert move == [("move", pytest.approx(0.35), 0.0)]
    assert ("eyes", "green") in rec.calls
    assert mock_client.sense()["moving"] is True


@needs_py27
def test_point_left_turns_robot_right_and_right_obstacle_blocks(mock_client):
    rec = Recorder(mock_client)
    record = make_pilot(rec, [("POINT_LEFT", 0.9)]).step(HAND, "Right", now=0.0)
    assert record.decision.action == "TURN_RIGHT"
    move = [c for c in rec.calls if c[0] == "move"]
    assert len(move) == 1 and move[0][1] == 0.0 and move[0][2] < 0  # negative theta = right

    rec2 = Recorder(mock_client)
    mock_client.stop()
    mock_client.set_mock(sonar_right=0.20)
    blocked = make_pilot(rec2, [("POINT_LEFT", 0.9)]).step(HAND, "Right", now=0.0)
    assert blocked.decision.action == "WARN_OBSTACLE"
    assert blocked.decision.eye_color == "red"
    assert "move" not in rec2.names()


@needs_py27
def test_point_right_turns_robot_left_with_positive_theta(mock_client):
    rec = Recorder(mock_client)
    record = make_pilot(rec, [("POINT_RIGHT", 0.9)]).step(HAND, "Right", now=0.0)
    assert record.decision.action == "TURN_LEFT"
    move = [c for c in rec.calls if c[0] == "move"]
    assert len(move) == 1 and move[0][1] == 0.0 and move[0][2] > 0


@needs_py27
def test_fist_toggles_sit_and_stand(mock_client):
    rec = Recorder(mock_client)
    pilot = make_pilot(rec, [("FIST", 0.9), ("NONE", 0.0), ("FIST", 0.9)])
    first = pilot.step(HAND, "Right", now=0.0)
    assert first.decision.action == "SIT"
    assert ("posture", "Sit") in rec.calls
    assert rec.names().index("stop") < rec.names().index("posture")  # stop before changing posture
    assert mock_client.sense()["posture"] == "Sit"

    assert pilot.step(None, None, now=1.0) is None
    second = pilot.step(HAND, "Right", now=3.0)
    assert second.decision.action == "STAND"
    assert ("posture", "StandInit") in rec.calls
    assert mock_client.sense()["posture"] == "StandInit"


@needs_py27
def test_voice_wave_waves_when_standing_and_still(mock_client):
    # VICTORY no longer waves (it toggles the input); WAVE stays reachable by voice.
    rec = Recorder(mock_client)
    class SilentMusic:  # never start real audio from a test
        def play_random(self):
            return None

        def stop(self):
            pass

    pilot = Pilot(rec, ScriptedPredictor([("NONE", 0.0)]), ContextDecider(None),
                  sense_interval=None, music=SilentMusic())
    record = pilot.handle_intent("WAVE", now=0.0)
    assert record.decision.action == "WAVE"
    assert ("dance" in rec.names() or "wave" in rec.names())
    assert record.bridge_ok is True


def test_voice_wave_with_person_too_close_warns_and_does_not_wave():
    client = FakeClient()
    record = make_pilot(client, [("NONE", 0.0)]).handle_intent("WAVE", now=0.0, size=0.4)
    assert record.decision.action == "WARN_TOO_CLOSE"
    assert "wave" not in client.names() and "dance" not in client.names()
    assert ("eyes", "red") in client.calls


# ---------------------------------------------------------------- confirmation
def test_low_confidence_asks_then_repeat_within_window_executes():
    client = FakeClient()
    pilot = make_pilot(client, [("THUMBS_UP", 0.45), ("NONE", 0.0), ("THUMBS_UP", 0.9)],
                       cooldown=2.5, confirm_window=4.0)
    ask = pilot.step(HAND, "Right", now=0.0)
    assert ask.decision.action == "ASK_CONFIRM"
    assert ask.decision.eye_color == "yellow"
    assert "move" not in client.names()
    assert pilot.confirm.pending_gesture == "THUMBS_UP"

    assert pilot.step(None, None, now=0.5) is None
    done = pilot.step(HAND, "Right", now=1.0)  # cooldown (2.5) not over: confirmation bypasses it
    assert done.decision.action == "ADVANCE"
    assert ("move", pytest.approx(0.35), 0.0) in client.calls
    assert pilot.confirm.pending_gesture is None


def test_held_gesture_confirms_when_confidence_rises():
    client = FakeClient()
    pilot = make_pilot(client, [("THUMBS_UP", 0.45), ("THUMBS_UP", 0.85)])
    assert pilot.step(HAND, "Right", now=0.0).decision.action == "ASK_CONFIRM"
    assert pilot.step(HAND, "Right", now=0.5).decision.action == "ADVANCE"


def test_confirmation_expires_after_window_and_does_not_execute():
    client = FakeClient()
    # Cooldown is longer than the window so only a valid confirmation could still fire.
    pilot = make_pilot(client, [("THUMBS_UP", 0.45), ("NONE", 0.0), ("THUMBS_UP", 0.9)],
                       cooldown=10.0, confirm_window=4.0)
    assert pilot.step(HAND, "Right", now=0.0).decision.action == "ASK_CONFIRM"
    assert pilot.step(None, None, now=1.0) is None
    assert pilot.step(HAND, "Right", now=5.0) is None  # window over, cooldown still running
    assert "move" not in client.names()
    assert pilot.confirm.pending_gesture is None
    assert ("eyes", "blue") in client.calls  # back to listening


def test_wrong_gesture_does_not_confirm():
    client = FakeClient()
    # (OPEN_PALM is the e-stop and bypasses the cooldown, so another gesture is used here)
    pilot = make_pilot(client, [("THUMBS_UP", 0.45), ("NONE", 0.0), ("POINT_LEFT", 0.9)],
                       cooldown=10.0)
    pilot.step(HAND, "Right", now=0.0)
    pilot.step(None, None, now=0.5)
    assert pilot.step(HAND, "Right", now=1.0) is None  # different gesture, still in cooldown
    assert "move" not in client.names()


def test_confirmation_tracker_state_machine():
    tracker = ConfirmationTracker(window=4.0)
    assert not tracker.matches("FIST", 0.9, 0.0)
    tracker.ask("FIST", now=10.0)
    assert tracker.matches("FIST", 0.6, 11.0)
    assert not tracker.matches("FIST", 0.59, 11.0)
    assert not tracker.matches("VICTORY", 0.9, 11.0)
    assert not tracker.expire(13.9)
    assert tracker.expire(14.1) is True
    assert tracker.pending_gesture is None
    assert tracker.expire(15.0) is False  # reported once


# --------------------------------------------------------- cooldown / triggers
def test_cooldown_suppresses_repeats_and_held_gesture_fires_once():
    client = FakeClient()
    pilot = make_pilot(client, [("THUMBS_UP", 0.9)] * 4 + [("NONE", 0.0), ("THUMBS_UP", 0.9),
                                                          ("NONE", 0.0), ("THUMBS_UP", 0.9)],
                       cooldown=2.5)
    records = run_script(pilot, [(0.0, HAND), (0.5, HAND), (1.0, HAND), (1.5, HAND)])
    assert [r is not None for r in records] == [True, False, False, False]  # held: one command
    assert pilot.step(None, None, now=1.6) is None
    assert pilot.step(HAND, "Right", now=2.0) is None  # new gesture, cooldown still running
    assert pilot.step(None, None, now=2.2) is None
    assert pilot.step(HAND, "Right", now=3.0) is not None  # cooldown over
    assert client.names().count("move") == 2


def test_cooldown_remaining_counts_down():
    client = FakeClient()
    pilot = make_pilot(client, [("THUMBS_UP", 0.9)], cooldown=2.5)
    assert pilot.cooldown_remaining(0.0) == 0.0
    pilot.step(HAND, "Right", now=10.0)
    assert pilot.cooldown_remaining(11.0) == pytest.approx(1.5)
    assert pilot.cooldown_remaining(20.0) == 0.0


def test_none_gesture_never_acts():
    client = FakeClient()
    pilot = make_pilot(client, [("NONE", 0.0), ("NONE", 0.3)])
    assert [pilot.step(HAND, "Right", now=t) for t in (0.0, 1.0, 2.0)] == [None, None, None]
    assert [pilot.step(None, None, now=t) for t in (3.0, 4.0)] == [None, None]
    assert set(client.names()) <= {"sense"}  # nothing was sent: no move/say/eyes/stop


def test_no_hand_is_forwarded_to_predictor_as_none():
    client = FakeClient()
    predictor = ScriptedPredictor([("NONE", 0.0)])
    pilot = Pilot(client, predictor, ContextDecider(None), sense_interval=None)
    pilot.step(None, None, now=0.0)
    assert predictor.seen == [(None, None)]


def test_hand_size_reaches_the_context():
    client = FakeClient()
    pilot = make_pilot(client, [("THUMBS_UP", 0.9)])
    record = pilot.step(HAND, "Right", now=0.0)
    assert record.context["hand_size"] == pytest.approx(0.1)
    assert tuple(record.context) == tuple(FEATURE_COLUMNS)


# ------------------------------------------------------------------- executor
@pytest.mark.parametrize("dec,moving,expected", [
    (decision("ADVANCE", {"x": 0.2}), False, [("move", 0.2, 0.0)]),
    (decision("ADVANCE_SHORT", {"x": 0.1}, eye="yellow"), False, [("move", 0.1, 0.0)]),
    (decision("TURN_LEFT", {"theta": 0.785}), False, [("move", 0.0, 0.785)]),
    (decision("TURN_RIGHT", {"theta": -0.785}), False, [("move", 0.0, -0.785)]),
    (decision("STOP"), True, [("stop",)]),
    (decision("WAVE"), False, [("wave",)]),
    (decision("SIT"), False, [("stop",), ("posture", "Sit")]),
    (decision("STAND"), False, [("stop",), ("posture", "StandInit")]),
    (decision("WARN_OBSTACLE", eye="red"), True, [("stop",)]),
    (decision("WARN_MUST_STOP", eye="red"), True, [("stop",)]),
    (decision("WARN_OBSTACLE", eye="red"), False, []),
    (decision("WARN_TOO_CLOSE", eye="red"), False, []),
    (decision("WARN_MUST_STAND", eye="yellow"), False, []),
    (decision("ASK_CONFIRM", eye="yellow"), True, []),
    (decision("WAIT", eye="blue"), True, []),
])
def test_executor_maps_actions_to_bridge_commands(dec, moving, expected):
    client = FakeClient()
    result = Executor(client).execute(dec, moving=moving, now=0.0)
    motion = [c for c in client.calls if c[0] not in ("eyes", "say")]
    assert motion == expected
    assert ("eyes", dec.eye_color) in client.calls
    assert ("say", dec.reason) in client.calls
    assert result.ok is True


def test_executor_stops_before_other_commands_for_safety_actions():
    client = FakeClient()
    Executor(client).execute(decision("WARN_OBSTACLE", eye="red"), moving=True, now=0.0)
    assert client.names()[0] == "stop"


def test_executor_reports_bridge_refusal_with_red_eyes_and_spoken_error():
    client = FakeClient(fail={"move"})
    result = Executor(client).execute(decision("ADVANCE", {"x": 0.2}), moving=False, now=0.0)
    assert result.ok is False
    assert result.errors and "move" in result.errors[0]
    assert client.calls[-2] == ("eyes", "red")
    last = client.calls[-1]
    assert last[0] == "say" and "No pude" in last[1]


def test_bridge_refusal_is_logged_as_not_ok(tmp_path):
    client = FakeClient(fail={"move"})
    log_path = tmp_path / "log.csv"
    logger = DecisionLogger(str(log_path))
    record = make_pilot(client, [("THUMBS_UP", 0.9)], logger=logger).step(HAND, "Right", now=0.0)
    logger.close()
    assert record.bridge_ok is False
    row = next(csv.DictReader(open(log_path, encoding="utf-8")))
    assert row["bridge_ok"] == "0"


def test_failed_sense_blocks_motion_but_stop_still_works():
    client = FakeClient(sense_ok=False)
    advance = make_pilot(client, [("THUMBS_UP", 0.9)]).step(HAND, "Right", now=0.0)
    assert advance.decision.action == "WARN_OBSTACLE"
    assert "move" not in client.names()
    stop = make_pilot(client, [("OPEN_PALM", 0.9)]).step(HAND, "Right", now=0.0)
    assert stop.decision.action == "STOP"
    assert ("stop",) in client.calls


# ------------------------------------------------------------------ speech gate
def test_speech_gate_dedups_identical_sentence_within_three_seconds():
    gate = SpeechGate(min_gap=3.0)
    assert gate.allow("Hola", now=0.0) is True
    assert gate.allow("Hola", now=2.9) is False
    assert gate.allow("Otra frase", now=3.0) is True  # a different sentence is always spoken
    assert gate.allow("Hola", now=3.1) is True
    assert gate.allow("Hola", now=6.2) is True  # more than 3 s later


def test_executor_does_not_repeat_the_same_sentence_within_three_seconds():
    client = FakeClient()
    executor = Executor(client)
    dec = decision("WARN_OBSTACLE", reason="Hay un obstaculo.", eye="red")
    executor.execute(dec, moving=False, now=0.0)
    executor.execute(dec, moving=False, now=1.0)
    assert client.calls.count(("say", "Hay un obstaculo.")) == 1
    assert client.calls.count(("eyes", "red")) == 2  # eyes are always updated
    executor.execute(dec, moving=False, now=3.5)
    assert client.calls.count(("say", "Hay un obstaculo.")) == 2


# ------------------------------------------------------------------- fallback
def test_missing_context_model_falls_back_to_rules(tmp_path, capsys):
    decider = brain_main.load_context_decider(str(tmp_path / "nope.joblib"))
    assert isinstance(decider, ContextDecider) and not decider.has_model
    assert "context" in capsys.readouterr().err.lower()
    client = FakeClient(sonar_left=0.2, sonar_right=0.2)
    record = make_pilot(client, [("THUMBS_UP", 0.9)], decider=decider).step(HAND, "Right", now=0.0)
    assert record.decision.action == "WARN_OBSTACLE"
    assert record.requested_intent == "ADVANCE"


def test_missing_gesture_model_exits_with_code_2(tmp_path, capsys):
    code = brain_main.main(["--mock", "--gesture-model", str(tmp_path / "missing.joblib"),
                            "--log", str(tmp_path / "log.csv")])
    assert code == 2
    assert "brain.gestures.train" in capsys.readouterr().err


def test_robot_or_mock_is_required(capsys):
    with pytest.raises(SystemExit) as exc:
        brain_main.main([])
    assert exc.value.code == 2


def test_help_lists_the_flags(capsys):
    with pytest.raises(SystemExit) as exc:
        brain_main.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--mock", "--robot", "--ip", "--port", "--camera", "--gesture-model",
                 "--context-model", "--cooldown", "--confirm-window", "--log", "--no-video"):
        assert flag in out


# ------------------------------------------------------------------- shutdown
def test_shutdown_sends_stop_and_blue_eyes_then_closes():
    client = FakeClient()
    shutdown(client)
    assert client.calls == [("stop",), ("eyes", "blue"), ("close",)]


def test_shutdown_survives_a_dead_bridge():
    class Dead(FakeClient):
        def stop(self):
            raise NaoBridgeError("dead")

        def eyes(self, color):
            raise NaoBridgeError("dead")

    client = Dead()
    shutdown(client)  # must not raise
    assert client.closed is True


def test_run_session_always_shuts_down_even_on_error():
    client = FakeClient()

    def boom():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        run_session(client, boom)
    assert ("stop",) in client.calls and ("eyes", "blue") in client.calls and client.closed


@needs_py27
def test_shutdown_with_the_real_mock_bridge():
    client = NaoClient(mock=True)
    rec = Recorder(client)
    shutdown(rec)
    assert rec.names()[:2] == ["stop", "eyes"] and rec.calls[1] == ("eyes", "blue")


# -------------------------------------------------------------- mock scenario
def test_mock_scenario_keys_drive_set_mock():
    client = FakeClient()
    scenario = MockScenario(client)
    assert scenario.handle_key("2") is True
    assert client.calls[-1] == ("set_mock", {"sonar_left": 0.40, "sonar_right": 0.40, "posture": "StandInit"})
    scenario.handle_key("3")
    assert client.calls[-1][1]["sonar_left"] == 0.20 and client.calls[-1][1]["sonar_right"] == 0.20
    scenario.handle_key("1")
    assert client.calls[-1][1]["sonar_left"] == 1.5 and client.calls[-1][1]["sonar_right"] == 1.5
    scenario.handle_key("l")
    assert client.calls[-1][1]["sonar_left"] == 0.20 and client.calls[-1][1]["sonar_right"] == 1.5
    scenario.handle_key("r")
    assert client.calls[-1][1]["sonar_right"] == 0.20
    scenario.handle_key("l")  # toggle back
    assert client.calls[-1][1]["sonar_left"] == 1.5
    scenario.handle_key("s")
    assert client.calls[-1][1]["posture"] == "Sit"
    scenario.handle_key("s")
    assert client.calls[-1][1]["posture"] == "StandInit"
    assert scenario.handle_key("x") is False
    assert "StandInit" in scenario.describe()


# --------------------------------------------------------------------- overlay
def test_overlay_is_ascii_spanish_and_complete():
    client = FakeClient(posture="Sit")
    pilot = make_pilot(client, [("THUMBS_UP", 0.45)])
    pilot.step(HAND, "Right", now=0.0)
    scenario = MockScenario(client)
    lines = overlay_lines(pilot, scenario, now=1.0)
    text = "\n".join(lines)
    assert text.isascii()
    for needle in ("THUMBS_UP", "0.45", "Sonar", "Sit", "ASK_CONFIRM", "Confirmar"):
        assert needle in text
    assert "Espera" in text  # cooldown running
    assert "Escenario" in text  # mock scenario shown
    assert overlay_lines(pilot, None, now=100.0)  # works without a mock scenario


def test_log_has_the_contract_columns(tmp_path):
    logger = DecisionLogger(str(tmp_path / "x.csv"))
    logger.close()
    header = next(csv.reader(open(tmp_path / "x.csv", encoding="utf-8")))
    assert header == ["timestamp", "gesture", "confidence", "sonar_left", "sonar_right",
                      "hand_size", "moving", "standing", "requested_intent", "action", "params",
                      "reason", "eye_color", "overridden", "bridge_ok", "simulated", "latency_ms",
                      "mode", "input", "bridge_error"]


# ------------------------------------------------- virtual sonar (Choregraphe)
def _virtual_pilot(virtual):
    client = FakeClient(sonar_left=0.0, sonar_right=0.0)  # Choregraphe reports 0.0
    pilot = Pilot(client, ScriptedPredictor([("THUMBS_UP", 0.95)]), ContextDecider(None),
                  sense_interval=None, sonar_override=virtual)
    return client, pilot


def test_virtual_robot_without_override_refuses_to_move():
    client, pilot = _virtual_pilot(None)
    record = pilot.step(HAND, "Right", now=0.0)
    assert record.decision.action == "WARN_OBSTACLE"
    assert "move" not in client.names()


def test_virtual_sonar_lets_the_virtual_robot_advance_and_marks_it_simulated():
    from brain.main import VirtualSonar

    client, pilot = _virtual_pilot(VirtualSonar())
    record = pilot.step(HAND, "Right", now=0.0)
    assert record.decision.action == "ADVANCE"
    assert ("move", 0.35, 0.0) in client.calls
    assert record.simulated is True  # the sensing was simulated, the log must say so
    assert pilot.sense["sonar_left"] == pytest.approx(1.5)


@pytest.mark.parametrize("key,action", [("3", "WARN_OBSTACLE"), ("2", "ADVANCE_SHORT"), ("1", "ADVANCE")])
def test_virtual_sonar_keys_place_simulated_obstacles(key, action):
    from brain.main import VirtualSonar

    virtual = VirtualSonar()
    assert virtual.handle_key(key) is True
    _, pilot = _virtual_pilot(virtual)
    assert pilot.step(HAND, "Right", now=0.0).decision.action == action


def test_virtual_sonar_side_key_and_no_posture_key():
    from brain.main import VirtualSonar

    virtual = VirtualSonar()
    assert virtual.handle_key("r") is True
    patched = virtual.patch({"ok": True, "sonar_left": 0.0, "sonar_right": 0.0, "posture": "Sit"})
    assert patched["sonar_right"] == pytest.approx(0.20) and patched["sonar_left"] == pytest.approx(1.5)
    assert patched["posture"] == "Sit"  # posture always comes from the robot
    assert virtual.handle_key("s") is False
    assert "virtual" in virtual.describe().lower()


@pytest.mark.parametrize("argv", [["--robot", "zyra", "--virtual-sonar"], ["--mock", "--virtual-sonar"]])
def test_virtual_sonar_is_refused_on_the_real_robot_and_in_mock(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        brain_main.main(argv)
    assert exc.value.code == 2
    assert "--virtual-sonar" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [
    ["--robot", "nyx", "--ip", "192.168.108.90", "--port", "9559", "--virtual-sonar"],
    ["--robot", "sim", "--virtual-sonar"],  # the sim profile points to another machine
])
def test_virtual_sonar_only_on_this_computer(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        brain_main.main(argv)
    assert exc.value.code == 2
    assert "127.0.0.1" in capsys.readouterr().err
