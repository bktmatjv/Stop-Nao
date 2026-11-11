"""Tests for the integration loop (brain.main). No webcam and no real robot.

Pure logic is exercised with stub predictors/deciders; the star case runs
through ``NaoClient(mock=True)`` (real Python 2.7 bridge, no naoqi, no robot).
"""
import csv
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from brain import main as brain_main  # noqa: E402
from brain.context import FEATURE_COLUMNS  # noqa: E402
from brain.context.decider import ContextDecider  # noqa: E402
from brain.nao_client import NaoBridgeError, NaoClient  # noqa: E402

PYTHON27 = r"C:\Python27\python.exe"
needs_py27 = pytest.mark.skipif(
    not os.path.isfile(PYTHON27), reason="Python 2.7 not installed"
)


# --------------------------------------------------------------------- stubs
class ScriptedPredictor:
    """Returns the scripted (label, confidence) pairs, repeating the last one."""

    def __init__(self, script):
        self.script = list(script)
        self.i = 0
        self.resets = 0

    def update(self, landmarks):
        item = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return item

    def reset(self):
        self.resets += 1


class ConstPipeline:
    """sklearn-like pipeline that always predicts the same action."""

    def __init__(self, action):
        self.action = action

    def predict(self, frame):
        return np.array([self.action] * len(frame))


def const_decider(action="ADVANCE"):
    return ContextDecider(bundle={
        "pipeline": ConstPipeline(action),
        "feature_columns": list(FEATURE_COLUMNS),
        "model_name": "stub",
    })


class FakeClient:
    """Records calls; returns a fixed sense reading."""

    def __init__(self, sonar_left=1.5, sonar_right=1.5, moving=False):
        self.sonar = (sonar_left, sonar_right)
        self.moving = moving
        self.acts = []
        self.closed = False
        self.fail_act = False

    def sense(self):
        return {"ok": True, "sonar_left": self.sonar[0],
                "sonar_right": self.sonar[1], "moving": self.moving}

    def act(self, action):
        if self.fail_act:
            raise NaoBridgeError("bridge died")
        self.acts.append(action)
        return {"ok": True, "action": action}

    def set_mock(self, left, right):
        self.sonar = (left, right)
        return {"ok": True}

    def close(self):
        self.closed = True


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


LANDMARKS = np.zeros((33, 4))
LANDMARKS[11] = [0.40, 0.4, 0, 0.9]   # left shoulder
LANDMARKS[12] = [0.56, 0.4, 0, 0.9]   # right shoulder -> person_distance 0.16


def make_pilot(client, script, decider=None, cooldown=2.5, clock=None, logger=None):
    return brain_main.Pilot(
        client, ScriptedPredictor(script), decider or const_decider("ADVANCE"),
        cooldown=cooldown, clock=clock or Clock(), logger=logger,
        sense_interval=None,
    )


# ------------------------------------------------------------- feature dict
def test_feature_dict_keys_match_context_columns():
    feats = brain_main.build_features(
        "STOP", 0.9, {"sonar_left": 1.0, "sonar_right": 0.8, "moving": True}, 0.16)
    assert tuple(feats) == FEATURE_COLUMNS
    assert feats["moving"] == 1
    assert feats["gesture"] == "STOP"
    assert feats["person_distance"] == pytest.approx(0.16)


# --------------------------------------------------------------- star case
@needs_py27
def test_star_case_advance_with_obstacle_warns_and_logs(tmp_path):
    log_path = tmp_path / "runs" / "log.csv"
    logger = brain_main.DecisionLogger(str(log_path))
    with NaoClient(mock=True, python27=PYTHON27, timeout=15) as client:
        client.set_mock(0.2, 0.2)
        pilot = make_pilot(client, [("ADVANCE", 0.9)], logger=logger)
        decision = pilot.step(LANDMARKS)
    logger.close()

    assert decision is not None
    assert decision.action == "WARN_OBSTACLE"
    assert "override" in decision.reason
    assert decision.ack["ok"] is True and decision.ack["action"] == "WARN_OBSTACLE"
    rows = list(csv.DictReader(open(log_path, newline="", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["action"] == "WARN_OBSTACLE"
    assert "override" in rows[0]["reason"]
    assert rows[0]["gesture"] == "ADVANCE"
    assert float(rows[0]["sonar_left"]) == pytest.approx(0.2)
    assert float(rows[0]["person_distance"]) == pytest.approx(0.16)
    assert float(rows[0]["latency_ms"]) >= 0.0


def test_clear_path_advance_is_sent():
    client = FakeClient()
    decision = make_pilot(client, [("ADVANCE", 0.9)]).step(LANDMARKS)
    assert decision.action == "ADVANCE"
    assert client.acts == ["ADVANCE"]


# ----------------------------------------------------- cooldown / NONE rules
def test_none_gesture_never_acts():
    client = FakeClient()
    pilot = make_pilot(client, [("NONE", 0.0), ("NONE", 0.4)])
    for _ in range(20):
        assert pilot.step(LANDMARKS) is None
    assert client.acts == []


def test_cooldown_suppresses_repeated_commands_and_new_gesture_edge():
    clock = Clock()
    client = FakeClient()
    script = [("ADVANCE", 0.9), ("NONE", 0.0), ("ADVANCE", 0.9)]
    pilot = make_pilot(client, script, cooldown=2.5, clock=clock)

    assert pilot.step(LANDMARKS).action == "ADVANCE"       # fires
    clock.t += 0.5
    assert pilot.step(LANDMARKS) is None                    # NONE
    clock.t += 0.5
    assert pilot.step(LANDMARKS) is None                    # new ADVANCE, inside cooldown
    assert client.acts == ["ADVANCE"]
    assert pilot.cooldown_remaining() == pytest.approx(1.5)

    clock.t += 2.0                                          # cooldown over, gesture still held
    assert pilot.step(LANDMARKS).action == "ADVANCE"
    assert client.acts == ["ADVANCE", "ADVANCE"]


def test_held_gesture_does_not_repeat_after_cooldown():
    clock = Clock()
    client = FakeClient()
    pilot = make_pilot(client, [("TURN_LEFT", 0.9)], cooldown=1.0, clock=clock)
    assert pilot.step(LANDMARKS) is not None
    for _ in range(5):
        clock.t += 2.0
        assert pilot.step(LANDMARKS) is None
    assert len(client.acts) == 1


def test_gesture_change_after_cooldown_fires_again():
    clock = Clock()
    client = FakeClient()
    pilot = make_pilot(client, [("TURN_LEFT", 0.9), ("STOP", 0.9)], cooldown=1.0, clock=clock,
                       decider=brain_main.RuleBasedDecider())
    assert pilot.step(LANDMARKS).action == "TURN_LEFT"
    clock.t += 2.0
    assert pilot.step(LANDMARKS).action == "STOP"


def test_missing_pose_is_treated_as_none():
    client = FakeClient()
    pilot = make_pilot(client, [("ADVANCE", 0.9)])
    assert pilot.step(None) is None
    assert client.acts == []
    assert pilot.predictor.resets == 1


# ------------------------------------------------------------ fallback policy
def test_fallback_used_when_context_model_missing(tmp_path, capsys):
    decider = brain_main.load_decider(str(tmp_path / "nope.joblib"))
    assert isinstance(decider, brain_main.RuleBasedDecider)
    err = capsys.readouterr().err
    assert "WARNING" in err and "rule-based" in err


def test_fallback_policy_rules():
    d = brain_main.RuleBasedDecider()

    def feats(gesture, left=1.5, right=1.5, dist=0.16, conf=0.9):
        return brain_main.build_features(
            gesture, conf, {"sonar_left": left, "sonar_right": right, "moving": False}, dist)

    assert d.decide_with_reason(feats("ADVANCE"))[0] == "ADVANCE"
    action, reason = d.decide_with_reason(feats("ADVANCE", left=0.2))
    assert action == "WARN_OBSTACLE" and "safety" in reason
    assert d.decide_with_reason(feats("STOP"))[0] == "STOP"
    assert d.decide_with_reason(feats("TURN_LEFT"))[0] == "TURN_LEFT"
    assert d.decide_with_reason(feats("TURN_RIGHT"))[0] == "TURN_RIGHT"
    assert d.decide_with_reason(feats("ADVANCE", dist=0.03))[0] == "ASK_CLOSER"
    assert d.decide_with_reason(feats("NONE"))[0] == "WAIT"


def test_load_decider_uses_model_when_present(tmp_path):
    import joblib
    path = tmp_path / "ctx.joblib"
    joblib.dump({"pipeline": ConstPipeline("STOP"),
                 "feature_columns": list(FEATURE_COLUMNS), "model_name": "stub"}, path)
    decider = brain_main.load_decider(str(path))
    assert isinstance(decider, ContextDecider)


# ----------------------------------------------------------------- shutdown
def test_shutdown_sends_stop_then_closes():
    client = FakeClient()
    brain_main.safe_shutdown(client)
    assert client.acts == ["STOP"]
    assert client.closed


def test_shutdown_closes_even_if_stop_fails():
    client = FakeClient()
    client.fail_act = True
    brain_main.safe_shutdown(client)
    assert client.closed


def test_run_session_sends_stop_on_exception():
    client = FakeClient()

    def boom():
        raise RuntimeError("camera exploded")

    with pytest.raises(RuntimeError):
        brain_main.run_session(client, boom)
    assert client.acts == ["STOP"]
    assert client.closed


@needs_py27
def test_shutdown_against_mock_bridge_stops_process():
    client = NaoClient(mock=True, python27=PYTHON27, timeout=15)
    brain_main.safe_shutdown(client)
    assert client._proc.poll() is not None


# -------------------------------------------------------------- mock toggle
def test_mock_obstacle_toggle():
    client = FakeClient()
    toggle = brain_main.MockObstacle(client)
    assert toggle.toggle() is True
    assert client.sonar == (0.2, 0.2)
    assert toggle.toggle() is False
    assert client.sonar == (1.5, 1.5)


# --------------------------------------------------------------------- CLI
def test_help_works():
    out = subprocess.run(
        [sys.executable, "-m", "brain.main", "--help"], cwd=ROOT,
        capture_output=True, text=True, timeout=60)
    assert out.returncode == 0
    for flag in ("--robot", "--mock", "--camera", "--gesture-model",
                 "--context-model", "--cooldown", "--log", "--no-video"):
        assert flag in out.stdout


def test_default_log_path_is_under_runs():
    args = brain_main.build_parser().parse_args(["--robot", "sim", "--mock"])
    assert args.log is None
    path = brain_main.default_log_path()
    assert path.replace("\\", "/").startswith("runs/") and path.endswith(".csv")


def test_overlay_lines_show_gesture_sonar_action_and_cooldown():
    clock = Clock()
    client = FakeClient(sonar_left=0.2, sonar_right=0.2)
    pilot = make_pilot(client, [("ADVANCE", 0.9)], clock=clock)
    pilot.step(LANDMARKS)
    text = "\n".join(brain_main.overlay_lines(pilot, MockObstacleStub()))
    assert "ADVANCE" in text and "0.20" in text and "override" in text
    assert "Espera" in text and "obstaculo simulado" in text


class MockObstacleStub:
    blocked = True
