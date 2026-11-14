"""Tests for brain.context (intents, safety, decider, synthetic data, training, collect)."""

import json
import math
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.context import ACTIONS, FEATURE_COLUMNS, GESTURES, SCENARIO_COLUMNS  # noqa: E402
from brain.context import collect, intents, make_synthetic, safety, train  # noqa: E402
from brain.context.decider import ContextDecider  # noqa: E402
from brain.context.intents import GESTURE_TO_INTENT, intent_for  # noqa: E402
from brain.context.safety import Decision, validate  # noqa: E402


def ctx(**over):
    base = {
        "gesture": "THUMBS_UP",
        "gesture_confidence": 0.95,
        "sonar_left": 1.5,
        "sonar_right": 1.5,
        "hand_size": 0.15,
        "moving": False,
        "standing": True,
    }
    base.update(over)
    return base


# ------------------------------------------------------------- contract
def test_contract_constants():
    assert GESTURES == ("THUMBS_UP", "OPEN_PALM", "POINT_LEFT", "POINT_RIGHT", "VICTORY", "FIST", "LOVE_YOU", "NONE")
    assert FEATURE_COLUMNS == ("gesture", "gesture_confidence", "sonar_left", "sonar_right",
                               "hand_size", "moving", "standing")
    assert SCENARIO_COLUMNS == (*FEATURE_COLUMNS, "action", "scenario_id")
    assert ACTIONS == ("ADVANCE", "ADVANCE_SHORT", "TURN_LEFT", "TURN_RIGHT", "STOP", "WAVE", "GREET", "SIT",
                       "STAND", "WARN_OBSTACLE", "WARN_TOO_CLOSE", "WARN_MUST_STOP",
                       "WARN_MUST_STAND", "ASK_CONFIRM", "WAIT")
    assert (safety.SAFE_FRONT, safety.SAFE_SIDE, safety.ADVANCE_STEP, safety.MIN_STEP) == (0.30, 0.25, 0.35, 0.05)
    assert (safety.TURN_ANGLE, safety.MIN_CONF, safety.TOO_CLOSE_HAND) == (0.785, 0.6, 0.25)


# --------------------------------------------------------------- intents
def test_intent_mapping_basic():
    assert intent_for("THUMBS_UP", True) == "ADVANCE"
    assert intent_for("OPEN_PALM", True) == "STOP"
    assert intent_for("VICTORY", True) is None  # VICTORY toggles the input now; WAVE is voice only
    assert intent_for("NONE", True) is None
    assert GESTURE_TO_INTENT["NONE"] is None
    assert GESTURE_TO_INTENT["FIST"] == "TOGGLE_POSTURE"
    assert set(GESTURE_TO_INTENT) == set(GESTURES)


def test_pointing_turns_toward_pointed_side():
    # the person faces the robot: their left is the robot's right
    assert intents.POINT_TURNS_TOWARD_POINTED_SIDE is True
    assert intent_for("POINT_LEFT", True) == "TURN_RIGHT"
    assert intent_for("POINT_RIGHT", True) == "TURN_LEFT"


def test_pointing_literal_side_when_flag_off(monkeypatch):
    monkeypatch.setattr(intents, "POINT_TURNS_TOWARD_POINTED_SIDE", False)
    assert intent_for("POINT_LEFT", True) == "TURN_LEFT"
    assert intent_for("POINT_RIGHT", True) == "TURN_RIGHT"


def test_fist_toggles_by_posture():
    assert intent_for("FIST", True) == "SIT"
    assert intent_for("FIST", False) == "STAND"


# ---------------------------------------------------------------- safety
def test_star_case_advance_with_obstacle_is_refused():
    d = validate("ADVANCE", ctx(sonar_left=0.20, sonar_right=1.5))
    assert isinstance(d, Decision)
    assert d.action == "WARN_OBSTACLE"
    assert d.eye_color == "red"
    assert d.overridden is True
    assert "20 cent" in d.reason
    assert d.reason.endswith("No puedo avanzar.")


def test_front_is_min_of_both_sonars():
    d = validate("ADVANCE", ctx(sonar_left=1.5, sonar_right=0.22))
    assert d.action == "WARN_OBSTACLE" and "22 cent" in d.reason


def test_graded_advance_gives_only_the_safe_distance():
    d = validate("ADVANCE", ctx(sonar_left=0.40, sonar_right=0.90))
    assert d.action == "ADVANCE_SHORT"
    assert d.params["x"] == pytest.approx(0.10, abs=1e-6)
    assert d.eye_color == "yellow"
    assert d.overridden is True
    assert "10 cent" in d.reason


def test_graded_advance_below_min_step_is_refused():
    d = validate("ADVANCE", ctx(sonar_left=0.33, sonar_right=0.33))  # only 3 cm of margin
    assert d.action == "WARN_OBSTACLE"
    d = validate("ADVANCE", ctx(sonar_left=0.35, sonar_right=0.35))  # exactly MIN_STEP
    assert d.action == "ADVANCE_SHORT" and d.params["x"] == pytest.approx(0.05, abs=1e-6)


def test_free_path_advances_20_cm():
    d = validate("ADVANCE", ctx(sonar_left=1.0, sonar_right=1.0))
    assert d.action == "ADVANCE" and d.params == {"x": 0.35}
    assert d.eye_color == "green" and d.overridden is False
    # boundary: front == SAFE_FRONT + ADVANCE_STEP is free
    assert validate("ADVANCE", ctx(sonar_left=0.65, sonar_right=0.65)).action == "ADVANCE"


def test_safe_front_boundary():
    assert validate("ADVANCE", ctx(sonar_left=0.30, sonar_right=0.30)).action == "WARN_OBSTACLE"
    assert validate("ADVANCE", ctx(sonar_left=0.299, sonar_right=1.0)).action == "WARN_OBSTACLE"


@pytest.mark.parametrize("left,right", [(float("nan"), 1.0), (1.0, float("nan")), (0.0, 1.0),
                                        (-1.0, 1.0), (None, 1.0), (1.0, None)])
def test_invalid_sonar_is_unsafe_for_advance(left, right):
    d = validate("ADVANCE", ctx(sonar_left=left, sonar_right=right))
    assert d.action == "WARN_OBSTACLE" and d.eye_color == "red" and d.reason


def test_advance_when_sitting_must_stand_first():
    d = validate("ADVANCE", ctx(standing=False))
    assert d.action == "WARN_MUST_STAND" and d.overridden is True


def test_low_confidence_asks_to_repeat():
    d = validate("ADVANCE", ctx(gesture_confidence=0.59))
    assert d.action == "ASK_CONFIRM" and d.eye_color == "yellow" and d.overridden is True
    assert d.reason == "No estoy seguro del gesto. ¿Puedes repetirlo?"
    # confidence check wins even over an obstacle (we do not trust the gesture)
    assert validate("ADVANCE", ctx(gesture_confidence=0.1, sonar_left=0.1)).action == "ASK_CONFIRM"
    assert validate("ADVANCE", ctx(gesture_confidence=0.6)).action == "ADVANCE"


def test_stop_is_always_allowed():
    for over in (dict(sonar_left=0.0, sonar_right=0.0), dict(gesture_confidence=0.0),
                 dict(standing=False), dict(moving=True), dict(sonar_left=float("nan"))):
        d = validate("STOP", ctx(**over))
        assert d.action == "STOP" and d.eye_color == "green" and d.overridden is False


def test_wait_passes_even_without_confidence():
    d = validate("WAIT", ctx(gesture="NONE", gesture_confidence=0.0))
    assert d.action == "WAIT" and d.eye_color == "blue" and d.overridden is False


def test_turn_left_checks_left_sonar_only():
    ok = validate("TURN_LEFT", ctx(sonar_left=1.0, sonar_right=0.1))
    assert ok.action == "TURN_LEFT" and ok.params == {"theta": 0.785} and ok.overridden is False
    blocked = validate("TURN_LEFT", ctx(sonar_left=0.20, sonar_right=2.0))
    assert blocked.action == "WARN_OBSTACLE" and blocked.eye_color == "red"
    assert "izquierda" in blocked.reason and "20 cent" in blocked.reason


def test_turn_right_checks_right_sonar_only():
    ok = validate("TURN_RIGHT", ctx(sonar_left=0.1, sonar_right=1.0))
    assert ok.action == "TURN_RIGHT" and ok.params == {"theta": -0.785}
    blocked = validate("TURN_RIGHT", ctx(sonar_left=2.0, sonar_right=0.24))
    assert blocked.action == "WARN_OBSTACLE" and "derecha" in blocked.reason


def test_turn_boundary_and_invalid_side_sonar():
    assert validate("TURN_LEFT", ctx(sonar_left=0.25)).action == "TURN_LEFT"
    assert validate("TURN_LEFT", ctx(sonar_left=0.249)).action == "WARN_OBSTACLE"
    d = validate("TURN_RIGHT", ctx(sonar_right=float("nan")))
    assert d.action == "WARN_OBSTACLE" and "derecha" in d.reason


def test_turn_when_sitting_must_stand_first():
    assert validate("TURN_LEFT", ctx(standing=False)).action == "WARN_MUST_STAND"


def test_wave_rules():
    ok = validate("WAVE", ctx())
    assert ok.action == "WAVE" and ok.eye_color == "magenta" and ok.overridden is False
    assert validate("WAVE", ctx(moving=True)).action == "WARN_MUST_STOP"
    close = validate("WAVE", ctx(hand_size=0.30))
    assert close.action == "WARN_TOO_CLOSE" and close.eye_color == "red"
    assert validate("WAVE", ctx(hand_size=0.25)).action == "WAVE"
    assert validate("WAVE", ctx(standing=False)).action == "WARN_MUST_STAND"


@pytest.mark.parametrize("action", ["SIT", "STAND"])
def test_posture_changes_need_the_robot_to_be_still(action):
    d = validate(action, ctx(moving=True))
    assert d.action == "WARN_MUST_STOP" and d.overridden is True and d.reason
    ok = validate(action, ctx(moving=False))
    assert ok.action == action and ok.eye_color == "white" and ok.overridden is False


def test_every_action_yields_a_complete_decision():
    for action in ACTIONS:
        for over in ({}, dict(sonar_left=0.1), dict(moving=True), dict(standing=False),
                     dict(gesture_confidence=0.2)):
            d = validate(action, ctx(**over))
            assert d.action in ACTIONS
            assert isinstance(d.params, dict)
            assert d.reason.strip()
            assert d.eye_color in {"green", "yellow", "red", "blue", "magenta", "cyan", "white"}
            assert isinstance(d.overridden, bool)
            assert d.overridden == (d.action != action)


def test_validate_is_pure():
    c = ctx(sonar_left=0.2)
    before = dict(c)
    validate("ADVANCE", c)
    assert c == before and not math.isnan(c["sonar_left"])


# --------------------------------------------------------------- decider
def _constant_bundle(action):
    pipe = Pipeline([("clf", DummyClassifier(strategy="constant", constant=action))])
    pipe.fit(pd.DataFrame({c: [0] for c in FEATURE_COLUMNS}), [action])
    return {"pipeline": pipe, "feature_columns": list(FEATURE_COLUMNS),
            "labels": list(ACTIONS), "model_name": "stub"}


def test_decider_without_model_uses_intents_then_validates():
    d = ContextDecider()
    assert d.decide(ctx()).action == "ADVANCE"
    assert d.decide(ctx(sonar_left=0.2)).action == "WARN_OBSTACLE"
    assert d.decide(ctx(gesture="POINT_LEFT")).action == "TURN_RIGHT"
    assert d.decide(ctx(gesture="FIST", standing=True)).action == "SIT"
    assert d.decide(ctx(gesture="FIST", standing=False)).action == "STAND"
    assert d.decide(ctx(gesture="NONE", gesture_confidence=0.0)).action == "WAIT"
    assert d.decide(ctx(gesture_confidence=0.3)).action == "ASK_CONFIRM"


def test_decider_stub_model_saying_advance_is_still_blocked_by_safety():
    d = ContextDecider(bundle=_constant_bundle("ADVANCE"))
    blocked = d.decide(ctx(sonar_left=0.2))
    assert blocked.action == "WARN_OBSTACLE" and blocked.overridden is True
    assert d.decide(ctx(sonar_left=1.0)).action == "ADVANCE"


def test_decider_loads_from_joblib_file(tmp_path):
    path = tmp_path / "m.joblib"
    joblib.dump(_constant_bundle("STOP"), path)
    assert ContextDecider(path).decide(ctx(sonar_left=0.1)).action == "STOP"
    assert ContextDecider(model_path=str(path)).decide(ctx()).action == "STOP"


def test_decider_missing_feature_raises_with_model():
    d = ContextDecider(bundle=_constant_bundle("STOP"))
    with pytest.raises(KeyError):
        d.decide({"gesture": "STOP"})


def test_explain_without_model_or_with_stub_is_a_short_note():
    for d in (ContextDecider(), ContextDecider(bundle=_constant_bundle("STOP"))):
        text = d.explain(ctx())
        assert isinstance(text, str) and text and "Resultado" not in text and "\n" not in text.strip()


def _tree_decider():
    df = make_synthetic.generate(400, noise=0.05, seed=2)
    pipe = train.build_pipeline(DecisionTreeClassifier(max_depth=4, random_state=0))
    pipe.fit(df[list(FEATURE_COLUMNS)], df["action"])
    bundle = {"pipeline": pipe, "feature_columns": list(FEATURE_COLUMNS),
              "labels": list(ACTIONS), "model_name": "DecisionTree"}
    return ContextDecider(bundle=bundle)


def test_explain_decision_tree_shows_the_path():
    d = _tree_decider()
    text = d.explain(ctx(sonar_left=0.2, sonar_right=0.2))
    assert "Resultado del árbol:" in text
    assert "sonar_left" in text or "sonar_right" in text or "gesture" in text
    assert "<=" in text or ">" in text
    predicted = text.split("Resultado del árbol:")[1].strip()
    assert predicted in ACTIONS


# ------------------------------------------------------------- synthetic
def test_synthetic_schema_and_source():
    df = make_synthetic.generate(200, noise=0.1, seed=1)
    assert list(df.columns) == [*SCENARIO_COLUMNS, "source"]
    assert len(df) == 200
    assert set(df["source"]) == {"synthetic"}
    assert set(df["action"]) <= set(ACTIONS)
    assert set(df["gesture"]) <= set(GESTURES)
    assert df["scenario_id"].is_unique
    assert df["sonar_left"].between(0.05, 2.6).all()
    assert df["hand_size"].between(0.0, 0.5).all()
    assert set(df["moving"]) <= {0, 1} and set(df["standing"]) <= {0, 1}


def test_synthetic_noise_changes_labels():
    clean = make_synthetic.generate(400, noise=0.0, seed=3)
    noisy = make_synthetic.generate(400, noise=0.3, seed=3)
    assert (clean["action"] != noisy["action"]).sum() > 0
    row = clean.iloc[0].to_dict()
    assert make_synthetic.reference_policy(row) == row["action"]


def test_reference_policy_star_case():
    row = ctx(sonar_left=0.2)
    assert make_synthetic.reference_policy(row) == "WARN_OBSTACLE"
    assert make_synthetic.reference_policy(ctx()) == "ADVANCE"


def test_synthetic_cli_writes_csv(tmp_path):
    out = tmp_path / "syn.csv"
    assert make_synthetic.main(["--n", "50", "--out", str(out), "--noise", "0.1"]) == 0
    assert len(pd.read_csv(out)) == 50


# ----------------------------------------------------------------- train
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ctx")
    syn = tmp / "syn.csv"
    real = tmp / "real.csv"
    make_synthetic.generate(300, noise=0.1, seed=7).to_csv(syn, index=False)
    # "real" file: contract columns only (no source column), scene ids shared by rows
    r = make_synthetic.generate(60, noise=0.1, seed=8)[list(SCENARIO_COLUMNS)]
    r["scenario_id"] = [f"esc{i % 12:02d}" for i in range(len(r))]
    r.to_csv(real, index=False)
    out = tmp / "model.joblib"
    report = train.train(real, syn, out, tmp / "fig", tmp / "met")
    return tmp, out, report


def test_train_end_to_end_artifacts(trained):
    tmp, out, report = trained
    for name in ["context_cm_dt", "context_cm_rf", "context_tree",
                 "context_feature_importance", "context_model_comparison"]:
        assert (tmp / "fig" / f"{name}.png").stat().st_size > 0
    metrics = json.loads((tmp / "met" / "context_metrics.json").read_text(encoding="utf-8"))
    assert set(metrics["models"]) == {"DecisionTree", "RandomForest"}
    assert metrics["rows_by_source"] == {"synthetic": 300, "real": 60}
    assert 0.0 <= metrics["safety"]["override_rate_test"] <= 1.0
    assert "rule_only" in metrics and 0.0 <= metrics["rule_only"]["test"]["accuracy"] <= 1.0
    assert 0.0 <= report["models"]["RandomForest"]["test"]["accuracy"] <= 1.0
    assert "test_real_only" in report["models"]["DecisionTree"]
    bundle = joblib.load(out)
    assert set(bundle) == {"pipeline", "feature_columns", "labels", "model_name"}
    assert bundle["feature_columns"] == list(FEATURE_COLUMNS)
    assert bundle["model_name"] in {"DecisionTree", "RandomForest"}


def test_train_needs_enough_groups(tmp_path):
    df = make_synthetic.generate(10, seed=0)
    df["scenario_id"] = "same"
    df.to_csv(tmp_path / "d.csv", index=False)
    with pytest.raises(ValueError, match="at least 4"):
        train.train(tmp_path / "d.csv", None, tmp_path / "m.joblib", tmp_path / "f", tmp_path / "x")


def test_decider_with_trained_model_applies_safety(trained):
    _, out, _ = trained
    decider = ContextDecider(out)
    decision = decider.decide(ctx(sonar_left=0.2, sonar_right=0.2))
    assert decision.action != "ADVANCE" and decision.action in ACTIONS
    assert decision.reason


# --------------------------------------------------------------- collect
def test_collect_help_works():
    proc = subprocess.run([sys.executable, "-m", "brain.context.collect", "--help"],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0
    for flag in ("--scenario-id", "--mock", "--robot", "--ip", "--port", "--out"):
        assert flag in proc.stdout


def _idx(options, name):
    return str(list(options).index(name) + 1)


def test_collect_mock_appends_rows(tmp_path):
    out = tmp_path / "sub" / "scenarios.csv"
    answers = iter([
        "0.20", "1,5", "n", "s",                       # sonars (decimal comma), moving no, standing yes
        _idx(GESTURES, "THUMBS_UP"), "", "",           # default confidence and hand_size
        _idx(ACTIONS, "WARN_OBSTACLE"),                # human label
        "",                                            # another sample
        "1.0", "1.0", "s", "n",                        # moving, not standing
        "FIST", "0.8", "0,2",                          # name instead of number
        "STAND",
        "q",
    ])
    rc = collect.main(["--scenario-id", "esc01", "--out", str(out), "--mock"],
                      input_fn=lambda _prompt="": next(answers))
    assert rc == 0
    df = pd.read_csv(out)
    assert list(df.columns) == list(SCENARIO_COLUMNS)
    assert df["scenario_id"].tolist() == ["esc01", "esc01"]
    assert df["gesture"].tolist() == ["THUMBS_UP", "FIST"]
    assert df["action"].tolist() == ["WARN_OBSTACLE", "STAND"]
    assert df["gesture_confidence"].tolist() == [1.0, 0.8]
    assert df["hand_size"].tolist() == [0.15, 0.2]
    assert df["sonar_left"].tolist() == [0.2, 1.0]
    assert df["sonar_right"].tolist() == [1.5, 1.0]
    assert df["moving"].tolist() == [0, 1]
    assert df["standing"].tolist() == [1, 0]


def test_collect_reads_robot_through_client_sense(tmp_path):
    class FakeClient:
        def sense(self):
            return {"ok": True, "sonar_left": 0.9, "sonar_right": None, "moving": False, "posture": "StandInit"}

    answers = iter([_idx(GESTURES, "OPEN_PALM"), "", "", _idx(ACTIONS, "STOP")])
    out = tmp_path / "s.csv"
    row = collect.record_one("e1", out, FakeClient(), mock=False, input_fn=lambda _p="": next(answers))
    assert row["standing"] == 1 and row["moving"] == 0
    assert row["sonar_left"] == 0.9 and row["sonar_right"] == 0.0  # unreadable sonar stored as invalid (0)
    assert pd.read_csv(out).loc[0, "action"] == "STOP"


@pytest.mark.parametrize("posture,standing", [("Stand", 1), ("StandInit", 1), ("StandZero", 1),
                                              ("Sit", 0), ("Crouch", 0), ("Unknown", 0), (None, 0)])
def test_standing_from_posture(posture, standing):
    assert collect.is_standing(posture) == bool(standing)
