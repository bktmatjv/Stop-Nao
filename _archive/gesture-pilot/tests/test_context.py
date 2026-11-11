"""Tests for brain.context (safety, synthetic data, training, decider, collect)."""

import json
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.context import ACTIONS, FEATURE_COLUMNS, SCENARIO_COLUMNS  # noqa: E402
from brain.context import collect, make_synthetic, train  # noqa: E402
from brain.context.decider import ContextDecider  # noqa: E402
from brain.context.safety import apply_safety  # noqa: E402


# ---------------------------------------------------------------- safety
def test_demo_case_advance_with_obstacle_warns():
    assert apply_safety("ADVANCE", 0.2, 1.5) == "WARN_OBSTACLE"
    assert apply_safety("ADVANCE", 1.5, 0.2) == "WARN_OBSTACLE"


def test_advance_with_free_path_stays_advance():
    assert apply_safety("ADVANCE", 1.0, 1.0) == "ADVANCE"


def test_threshold_boundary_is_not_an_obstacle():
    assert apply_safety("ADVANCE", 0.30, 0.30) == "ADVANCE"
    assert apply_safety("ADVANCE", 0.299, 0.30) == "WARN_OBSTACLE"


@pytest.mark.parametrize("action", [a for a in ACTIONS if a != "ADVANCE"])
def test_other_actions_never_changed(action):
    assert apply_safety(action, 0.05, 0.05) == action


def test_stop_never_changed():
    assert apply_safety("STOP", 0.0, 0.1) == "STOP"


@pytest.mark.parametrize("left,right", [(float("nan"), 1.0), (1.0, float("nan")), (0.0, 1.0), (-1.0, 1.0), (None, 1.0)])
def test_invalid_sonar_is_unsafe_for_advance(left, right):
    assert apply_safety("ADVANCE", left, right) == "WARN_OBSTACLE"


def test_custom_threshold():
    assert apply_safety("ADVANCE", 0.4, 0.4, threshold=0.5) == "WARN_OBSTACLE"


# ------------------------------------------------------------- synthetic
def test_synthetic_schema_and_source():
    df = make_synthetic.generate(200, noise=0.1, seed=1)
    assert list(df.columns) == [*SCENARIO_COLUMNS, "source"]
    assert len(df) == 200
    assert set(df["source"]) == {"synthetic"}
    assert set(df["action"]) <= set(ACTIONS)
    assert df["scenario_id"].is_unique
    assert df["sonar_left"].between(0.05, 2.6).all()
    assert set(df["moving"]) <= {0, 1}


def test_synthetic_noise_changes_labels():
    clean = make_synthetic.generate(400, noise=0.0, seed=3)
    noisy = make_synthetic.generate(400, noise=0.3, seed=3)
    assert (clean["action"] != noisy["action"]).sum() > 0
    assert make_synthetic.reference_policy(clean.iloc[0].to_dict()) == clean.iloc[0]["action"]


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
    assert "override_rate_test" in metrics["safety"]
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
    with pytest.raises(ValueError):
        train.train(tmp_path / "d.csv", None, tmp_path / "m.joblib", tmp_path / "f", tmp_path / "x")


def test_decider_with_trained_model_applies_safety(trained):
    _, out, _ = trained
    decider = ContextDecider(out)
    blocked = {"gesture": "ADVANCE", "gesture_confidence": 1.0, "sonar_left": 0.2,
               "sonar_right": 0.2, "person_distance": 0.2, "moving": 0}
    assert decider.decide(blocked) != "ADVANCE"
    assert decider.decide(blocked) in ACTIONS


# --------------------------------------------------------------- decider
def _constant_bundle(action):
    pipe = Pipeline([("clf", DummyClassifier(strategy="constant", constant=action))])
    pipe.fit(pd.DataFrame({c: [0] for c in FEATURE_COLUMNS}), [action])
    return {"pipeline": pipe, "feature_columns": list(FEATURE_COLUMNS),
            "labels": list(ACTIONS), "model_name": "stub"}


def _features(left, right):
    return {"gesture": "ADVANCE", "gesture_confidence": 0.9, "sonar_left": left,
            "sonar_right": right, "person_distance": 0.2, "moving": 0}


def test_decider_overrides_stub_model_saying_advance():
    d = ContextDecider(bundle=_constant_bundle("ADVANCE"))
    assert d.decide(_features(0.2, 1.0)) == "WARN_OBSTACLE"
    action, reason = d.decide_with_reason(_features(0.2, 1.0))
    assert action == "WARN_OBSTACLE" and "safety" in reason


def test_decider_keeps_advance_when_clear_and_reason_is_model():
    d = ContextDecider(bundle=_constant_bundle("ADVANCE"))
    action, reason = d.decide_with_reason(_features(1.0, 1.0))
    assert action == "ADVANCE" and "safety" not in reason


def test_decider_loads_from_joblib_file(tmp_path):
    path = tmp_path / "m.joblib"
    joblib.dump(_constant_bundle("STOP"), path)
    assert ContextDecider(path).decide(_features(0.1, 0.1)) == "STOP"


def test_decider_missing_feature_raises():
    d = ContextDecider(bundle=_constant_bundle("STOP"))
    with pytest.raises(KeyError):
        d.decide({"gesture": "STOP"})


# --------------------------------------------------------------- collect
def test_collect_help_works():
    proc = subprocess.run([sys.executable, "-m", "brain.context.collect", "--help"],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "--scenario-id" in proc.stdout and "--mock" in proc.stdout


def test_collect_mock_appends_rows(tmp_path):
    out = tmp_path / "sub" / "scenarios.csv"
    answers = iter([
        "1", "", "0.16", "7",   # gesture ADVANCE, conf default, distance, action WAIT
        "",                     # another sample
        "2", "0.8", "0,2", "4",  # gesture STOP, conf 0.8, distance 0,2 (decimal comma), action STOP
        "q",
    ])
    rc = collect.main(["--scenario-id", "esc01", "--out", str(out), "--mock"],
                      input_fn=lambda _prompt="": next(answers))
    assert rc == 0
    df = pd.read_csv(out)
    assert list(df.columns) == list(SCENARIO_COLUMNS)
    assert len(df) == 2
    assert df["scenario_id"].tolist() == ["esc01", "esc01"]
    assert df["action"].tolist() == ["WAIT", "STOP"]
    assert df["gesture_confidence"].tolist() == [1.0, 0.8]
    assert df["person_distance"].tolist() == [0.16, 0.2]
