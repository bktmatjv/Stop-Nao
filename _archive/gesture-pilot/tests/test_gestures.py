"""Tests for the gesture pipeline. Uses SYNTHETIC data written to tmp dirs only."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.gestures import LABELS  # noqa: E402
from brain.gestures import make_synthetic, train  # noqa: E402
from brain.gestures.capture import build_parser, landmarks_to_row  # noqa: E402
from brain.gestures.features import (  # noqa: E402
    CSV_COLUMNS,
    FEATURE_NAMES,
    L_ELBOW,
    L_HIP,
    L_SHOULDER,
    L_WRIST,
    R_ELBOW,
    R_SHOULDER,
    R_WRIST,
    extract_features,
    missing_key_landmarks,
    person_distance,
)
from brain.gestures.predictor import GesturePredictor  # noqa: E402


def _pose(label="TURN_LEFT", seed=0):
    rng = np.random.default_rng(seed)
    params = make_synthetic.make_person_params(rng)
    return make_synthetic.generate_landmarks(label, rng, params, n_frames=5)[0], params


# ------------------------------------------------------------------ features
def test_feature_vector_matches_names():
    lm, _ = _pose()
    feats = extract_features(lm)
    assert feats is not None
    assert feats.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(feats))


def test_features_invariant_to_translation_and_scale():
    lm, _ = _pose("STOP", seed=3)
    base = extract_features(lm)
    moved = lm.copy()
    moved[:, 0] = lm[:, 0] * 1.7 + 0.21
    moved[:, 1] = lm[:, 1] * 1.7 - 0.13
    np.testing.assert_allclose(extract_features(moved), base, atol=1e-9)


@pytest.mark.parametrize("idx", [L_SHOULDER, R_SHOULDER])
def test_low_visibility_of_a_shoulder_returns_none(idx):
    lm, _ = _pose()
    lm[idx, 3] = 0.3
    assert extract_features(lm) is None


@pytest.mark.parametrize(
    "idx,name",
    [(L_ELBOW, "l_elbow"), (R_ELBOW, "r_elbow"), (L_WRIST, "l_wrist"), (R_WRIST, "r_wrist")],
)
def test_out_of_frame_arm_landmark_is_imputed_and_flagged(idx, name):
    lm, _ = _pose()
    lm[idx, 3] = 0.3
    feats = extract_features(lm)
    assert feats is not None and np.all(np.isfinite(feats))
    names = list(FEATURE_NAMES)
    assert feats[names.index(f"{name}_visible")] == 0.0
    assert feats[names.index(f"{name}_x")] == 0.0 and feats[names.index(f"{name}_y")] == 0.0
    full = extract_features(_pose()[0])
    assert full[names.index(f"{name}_visible")] == 1.0


def test_missing_key_landmarks_lists_out_of_frame_joints():
    lm, _ = _pose()
    assert missing_key_landmarks(lm) == []
    lm[L_WRIST, 3] = 0.1
    lm[R_ELBOW, 3] = 0.2
    assert missing_key_landmarks(lm) == ["r_elbow", "l_wrist"]


def test_capture_framing_warning():
    from brain.gestures.capture import framing_warning

    lm, _ = _pose()
    assert framing_warning(lm) is None
    assert "SIN PERSONA" in framing_warning(None)
    lm[L_WRIST, 3] = 0.1
    msg = framing_warning(lm)
    assert msg.startswith("FUERA DE CUADRO") and "muneca izq" in msg
    assert msg.isascii()


def test_build_features_keeps_frames_with_one_arm_out_of_frame():
    rng = np.random.default_rng(5)
    params = make_synthetic.make_person_params(rng)
    frames = make_synthetic.generate_landmarks("TURN_LEFT", rng, params, n_frames=10)
    frames[:4, R_WRIST, 3] = 0.1  # resting arm out of frame
    frames[4:6, L_SHOULDER, 3] = 0.1  # unusable: no scale reference
    import pandas as pd

    rows = [landmarks_to_row("ana", "TURN_LEFT", f) for f in frames]
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    X, y, groups, stats = train.build_features(df)
    assert X.shape == (8, len(FEATURE_NAMES))
    assert stats["rows_dropped_missing"] == 2
    assert stats["rows_imputed"] == 4


def test_svm_pipeline_gives_probabilities_without_future_warning():
    import warnings

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = np.repeat([0, 1, 2], 20)
    X[y == 1] += 3
    X[y == 2] -= 3
    svm, grid = train.get_search_spaces(quick=True, seed=0)["svm"]
    params = {k: v[0] for k, v in grid.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        model = svm.set_params(**params).fit(X, y)
        proba = model.predict_proba(X[:3])
    assert proba.shape == (3, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_hips_not_visible_falls_back_to_shoulder_midpoint():
    lm, _ = _pose()
    lm[[L_HIP, 24], 3] = 0.1
    feats = extract_features(lm)
    assert feats is not None and np.all(np.isfinite(feats))


def test_left_and_right_arm_gestures_produce_different_features():
    rng = np.random.default_rng(1)
    params = make_synthetic.make_person_params(rng)
    left = make_synthetic.generate_landmarks("TURN_LEFT", rng, params, n_frames=30)
    right = make_synthetic.generate_landmarks("TURN_RIGHT", rng, params, n_frames=30)
    f_left = np.mean([extract_features(f) for f in left], axis=0)
    f_right = np.mean([extract_features(f) for f in right], axis=0)
    names = list(FEATURE_NAMES)
    # Left arm extended: left shoulder angle ~90 deg, right ~10 deg (and the opposite).
    assert f_left[names.index("l_shoulder_angle")] > 60 > f_left[names.index("r_shoulder_angle")]
    assert f_right[names.index("r_shoulder_angle")] > 60 > f_right[names.index("l_shoulder_angle")]
    assert np.linalg.norm(f_left - f_right) > 1.0


def test_person_distance_grows_when_closer():
    lm, _ = _pose()
    closer = lm.copy()
    closer[:, 0] = 0.5 + (lm[:, 0] - 0.5) * 1.5
    assert person_distance(closer) > person_distance(lm) > 0
    lm[L_SHOULDER, 3] = 0.1
    assert person_distance(lm) == 0.0


def test_capture_helpers():
    lm, _ = _pose()
    row = landmarks_to_row("ana", "STOP", lm)
    assert len(row) == len(CSV_COLUMNS)
    assert row[:2] == ["ana", "STOP"]
    parser = build_parser()
    args = parser.parse_args(["--person", "ana", "--label", "ADVANCE"])
    assert args.seconds == 20.0 and args.camera == 0
    with pytest.raises(SystemExit):
        parser.parse_args(["--person", "ana", "--label", "JUMP"])


# ------------------------------------------------------------------ pipeline
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("gest")
    data = root / "data"
    make_synthetic.main(["--out", str(data), "--frames", "60"])
    out = root / "models" / "gesture_model.joblib"
    figs, metrics = root / "figures", root / "metrics"
    train.main(
        [
            "--data", str(data), "--out", str(out),
            "--figures-dir", str(figs), "--metrics-dir", str(metrics),
            "--quick", "--n-jobs", "1",
        ]
    )
    return {"root": root, "data": data, "model": out, "figs": figs, "metrics": metrics}


def test_synthetic_data_is_marked_and_has_contract_columns(trained):
    assert (trained["data"] / "SYNTHETIC_README.txt").exists()
    import pandas as pd

    files = sorted(trained["data"].glob("*.csv"))
    assert len(files) == 4 * 5 * 2
    df = pd.read_csv(files[0])
    assert list(df.columns) == CSV_COLUMNS
    assert set(pd.concat([pd.read_csv(f) for f in files])["label"]) == set(LABELS)


def test_training_end_to_end_outputs(trained):
    import joblib

    for name in [
        "gesture_cm_rf.png", "gesture_cm_svm.png", "gesture_learning_curve.png",
        "gesture_class_distribution.png", "gesture_model_comparison.png",
    ]:
        assert (trained["figs"] / name).stat().st_size > 1000, name

    metrics = json.loads((trained["metrics"] / "gesture_metrics.json").read_text(encoding="utf-8"))
    for key in ["accuracy", "f1_macro", "per_class_report", "best_params", "cv_scores",
                "train_size", "test_size", "inference_ms_per_sample"]:
        assert key in metrics, key
    assert set(metrics["models"]) == {"random_forest", "svm"}
    assert metrics["split"]["test_person"] == "synth4"  # held out whole person
    assert metrics["train_size"] > 0 and metrics["test_size"] > 0
    assert metrics["accuracy"] > 0.7  # synthetic data is easy; guards against a broken pipeline

    artifact = joblib.load(trained["model"])
    assert set(artifact) == {"pipeline", "labels", "feature_names", "model_name"}
    assert artifact["labels"] == sorted(LABELS)
    assert artifact["feature_names"] == FEATURE_NAMES


def test_train_falls_back_with_fewer_than_three_persons(tmp_path):
    data = tmp_path / "data"
    make_synthetic.generate_dataset(data, n_persons=2, frames=40)
    with pytest.warns(UserWarning, match="<3"):
        metrics = train.train(
            train.build_parser().parse_args(
                ["--data", str(data), "--out", str(tmp_path / "m.joblib"),
                 "--figures-dir", str(tmp_path / "f"), "--metrics-dir", str(tmp_path / "mt"),
                 "--quick", "--n-jobs", "1"]
            )
        )
    assert metrics["split"]["test_person"] is None
    assert "fallback" in metrics["split"]["strategy"]


def _frames(label, n, seed):
    rng = np.random.default_rng(seed)
    params = make_synthetic.make_person_params(rng)
    return make_synthetic.generate_landmarks(label, rng, params, n_frames=n)


def test_predictor_majority_vote_and_thresholds(trained):
    pred = GesturePredictor(str(trained["model"]), window=7, min_conf=0.0)
    advance = _frames("ADVANCE", 12, seed=11)

    # Too few frames -> NONE with zero confidence.
    assert pred.update(advance[0]) == ("NONE", 0.0)
    out = None
    for f in advance[1:8]:
        out = pred.update(f)
    assert out[0] == "ADVANCE" and 0.0 < out[1] <= 1.0

    # A single outlier frame does not flip the majority.
    turn = _frames("TURN_LEFT", 1, seed=12)[0]
    assert pred.update(turn)[0] == "ADVANCE"

    # Frames without usable landmarks (no shoulder => no scale) vote NONE with zero confidence.
    bad = advance[0].copy()
    bad[L_SHOULDER, 3] = 0.0
    pred.reset()
    for _ in range(7):
        label, conf = pred.update(bad)
    assert (label, conf) == ("NONE", 0.0)

    # Impossible confidence threshold forces NONE.
    strict = GesturePredictor(str(trained["model"]), window=3, min_conf=1.01)
    for f in advance[:5]:
        label, _ = strict.update(f)
    assert label == "NONE"


def test_predictor_sustained_gesture_change(trained):
    pred = GesturePredictor(str(trained["model"]), window=5, min_conf=0.0)
    for f in _frames("STOP", 10, seed=21):
        pred.update(f)
    for f in _frames("TURN_RIGHT", 10, seed=22):
        label, _ = pred.update(f)
    assert label == "TURN_RIGHT"
