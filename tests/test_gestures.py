"""Tests for the hand gesture pipeline. Uses SYNTHETIC data written to tmp dirs only."""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain.gestures import LABELS  # noqa: E402
from brain.gestures import make_synthetic, train  # noqa: E402
from brain.gestures.capture import (  # noqa: E402
    INSTRUCTIONS,
    build_parser,
    hand_warning,
    landmarks_to_row,
)
from brain.gestures.features import (  # noqa: E402
    ABSOLUTE_FEATURES,
    CSV_COLUMNS,
    FEATURE_NAMES,
    MIN_HAND_SIZE,
    NUM_LANDMARKS,
    extract_features,
    hand_size,
)
from brain.gestures.models import AdaptiveCalibratedSVC  # noqa: E402
from brain.gestures.predictor import GesturePredictor  # noqa: E402

NAMES = list(FEATURE_NAMES)


def _hand(label="FIST", handedness="Right", palm=0.1, center=(0.5, 0.5)):
    return make_synthetic.mean_hand(label, handedness=handedness, palm=palm, center=center)


def _feat(lm, handedness="Right"):
    feats = extract_features(lm, handedness)
    assert feats is not None
    return feats


def _mirror(lm):
    out = lm.copy()
    out[:, 0] = 1.0 - lm[:, 0]
    return out


# ------------------------------------------------------------------ contract
def test_labels_contract():
    assert LABELS == ["THUMBS_UP", "OPEN_PALM", "POINT_LEFT", "POINT_RIGHT", "VICTORY", "FIST", "TWO_FINGERS", "NONE"]
    assert NUM_LANDMARKS == 21
    assert CSV_COLUMNS[:3] == ["person", "label", "handedness"]
    assert CSV_COLUMNS[3:6] == ["x0", "y0", "z0"] and CSV_COLUMNS[-3:] == ["x20", "y20", "z20"]
    assert len(CSV_COLUMNS) == 3 + 21 * 3


# ------------------------------------------------------------------ features
@pytest.mark.parametrize("label", LABELS)
@pytest.mark.parametrize("handedness", ["Left", "Right"])
def test_feature_vector_matches_names(label, handedness):
    feats = _feat(_hand(label, handedness), handedness)
    assert feats.shape == (len(FEATURE_NAMES),)
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)
    assert np.all(np.isfinite(feats))


def test_features_invariant_to_translation_and_scale():
    lm = _hand("VICTORY")
    base = _feat(lm)
    moved = lm.copy()
    moved[:, 0] = lm[:, 0] * 1.7 + 0.21
    moved[:, 1] = lm[:, 1] * 1.7 - 0.13
    moved[:, 2] = 0.9  # z is ignored
    np.testing.assert_allclose(_feat(moved), base, atol=1e-5)  # arccos near 0 amplifies float noise


def test_degenerate_or_invalid_input_returns_none():
    lm = _hand("FIST")
    assert extract_features(lm[:20], "Right") is None  # wrong shape
    assert extract_features(lm[:, :2], "Right") is None  # no z column
    assert extract_features(lm, "Middle") is None  # unknown handedness
    assert extract_features(lm, None) is None
    nan = lm.copy()
    nan[8, 0] = np.nan
    assert extract_features(nan, "Right") is None
    collapsed = lm.copy()
    collapsed[9, :2] = collapsed[0, :2]  # wrist == middle MCP: palm size 0
    assert extract_features(collapsed, "Right") is None
    assert extract_features(np.zeros((21, 3)), "Left") is None
    assert hand_size(collapsed) == 0.0 and hand_size(lm[:5]) == 0.0 and hand_size(nan) == 0.0


def test_features_do_not_depend_on_the_handedness_label():
    # MediaPipe's Left/Right label flips with the pose (seen in real recordings:
    # the same physical hand was labeled "Right" only while pointing right), so
    # the features must be identical whatever label comes with the landmarks.
    assert "is_left" not in NAMES
    for label in LABELS:
        lm = _hand(label, "Right")
        np.testing.assert_allclose(_feat(lm, "Left"), _feat(lm, "Right"), atol=0, err_msg=label)


def test_mirror_image_keeps_mirror_invariant_shape_features():
    invariant = [i for i, n in enumerate(NAMES)
                 if n.startswith(("ext_", "bend_", "tipdist_", "spread_", "angle_", "n_fingers", "thumb_up"))]
    for label in LABELS:
        right = _hand(label, "Right")
        np.testing.assert_allclose(_feat(_mirror(right))[invariant], _feat(right)[invariant],
                                   atol=1e-5, err_msg=label)
        # Absolute image direction flips with the image.
        assert _feat(_mirror(right))[NAMES.index("index_dir_x")] == pytest.approx(
            -_feat(right)[NAMES.index("index_dir_x")])


@pytest.mark.parametrize("handedness", ["Left", "Right"])
def test_pointing_left_and_right_stay_distinguishable(handedness):
    left = _feat(_hand("POINT_LEFT", handedness), handedness)
    right = _feat(_hand("POINT_RIGHT", handedness), handedness)
    ix = NAMES.index("index_dir_x")
    # Camera faces the person: the person's left is the image's right (raw frame).
    assert left[ix] > 0.7 > -0.7 > right[ix]
    assert np.linalg.norm(left - right) > 1.0
    # The finger-shape part is the same pose, only the direction differs.
    ext_idx = [NAMES.index(f"ext_{f}") for f in ("index", "middle", "ring", "pinky")]
    np.testing.assert_allclose(left[ext_idx], right[ext_idx], atol=1e-9)


def test_thumbs_up_and_fist_differ_on_the_thumb_feature():
    up = _feat(_hand("THUMBS_UP"))
    fist = _feat(_hand("FIST"))
    t = NAMES.index("thumb_up_score")
    assert up[t] > 0.5 > 0.3 > fist[t]
    # Finger curl is the same for both (all four fingers curled).
    assert up[NAMES.index("n_fingers_extended")] == fist[NAMES.index("n_fingers_extended")] == 0


def test_finger_extension_features_separate_the_gestures():
    n = NAMES.index("n_fingers_extended")
    assert _feat(_hand("OPEN_PALM"))[n] == 4
    assert _feat(_hand("VICTORY"))[n] == 2
    assert _feat(_hand("POINT_LEFT"))[n] == 1
    assert _feat(_hand("FIST"))[n] == 0
    v = _feat(_hand("VICTORY"))
    assert v[NAMES.index("angle_index_middle")] > 10
    assert v[NAMES.index("spread_index_middle")] > _feat(_hand("FIST"))[NAMES.index("spread_index_middle")]


def test_hand_size_grows_when_closer():
    far, near = _hand(palm=0.06), _hand(palm=0.15)
    assert hand_size(near) > hand_size(far) > 0
    assert hand_size(near) == pytest.approx(0.15, rel=1e-6)


# ------------------------------------------------------------------ capture
def test_capture_helpers_and_parser():
    lm = _hand("FIST", palm=0.1)
    row = landmarks_to_row("ana", "FIST", "Right", lm)
    assert len(row) == len(CSV_COLUMNS)
    assert row[:3] == ["ana", "FIST", "Right"]
    with pytest.raises(ValueError):
        landmarks_to_row("ana", "FIST", "Right", lm[:5])
    with pytest.raises(ValueError):
        landmarks_to_row("ana", "FIST", "Middle", lm)

    args = build_parser().parse_args(["--person", "ana", "--label", "THUMBS_UP"])
    assert args.seconds == 15.0 and args.camera == 0
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--person", "ana", "--label", "JUMP"])


def test_capture_instructions_cover_every_label_in_ascii_spanish():
    assert set(INSTRUCTIONS) == set(LABELS)
    for label, text in INSTRUCTIONS.items():
        assert text.isascii(), label
    assert "izquierda" in INSTRUCTIONS["POINT_LEFT"] and "derecha" in INSTRUCTIONS["POINT_RIGHT"]


def test_capture_hand_warning():
    assert hand_warning(None) == "NO SE VE LA MANO"
    assert hand_warning(_hand(palm=0.1)) is None
    far = hand_warning(_hand(palm=MIN_HAND_SIZE / 2))
    assert far is not None and "LEJOS" in far and far.isascii()


def test_capture_help_runs(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--person" in out and "POINT_LEFT" in out


# ------------------------------------------------------------------ training pieces
def _write_csv(path, label, frames, handedness="Right", person="ana"):
    rows = [landmarks_to_row(person, label, handedness, f) for f in frames]
    pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(path, index=False)


def test_build_features_drops_invalid_rows_and_counts_them():
    rng = np.random.default_rng(5)
    params = make_synthetic.make_person_params(rng, "Right")
    frames = make_synthetic.generate_landmarks("FIST", rng, params, n_frames=10)
    frames[:2, 9, :2] = frames[:2, 0, :2]  # collapsed palm
    frames[2, 4, 0] = np.nan
    rows = [landmarks_to_row("ana", "FIST", "Right", np.nan_to_num(f)) for f in frames]
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.loc[2, "x4"] = np.nan
    X, y, groups, stats = train.build_features(df)
    assert X.shape == (7, len(FEATURE_NAMES))
    assert stats["rows_total"] == 10
    assert stats["rows_dropped_invalid"] == 3
    assert stats["rows_used"] == 7
    assert stats["rows_per_label"] == {"FIST": 7}


def test_load_dataset_requires_contract_columns(tmp_path):
    pd.DataFrame({"person": ["a"], "label": ["FIST"]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="missing columns"):
        train.load_dataset(tmp_path)


def _blobs(counts, seed=0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for k, n in enumerate(counts):
        X.append(rng.normal(loc=4.0 * k, scale=1.0, size=(n, 4)))
        y += [k] * n
    return np.vstack(X), np.array(y)


def test_svm_pipeline_gives_probabilities_without_future_warning():
    X, y = _blobs([20, 20, 20])
    svm, grid = train.get_search_spaces(quick=True, seed=0)["svm"]
    params = {k: v[0] for k, v in grid.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = svm.set_params(**params).fit(X, y)
        proba = model.predict_proba(X[:3])
    assert proba.shape == (3, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert (model.predict(X) == y).mean() > 0.9


@pytest.mark.parametrize("tiny", [1, 2])
def test_svm_with_a_tiny_class_does_not_crash(tiny):
    X, y = _blobs([20, 20, tiny])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = AdaptiveCalibratedSVC(C=10.0).fit(X, y)
        proba = model.predict_proba(X)
    assert proba.shape == (len(y), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert list(model.classes_) == [0, 1, 2]
    assert (model.predict(X[y < 2]) == y[y < 2]).mean() > 0.9


def test_grid_search_with_tiny_inner_class_does_not_crash():
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    X, y = _blobs([30, 30, 3])
    svm, grid = train.get_search_spaces(quick=True, seed=0)["svm"]
    gs = GridSearchCV(svm, grid, scoring="f1_macro", cv=StratifiedKFold(3), error_score="raise")
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        gs.fit(X, y)
    assert gs.best_score_ > 0.5


# ------------------------------------------------------------------ pipeline
@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("gest")
    data = root / "data"
    make_synthetic.main(["--out", str(data), "--frames", "60"])
    out = root / "models" / "gesture_model.joblib"
    figs, metrics = root / "figures", root / "metrics"
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
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
    files = sorted(trained["data"].glob("*.csv"))
    assert len(files) == 4 * len(LABELS) * 2
    df = pd.read_csv(files[0])
    assert list(df.columns) == CSV_COLUMNS
    allp = pd.concat([pd.read_csv(f) for f in files])
    assert set(allp["label"]) == set(LABELS)
    assert set(allp["handedness"]) == {"Left", "Right"}
    assert all(p.startswith("synth") for p in allp["person"].unique())


def test_training_end_to_end_outputs(trained):
    import joblib

    for name in [
        "gesture_cm_rf.png", "gesture_cm_svm.png", "gesture_learning_curve.png",
        "gesture_class_distribution.png", "gesture_model_comparison.png",
    ]:
        assert (trained["figs"] / name).stat().st_size > 1000, name

    metrics = json.loads((trained["metrics"] / "gesture_metrics.json").read_text(encoding="utf-8"))
    for key in ["accuracy", "f1_macro", "per_class_report", "best_params", "cv_scores",
                "train_size", "test_size", "inference_ms_per_sample", "preprocessing"]:
        assert key in metrics, key
    assert "rows_dropped_invalid" in metrics["preprocessing"]
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


def test_make_synthetic_cli_writes_marked_files(tmp_path):
    assert make_synthetic.main(["--out", str(tmp_path / "s"), "--frames", "5"]) == 0
    assert len(list((tmp_path / "s").glob("*.csv"))) == 4 * len(LABELS) * 2


# ------------------------------------------------------------------ predictor
def _frames(label, n, seed, handedness="Right"):
    rng = np.random.default_rng(seed)
    params = make_synthetic.make_person_params(rng, handedness)
    return make_synthetic.generate_landmarks(label, rng, params, n_frames=n)


def test_predictor_majority_vote_and_thresholds(trained):
    pred = GesturePredictor(str(trained["model"]), window=7, min_conf=0.0)
    thumbs = _frames("THUMBS_UP", 12, seed=11)

    # Too few frames -> NONE with zero confidence.
    assert pred.update(thumbs[0], "Right") == ("NONE", 0.0)
    out = None
    for f in thumbs[1:8]:
        out = pred.update(f, "Right")
    assert out[0] == "THUMBS_UP" and 0.0 < out[1] <= 1.0

    # A single outlier frame does not flip the majority.
    fist = _frames("FIST", 1, seed=12)[0]
    assert pred.update(fist, "Right")[0] == "THUMBS_UP"

    # Frames without usable landmarks vote NONE with zero confidence.
    bad = thumbs[0].copy()
    bad[9, :2] = bad[0, :2]
    pred.reset()
    for _ in range(7):
        label, conf = pred.update(bad, "Right")
    assert (label, conf) == ("NONE", 0.0)

    # Impossible confidence threshold forces NONE.
    strict = GesturePredictor(str(trained["model"]), window=3, min_conf=1.01)
    for f in thumbs[:5]:
        label, _ = strict.update(f, "Right")
    assert label == "NONE"


def test_predictor_no_hand_returns_none_immediately(trained):
    pred = GesturePredictor(str(trained["model"]), window=5, min_conf=0.0)
    for f in _frames("VICTORY", 8, seed=31):
        label, _ = pred.update(f, "Right")
    assert label == "VICTORY"
    assert pred.update(None, "Right") == ("NONE", 0.0)
    assert pred.update(None, None) == ("NONE", 0.0)


def test_predictor_sustained_gesture_change_and_pointing_sides(trained):
    pred = GesturePredictor(str(trained["model"]), window=5, min_conf=0.0)
    for f in _frames("OPEN_PALM", 10, seed=21):
        pred.update(f, "Right")
    for f in _frames("POINT_RIGHT", 10, seed=22):
        label, _ = pred.update(f, "Right")
    assert label == "POINT_RIGHT"

    # Same gesture with the other hand (pointing direction stays in image terms).
    pred.reset()
    for f in _frames("POINT_LEFT", 10, seed=23, handedness="Left"):
        label, _ = pred.update(f, "Left")
    assert label == "POINT_LEFT"
