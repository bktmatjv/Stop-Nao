"""Tests for dance.model (baseline, grouped training, inference, smoothing).

All data here is SYNTHETIC and exists only to exercise the code paths; it is
never written outside pytest's tmp_path and is never real music data.
"""
import json
import subprocess
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from dance import STYLES, config  # noqa: E402
from dance.model import infer, train  # noqa: E402
from dance.model.baseline import BpmRuleBaseline  # noqa: E402
from dance.model.infer import DancePredictor, SegmentPrediction, smooth_predictions  # noqa: E402

FEATURE_NAMES = (
    ["bpm", "rms_mean", "onset_mean", "spectral_centroid_mean"]
    + [f"mfcc{i}_mean" for i in range(1, 14)]
)
CLASS_BPM = {"SOFT": 75.0, "MODERATE": 110.0, "ENERGETIC": 150.0}
CLASS_RMS = {"SOFT": 0.04, "MODERATE": 0.10, "ENERGETIC": 0.20}


def _synthetic_frames(n_songs=6, per_class=4, classes=tuple(STYLES), seed=0):
    rng = np.random.default_rng(seed)
    seg_rows, lab_rows = [], []
    for s in range(n_songs):
        song_id = f"song{s + 1:02d}"
        idx = 0
        for label in classes:
            for _ in range(per_class):
                feats = rng.normal(0.0, 1.0, len(FEATURE_NAMES))
                feats[0] = CLASS_BPM[label] + rng.normal(0, 6)
                feats[1] = CLASS_RMS[label] + rng.normal(0, 0.01)
                feats[2] = STYLES.index(label) + rng.normal(0, 0.4)
                row = {
                    "segment_id": f"{song_id}_{idx:03d}",
                    "song_id": song_id,
                    "title": f"Synthetic {s}",
                    "start_s": idx * 8.0,
                    "end_s": (idx + 1) * 8.0,
                }
                row.update(dict(zip(FEATURE_NAMES, feats)))
                seg_rows.append(row)
                lab_rows.append(
                    {"segment_id": row["segment_id"], "label": label, "agreement": 1.0, "n_annotators": 4}
                )
                idx += 1
    return pd.DataFrame(seg_rows), pd.DataFrame(lab_rows)


def _write_dataset(tmp_path, **kw):
    seg, lab = _synthetic_frames(**kw)
    seg_path = tmp_path / "segments.csv"
    lab_path = tmp_path / "labels.csv"
    seg.to_csv(seg_path, index=False)
    lab.to_csv(lab_path, index=False)
    return seg_path, lab_path


# ---------------------------------------------------------------- baseline


def test_baseline_fit_predict_proba():
    seg, lab = _synthetic_frames()
    X = seg[FEATURE_NAMES]
    y = lab["label"].to_numpy()
    clf = BpmRuleBaseline().fit(X, y)
    pred = clf.predict(X)
    assert set(pred) <= set(STYLES)
    assert (pred == y).mean() > 0.8  # separable synthetic data
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), len(clf.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert len(clf.thresholds_) == len(clf.classes_) - 1


def test_baseline_accepts_ndarray_with_feature_names():
    seg, lab = _synthetic_frames()
    clf = BpmRuleBaseline(feature_names=FEATURE_NAMES).fit(seg[FEATURE_NAMES].to_numpy(), lab["label"])
    assert clf.predict(seg[FEATURE_NAMES].to_numpy()).shape == (len(seg),)


def test_baseline_requires_bpm_and_rms_columns():
    seg, lab = _synthetic_frames()
    with pytest.raises(ValueError, match="bpm"):
        BpmRuleBaseline().fit(seg[["onset_mean", "rms_mean"]], lab["label"])


# ---------------------------------------------------------------- grouped CV


@pytest.mark.parametrize("n_songs,expected_splits", [(6, 6), (10, 10), (12, 5)])
def test_make_cv_splits_never_mixes_songs(n_songs, expected_splits):
    seg, lab = _synthetic_frames(n_songs=n_songs, per_class=2)
    groups = seg["song_id"].to_numpy()
    splits = train.make_cv_splits(seg[FEATURE_NAMES].to_numpy(), lab["label"].to_numpy(), groups)
    assert len(splits) == expected_splits
    all_test = np.concatenate([te for _, te in splits])
    assert sorted(all_test) == list(range(len(seg)))
    for tr, te in splits:
        assert set(groups[tr]).isdisjoint(set(groups[te]))
        assert len(tr) > 0 and len(te) > 0


def test_load_dataset_joins_and_validates(tmp_path):
    seg_path, lab_path = _write_dataset(tmp_path)
    ds = train.load_dataset(seg_path, lab_path)
    assert ds.X.shape == (6 * 12, len(FEATURE_NAMES))
    assert ds.feature_names == FEATURE_NAMES
    assert set(ds.y) == set(STYLES)
    assert len(set(ds.groups)) == 6


def test_too_few_songs_is_clear_error(tmp_path):
    seg_path, lab_path = _write_dataset(tmp_path, n_songs=2)
    with pytest.raises(train.DatasetError, match="3"):
        train.load_dataset(seg_path, lab_path)


def test_missing_class_is_clear_error(tmp_path):
    seg_path, lab_path = _write_dataset(tmp_path, classes=("SOFT", "ENERGETIC"))
    with pytest.raises(train.DatasetError, match="MODERATE"):
        train.load_dataset(seg_path, lab_path)


# ---------------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("train")
    seg_path, lab_path = _write_dataset(tmp)
    out = tmp / "models" / "dance_model.joblib"
    figs = tmp / "figures"
    mets = tmp / "metrics"
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)  # SVC(probability=True) deprecation must not appear
        rc = train.main(
            [
                "--segments", str(seg_path), "--labels", str(lab_path),
                "--out", str(out), "--figures-dir", str(figs), "--metrics-dir", str(mets),
                "--quick", "--seed", "0",
            ]
        )
    return SimpleNamespace(rc=rc, out=out, figs=figs, mets=mets)


def test_train_cli_writes_artifact(trained):
    assert trained.rc == 0
    art = joblib.load(trained.out)
    assert set(art) == {"pipeline", "labels", "feature_names", "model_name", "segment_seconds"}
    assert art["feature_names"] == FEATURE_NAMES
    assert set(art["labels"]) == set(STYLES)
    assert art["model_name"] in {"random_forest", "svm"}
    assert art["segment_seconds"] == config.SEGMENT_SECONDS
    proba = art["pipeline"].predict_proba(np.zeros((2, len(FEATURE_NAMES))))
    assert proba.shape == (2, 3)
    assert list(art["pipeline"].classes_) == list(art["labels"])


def test_train_cli_writes_figures(trained):
    for name in [
        "dance_cm_rf.png", "dance_cm_svm.png", "dance_cm_baseline.png",
        "dance_model_comparison.png", "dance_feature_importance.png",
        "dance_class_distribution.png",
    ]:
        f = trained.figs / name
        assert f.is_file() and f.stat().st_size > 1000, name


def test_train_cli_writes_metrics(trained):
    m = json.loads((trained.mets / "dance_metrics.json").read_text(encoding="utf-8"))
    assert m["n_songs"] == 6
    assert m["n_segments"] == 72
    assert m["class_counts"] == {"SOFT": 24, "MODERATE": 24, "ENERGETIC": 24}
    assert "alta varianza" in m["limitations"] and "independientes" in m["limitations"]
    assert m["best_model"] in {"random_forest", "svm"}
    for name in ["random_forest", "svm", "baseline"]:
        mm = m["models"][name]
        for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
            assert 0.0 <= mm[key] <= 1.0
        assert "cv_f1_macro_std" in mm and "per_class" in mm and "per_song_accuracy" in mm
        assert set(mm["per_song_accuracy"]) == {f"song{i:02d}" for i in range(1, 7)}
    best = m["models"][m["best_model"]]["cv_f1_macro_mean"]
    assert best >= m["models"]["random_forest"]["cv_f1_macro_mean"] - 1e-12
    assert best >= m["models"]["svm"]["cv_f1_macro_mean"] - 1e-12


def test_artifact_works_in_predictor_with_stub_features(trained, monkeypatch):
    stub = SimpleNamespace(
        FEATURE_NAMES=FEATURE_NAMES,
        extract_features=lambda y, sr: np.array([150.0, 0.2, 2.0] + [0.0] * (len(FEATURE_NAMES) - 3)),
        estimate_bpm=lambda y, sr: 150.0,
    )
    monkeypatch.setattr(infer, "_load_features_module", lambda: stub)
    pred = DancePredictor(trained.out).predict_segment(np.zeros(100), 22050)
    assert pred.style in STYLES
    assert pred.bpm == 150.0
    assert pred.confidence is not None and 0.0 <= pred.confidence <= 1.0
    assert set(pred.probabilities) == set(STYLES)


def test_no_future_warning_fitting_svm():
    seg, lab = _synthetic_frames()
    pipe = train.build_svm(seed=0, calib_cv=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pipe.fit(seg[FEATURE_NAMES].to_numpy(), lab["label"].to_numpy())
        pipe.predict_proba(seg[FEATURE_NAMES].to_numpy())
    assert "clf__estimator__C" in train.svm_grid(quick=False)


def test_train_main_reports_dataset_error(tmp_path, capsys):
    seg_path, lab_path = _write_dataset(tmp_path, n_songs=2)
    rc = train.main(
        ["--segments", str(seg_path), "--labels", str(lab_path), "--out", str(tmp_path / "m.joblib"),
         "--figures-dir", str(tmp_path / "f"), "--metrics-dir", str(tmp_path / "me"), "--quick"]
    )
    assert rc != 0
    assert "3" in capsys.readouterr().err
    assert not (tmp_path / "m.joblib").exists()


# ---------------------------------------------------------------- inference


class _StubPipeline:
    classes_ = np.array(["ENERGETIC", "MODERATE", "SOFT"])

    def predict_proba(self, X):
        assert X.shape == (1, 3)
        return np.array([[0.1, 0.7, 0.2]])


def _stub_artifact():
    return {
        "pipeline": _StubPipeline(),
        "labels": ["ENERGETIC", "MODERATE", "SOFT"],
        "feature_names": ["bpm", "rms_mean", "onset_mean"],
        "model_name": "stub",
        "segment_seconds": 8.0,
    }


def test_predict_segment_uses_features_and_bpm(monkeypatch):
    calls = {}

    def fake_extract(y, sr):
        calls["extract"] = (len(y), sr)
        return np.array([1.0, 2.0, 3.0])

    stub = SimpleNamespace(
        FEATURE_NAMES=["bpm", "rms_mean", "onset_mean"],
        extract_features=fake_extract,
        estimate_bpm=lambda y, sr: 101.5,
    )
    monkeypatch.setattr(infer, "_load_features_module", lambda: stub)
    p = DancePredictor(artifact=_stub_artifact()).predict_segment(np.zeros(50), 22050)
    assert calls["extract"] == (50, 22050)
    assert p.style == "MODERATE"
    assert p.confidence == pytest.approx(0.7)
    assert p.bpm == 101.5
    assert p.probabilities == {
        "ENERGETIC": pytest.approx(0.1),
        "MODERATE": pytest.approx(0.7),
        "SOFT": pytest.approx(0.2),
    }


def test_predict_segment_rejects_feature_mismatch(monkeypatch):
    stub = SimpleNamespace(
        FEATURE_NAMES=["other", "names", "here"],
        extract_features=lambda y, sr: np.zeros(3),
        estimate_bpm=lambda y, sr: 100.0,
    )
    monkeypatch.setattr(infer, "_load_features_module", lambda: stub)
    with pytest.raises(ValueError, match="features"):
        DancePredictor(artifact=_stub_artifact()).predict_segment(np.zeros(10), 22050)


def test_analyze_file_segments_each(monkeypatch):
    y = np.zeros(100)
    segs = [(0.0, 8.0, y[:40]), (8.0, 16.0, y[40:80]), (16.0, 20.0, y[80:])]
    stub = SimpleNamespace(
        FEATURE_NAMES=["bpm", "rms_mean", "onset_mean"],
        load_audio=lambda path, sr=22050: (y, 22050),
        segment_audio=lambda y_, sr, seconds=8.0: segs,
        extract_features=lambda y_, sr: np.zeros(3),
        estimate_bpm=lambda y_, sr: 90.0 + len(y_) / 10,
    )
    monkeypatch.setattr(infer, "_load_features_module", lambda: stub)
    out = DancePredictor(artifact=_stub_artifact()).analyze_file("fake.wav")
    assert [(s.start_s, s.end_s) for s in out] == [(0.0, 8.0), (8.0, 16.0), (16.0, 20.0)]
    assert [s.bpm for s in out] == [94.0, 94.0, 92.0]
    assert all(s.style == "MODERATE" for s in out)


def _sp(style, conf, i=0, bpm=100.0):
    return SegmentPrediction(start_s=i * 8.0, end_s=(i + 1) * 8.0, style=style, confidence=conf, bpm=bpm)


def test_smoothing_low_confidence_keeps_previous_style():
    preds = [_sp("SOFT", 0.9, 0), _sp("ENERGETIC", 0.3, 1), _sp("ENERGETIC", 0.3, 2), _sp("ENERGETIC", 0.9, 3)]
    out = smooth_predictions(preds, min_conf=0.5)
    assert [p.style for p in out] == ["SOFT", "SOFT", "SOFT", "ENERGETIC"]
    # bpm and times are preserved
    assert [p.bpm for p in out] == [p.bpm for p in preds]
    assert [p.start_s for p in out] == [p.start_s for p in preds]


def test_smoothing_removes_isolated_flip_and_keeps_real_changes():
    styles = ["SOFT", "SOFT", "ENERGETIC", "SOFT", "SOFT", "MODERATE", "MODERATE", "MODERATE"]
    preds = [_sp(s, 0.9, i) for i, s in enumerate(styles)]
    out = smooth_predictions(preds)
    assert [p.style for p in out] == ["SOFT"] * 5 + ["MODERATE"] * 3


def test_smoothing_edge_cases_and_input_untouched():
    assert smooth_predictions([]) == []
    one = [_sp("SOFT", 0.1)]
    assert smooth_predictions(one)[0].style == "SOFT"
    preds = [_sp("SOFT", 0.9, 0), _sp("ENERGETIC", 0.1, 1)]
    smooth_predictions(preds)
    assert preds[1].style == "ENERGETIC"


def test_smoothing_none_confidence_is_trusted():
    preds = [_sp("SOFT", None, 0), _sp("MODERATE", None, 1), _sp("MODERATE", None, 2)]
    assert [p.style for p in smooth_predictions(preds)] == ["SOFT", "MODERATE", "MODERATE"]


# ---------------------------------------------------------------- CLIs


@pytest.mark.parametrize("module,flag", [("dance.model.train", "--segments"), ("dance.model.infer", "--song")])
def test_cli_help(module, flag):
    r = subprocess.run(
        [sys.executable, "-m", module, "--help"], cwd=ROOT, capture_output=True, text=True, timeout=120
    )
    assert r.returncode == 0, r.stderr
    assert "usage" in r.stdout.lower()
    assert flag in r.stdout


@pytest.mark.parametrize("minority", [1, 2])
def test_svm_handles_tiny_classes_in_a_fit(minority):
    """Inner CV folds can leave a class with < 3 samples; the SVM must still fit."""
    import warnings

    rng = np.random.default_rng(0)
    n_major = 12
    X = np.vstack([rng.normal(0, 1, (n_major, 4)), rng.normal(4, 1, (n_major, 4)),
                   rng.normal(-4, 1, (minority, 4))])
    y = np.array(["SOFT"] * n_major + ["MODERATE"] * n_major + ["ENERGETIC"] * minority)
    pipe = train.build_svm(seed=0, calib_cv=3)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        pipe.fit(X, y)
    proba = pipe.predict_proba(X[:5])
    assert proba.shape == (5, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert set(pipe.classes_) == {"SOFT", "MODERATE", "ENERGETIC"}
    pipe.set_params(clf__estimator__C=10).fit(X, y)  # grid keys keep working
