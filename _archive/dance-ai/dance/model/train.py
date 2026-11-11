"""Train and evaluate the dance-style classifier (RF vs SVM vs rule baseline).

Usage::

    python -m dance.model.train --segments data/dataset/segments.csv \\
        --labels data/dataset/labels.csv --out models/dance_model.joblib \\
        --figures-dir report/figures --metrics-dir report/metrics [--quick] [--seed 42]

Evaluation protocol
-------------------
* Segments and labels are inner-joined on ``segment_id``; ``song_id`` is the
  group. Segments of one song are never in both train and test.
* Outer evaluation: leave-one-song-out when there are <= 10 songs, otherwise
  GroupKFold(5). Predictions are collected out-of-fold.
* Hyperparameters of RF and SVM are chosen by a NESTED search: inside every
  outer training fold a GridSearchCV with GroupKFold (scoring f1_macro) runs.
* The rule baseline (``BpmRuleBaseline``) is evaluated with the same splits and
  is only a comparison reference, never used on the robot.
* The model with the best mean cross-validated macro-F1 is refit on all data
  (grid search with grouped folds) and saved.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dance import STYLES, config
from dance.model.baseline import BpmRuleBaseline

META_COLUMNS = ["segment_id", "song_id", "title", "start_s", "end_s"]
MIN_SONGS = 3
LOGO_MAX_SONGS = 10
MODEL_NAMES = ("random_forest", "svm", "baseline")
DISPLAY = {"random_forest": "Random Forest", "svm": "SVM (RBF)", "baseline": "Baseline BPM/RMS"}
STYLE_ES = {"SOFT": "Suave", "MODERATE": "Moderado", "ENERGETIC": "Energético"}
STYLE_COLORS = {"SOFT": "#4C78A8", "MODERATE": "#E3A72F", "ENERGETIC": "#D1495B"}
MODEL_COLORS = {"random_forest": "#2E7D6B", "svm": "#4C78A8", "baseline": "#8A8F98"}


class DatasetError(ValueError):
    """The dataset cannot be used for grouped training (clear user-facing message)."""


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    feature_names: list
    segment_ids: np.ndarray
    frame: pd.DataFrame


# ------------------------------------------------------------------ data


def load_dataset(segments_path, labels_path) -> Dataset:
    segments_path, labels_path = Path(segments_path), Path(labels_path)
    for p in (segments_path, labels_path):
        if not p.is_file():
            raise DatasetError(f"File not found: {p}")
    seg = pd.read_csv(segments_path)
    lab = pd.read_csv(labels_path)
    for name, df, cols in (("segments", seg, ["segment_id", "song_id"]), ("labels", lab, ["segment_id", "label"])):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise DatasetError(f"{name} CSV is missing columns: {missing}")
    if seg["segment_id"].duplicated().any():
        raise DatasetError("segments.csv has duplicated segment_id values")
    if lab["segment_id"].duplicated().any():
        raise DatasetError("labels.csv has duplicated segment_id values")

    feature_names = [c for c in seg.columns if c not in META_COLUMNS]
    if not feature_names:
        raise DatasetError("segments.csv has no feature columns")
    lab = lab[["segment_id", "label"]].copy()
    lab["label"] = lab["label"].astype(str).str.strip().str.upper()
    unknown = sorted(set(lab["label"]) - set(STYLES))
    if unknown:
        raise DatasetError(f"Unknown labels {unknown}; expected {STYLES}")

    df = seg.merge(lab, on="segment_id", how="inner")
    if df.empty:
        raise DatasetError("No segment_id matches between segments.csv and labels.csv")
    if df[feature_names].isna().any().any():
        raise DatasetError("segments.csv contains missing feature values (NaN)")

    n_songs = df["song_id"].nunique()
    if n_songs < MIN_SONGS:
        raise DatasetError(
            f"At least {MIN_SONGS} songs are required for grouped evaluation (found {n_songs})."
        )
    missing_classes = [s for s in STYLES if s not in set(df["label"])]
    if missing_classes:
        raise DatasetError(
            f"Class(es) with no labelled segments: {missing_classes}. All of {STYLES} are required."
        )
    return Dataset(
        X=df[feature_names].to_numpy(dtype=float),
        y=df["label"].to_numpy(dtype=object),
        groups=df["song_id"].astype(str).to_numpy(),
        feature_names=feature_names,
        segment_ids=df["segment_id"].to_numpy(),
        frame=df,
    )


# ------------------------------------------------------------------ models


def make_cv_splits(X, y, groups):
    """Outer grouped splits: leave-one-song-out if <=10 songs, else GroupKFold(5)."""
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    cv = LeaveOneGroupOut() if n_groups <= LOGO_MAX_SONGS else GroupKFold(n_splits=5)
    return list(cv.split(X, y, groups))


def build_rf(seed: int = 42) -> Pipeline:
    return Pipeline(
        [("clf", RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=1))]
    )


class AdaptiveCalibratedSVC(ClassifierMixin, BaseEstimator):
    """SVC with sigmoid-calibrated probabilities that tolerates tiny classes.

    ``CalibratedClassifierCV(cv=k)`` fails when a class has fewer than ``k``
    samples, which happens inside nested, song-grouped CV folds of a small
    dataset. Each ``fit`` therefore uses ``cv = min(max_cv, smallest class
    count)``; if some class has a single sample (no calibration possible), the
    plain SVC is fitted and probabilities are a softmax of its decision scores.
    The wrapped estimator stays reachable as ``estimator`` so grid keys such as
    ``clf__estimator__C`` keep working.
    """

    def __init__(self, estimator=None, max_cv: int = 3):
        self.estimator = estimator
        self.max_cv = max_cv

    def fit(self, X, y):
        base = clone(self.estimator) if self.estimator is not None else SVC(kernel="rbf")
        _, counts = np.unique(y, return_counts=True)
        n_min = int(counts.min())
        if n_min >= 2:
            self.model_ = CalibratedClassifierCV(
                base, method="sigmoid", ensemble=False, cv=min(self.max_cv, n_min)
            ).fit(X, y)
            self.calibrated_ = True
        else:
            self.model_ = base.fit(X, y)
            self.calibrated_ = False
        self.classes_ = self.model_.classes_
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        if self.calibrated_:
            return self.model_.predict_proba(X)
        scores = np.asarray(self.model_.decision_function(X), dtype=float)
        if scores.ndim == 1:  # binary problem
            scores = np.column_stack([-scores, scores])
        scores = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        return exp / exp.sum(axis=1, keepdims=True)


def build_svm(seed: int = 42, calib_cv: int = 3) -> Pipeline:
    """StandardScaler + RBF SVM with calibrated probabilities.

    Calibration replaces ``SVC(probability=True)`` (deprecated in scikit-learn
    1.9); ``calib_cv`` is the maximum number of calibration folds.
    """
    svc = SVC(kernel="rbf", class_weight="balanced", random_state=seed)
    return Pipeline([("scaler", StandardScaler()), ("clf", AdaptiveCalibratedSVC(svc, max_cv=calib_cv))])


def rf_grid(quick: bool) -> dict:
    if quick:
        return {"clf__n_estimators": [50], "clf__max_depth": [None, 5]}
    return {
        "clf__n_estimators": [100, 300],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_leaf": [1, 3],
    }


def svm_grid(quick: bool) -> dict:
    if quick:
        return {"clf__estimator__C": [1, 10]}
    return {"clf__estimator__C": [0.1, 1, 10, 100], "clf__estimator__gamma": ["scale", 0.01, 0.1]}


def _inner_cv(groups) -> GroupKFold:
    return GroupKFold(n_splits=max(2, min(3, len(np.unique(groups)))))


def _search(estimator, grid, groups, n_jobs=1) -> GridSearchCV:
    return GridSearchCV(
        estimator, grid, cv=_inner_cv(groups), scoring="f1_macro", refit=True, n_jobs=n_jobs, error_score="raise"
    )


# ------------------------------------------------------------------ evaluation


def _fold_scores(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def evaluate_model(name, estimator, grid, data: Dataset, splits) -> dict:
    """Out-of-fold evaluation with nested grouped hyperparameter search."""
    y_pred = np.empty(len(data.y), dtype=object)
    folds, best_params = [], []
    for tr, te in splits:
        if grid:
            search = _search(clone(estimator), grid, data.groups[tr])
            search.fit(data.X[tr], data.y[tr], groups=data.groups[tr])
            model = search.best_estimator_
            best_params.append({k: _plain(v) for k, v in search.best_params_.items()})
        else:
            model = clone(estimator).fit(data.X[tr], data.y[tr])
        y_pred[te] = model.predict(data.X[te])
        folds.append(_fold_scores(data.y[te], y_pred[te]))

    pooled = _fold_scores(data.y, y_pred)
    report = classification_report(data.y, y_pred, labels=STYLES, output_dict=True, zero_division=0)
    per_song = {
        str(s): float(accuracy_score(data.y[data.groups == s], y_pred[data.groups == s]))
        for s in sorted(np.unique(data.groups))
    }
    result = {k: float(v) for k, v in pooled.items()}
    for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        vals = np.array([f[metric] for f in folds])
        result[f"cv_{metric}_mean"] = float(vals.mean())
        result[f"cv_{metric}_std"] = float(vals.std())
    result.update(
        {
            "n_folds": len(folds),
            "per_class": _plain(report),
            "per_song_accuracy": per_song,
            "confusion_matrix": confusion_matrix(data.y, y_pred, labels=STYLES).tolist(),
            "confusion_labels": list(STYLES),
            "fold_best_params": best_params,
            "display_name": DISPLAY[name],
        }
    )
    return result


def _plain(obj):
    """Convert numpy scalars/containers to plain JSON-friendly Python."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def limitations_text(n_songs: int, n_segments: int) -> str:
    return (
        f"Con solo {n_songs} canciones ({n_segments} segmentos) los resultados tienen alta varianza: "
        "cada pliegue de validación equivale a una o pocas canciones y una sola canción puede cambiar "
        "mucho las métricas. Los segmentos de una misma canción no son independientes entre sí "
        "(comparten timbre, tempo y producción), por eso la validación se agrupa por canción; aun así, "
        "las métricas son estimaciones optimistas o inestables y no deben leerse como rendimiento "
        "general sobre música nueva."
    )


# ------------------------------------------------------------------ figures


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _plot_confusion(plt, cm, title, path):
    import seaborn as sns

    names = [STYLE_ES[s] for s in STYLES]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sns.heatmap(
        np.asarray(cm), annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=names, yticklabels=names, ax=ax,
    )
    ax.set_xlabel("Estilo predicho")
    ax.set_ylabel("Estilo real (anotado)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_comparison(plt, results, path):
    metrics = [("accuracy", "Exactitud"), ("precision_macro", "Precisión macro"),
               ("recall_macro", "Exhaustividad macro"), ("f1_macro", "F1 macro")]
    width = 0.26
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for i, name in enumerate(MODEL_NAMES):
        means = [results[name][f"cv_{m}_mean"] for m, _ in metrics]
        stds = [results[name][f"cv_{m}_std"] for m, _ in metrics]
        ax.bar(x + (i - 1) * width, means, width, yerr=stds, capsize=3,
               label=DISPLAY[name], color=MODEL_COLORS[name])
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Media entre pliegues (barra = desv. estándar)")
    ax.set_title("Comparación de modelos (validación cruzada agrupada por canción)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=len(MODEL_NAMES))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_importance(plt, rf_pipeline, feature_names, path, top=15):
    imp = rf_pipeline.named_steps["clf"].feature_importances_
    order = np.argsort(imp)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.barh([feature_names[i] for i in order], imp[order], color=MODEL_COLORS["random_forest"])
    ax.set_xlabel("Importancia (reducción de impureza)")
    ax.set_title(f"Random Forest: {len(order)} características más importantes")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_distribution(plt, frame, path):
    counts = frame.groupby(["song_id", "label"]).size().unstack(fill_value=0).reindex(columns=STYLES, fill_value=0)
    fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(counts) + 2), 4.4))
    bottom = np.zeros(len(counts))
    for style in STYLES:
        ax.bar(counts.index, counts[style].to_numpy(), bottom=bottom, label=STYLE_ES[style], color=STYLE_COLORS[style])
        bottom += counts[style].to_numpy()
    ax.set_xlabel("Canción")
    ax.set_ylabel("Número de segmentos")
    ax.set_title("Distribución de estilos anotados por canción")
    ax.legend(frameon=False, title="Estilo")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_figures(results, rf_final, data: Dataset, figures_dir: Path):
    plt = _setup_matplotlib()
    figures_dir.mkdir(parents=True, exist_ok=True)
    for key, tag in (("random_forest", "rf"), ("svm", "svm"), ("baseline", "baseline")):
        _plot_confusion(
            plt, results[key]["confusion_matrix"],
            f"Matriz de confusión: {DISPLAY[key]} (CV por canción)",
            figures_dir / f"dance_cm_{tag}.png",
        )
    _plot_comparison(plt, results, figures_dir / "dance_model_comparison.png")
    _plot_importance(plt, rf_final, data.feature_names, figures_dir / "dance_feature_importance.png")
    _plot_distribution(plt, data.frame, figures_dir / "dance_class_distribution.png")


# ------------------------------------------------------------------ orchestration


def run_training(segments, labels, out, figures_dir, metrics_dir, quick=False, seed=42, log=print) -> dict:
    data = load_dataset(segments, labels)
    n_songs = len(np.unique(data.groups))
    class_counts = {s: int((data.y == s).sum()) for s in STYLES}
    calib_cv = max(2, min(3, min(class_counts.values())))
    splits = make_cv_splits(data.X, data.y, data.groups)
    scheme = "LeaveOneGroupOut" if n_songs <= LOGO_MAX_SONGS else "GroupKFold(5)"
    log(f"Songs: {n_songs} | segments: {len(data.y)} | classes: {class_counts} | CV: {scheme} ({len(splits)} folds)")

    candidates = {
        "random_forest": (build_rf(seed), rf_grid(quick)),
        "svm": (build_svm(seed, calib_cv), svm_grid(quick)),
        "baseline": (BpmRuleBaseline(feature_names=data.feature_names), None),
    }
    results = {}
    for name, (est, grid) in candidates.items():
        log(f"Evaluating {DISPLAY[name]} ...")
        results[name] = evaluate_model(name, est, grid, data, splits)
        log(
            f"  accuracy={results[name]['accuracy']:.3f} f1_macro={results[name]['f1_macro']:.3f} "
            f"(CV f1 mean {results[name]['cv_f1_macro_mean']:.3f} +/- {results[name]['cv_f1_macro_std']:.3f})"
        )

    best_name = max(("random_forest", "svm"), key=lambda n: results[n]["cv_f1_macro_mean"])
    finals, final_params = {}, {}
    for name in ("random_forest", "svm"):
        est, grid = candidates[name]
        search = _search(clone(est), grid, data.groups)
        search.fit(data.X, data.y, groups=data.groups)
        finals[name] = search.best_estimator_
        final_params[name] = {k: _plain(v) for k, v in search.best_params_.items()}

    pipeline = finals[best_name]
    artifact = {
        "pipeline": pipeline,
        "labels": [str(c) for c in pipeline.classes_],
        "feature_names": list(data.feature_names),
        "model_name": best_name,
        "segment_seconds": config.SEGMENT_SECONDS,
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out)

    make_figures(results, finals["random_forest"], data, Path(figures_dir))

    metrics = {
        "n_songs": int(n_songs),
        "n_segments": int(len(data.y)),
        "class_counts": class_counts,
        "songs": sorted(str(s) for s in np.unique(data.groups)),
        "cv_scheme": scheme,
        "nested_cv": "GridSearchCV(GroupKFold, scoring=f1_macro) inside each outer training fold",
        "quick": bool(quick),
        "seed": int(seed),
        "feature_names": list(data.feature_names),
        "selection_criterion": "mean cross-validated f1_macro (per outer fold)",
        "best_model": best_name,
        "final_best_params": final_params,
        "segment_seconds": config.SEGMENT_SECONDS,
        "limitations": limitations_text(n_songs, len(data.y)),
        "models": results,
    }
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "dance_metrics.json").write_text(
        json.dumps(_plain(metrics), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(f"Best model: {best_name} -> {out}")
    log(metrics["limitations"])
    return metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m dance.model.train",
        description="Train RF and SVM (grouped CV by song) plus a BPM/RMS rule baseline; save the best model.",
    )
    p.add_argument("--segments", default="data/dataset/segments.csv", help="segments CSV with features")
    p.add_argument("--labels", default="data/dataset/labels.csv", help="consensus labels CSV")
    p.add_argument("--out", default="models/dance_model.joblib", help="output model artifact")
    p.add_argument("--figures-dir", default="report/figures", help="directory for dance_*.png")
    p.add_argument("--metrics-dir", default="report/metrics", help="directory for dance_metrics.json")
    p.add_argument("--quick", action="store_true", help="small hyperparameter grids (fast check)")
    p.add_argument("--seed", type=int, default=42, help="random seed")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_training(
            args.segments, args.labels, args.out, args.figures_dir, args.metrics_dir,
            quick=args.quick, seed=args.seed,
        )
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
