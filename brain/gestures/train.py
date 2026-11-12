"""Train and compare gesture classifiers (RandomForest vs SVM RBF).

Usage:
    python -m brain.gestures.train --data data/gestures --out models/gesture_model.joblib

Protocol
--------
* Missing-value handling: frames whose landmarks are unusable
  (``extract_features`` returns None: non-finite values, unknown handedness or
  a collapsed palm) are dropped and counted; the number of frames with a very
  small (far) hand is reported too but they are kept.
* Features are translation/scale invariant and mirror-canonical for the left
  hand (see ``features.py``).
* Labels are encoded with ``LabelEncoder``.
* Split: with >= 3 persons, one person is held out as the test set and the rest
  is used for hyperparameter search with ``GroupKFold`` (groups = persons), so
  the model is always evaluated on people it has never seen. With < 3 persons
  a warning is printed and a stratified 20% hold-out + ``StratifiedKFold`` is
  used (optimistic: consecutive video frames leak between train and test).
* ``GridSearchCV`` with ``f1_macro`` scoring for both models; the best model is
  picked by cross-validation score on the training persons (never by test score).
* Figure text is in Spanish because the figures go into the Spanish report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # noqa: E402
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV,
    GroupKFold,
    StratifiedKFold,
    learning_curve,
    train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import LabelEncoder, StandardScaler  # noqa: E402

from . import LABELS  # noqa: E402
from .features import (  # noqa: E402
    FEATURE_NAMES,
    LANDMARK_COLUMNS,
    MIN_HAND_SIZE,
    NUM_LANDMARKS,
    extract_features,
    hand_size,
)
from .models import AdaptiveCalibratedSVC  # noqa: E402

MODEL_TITLES = {"random_forest": "Random Forest", "svm": "SVM (kernel RBF)"}


# --------------------------------------------------------------------------- data
def load_dataset(data_dir: Path) -> pd.DataFrame:
    """Concatenate all ``*.csv`` files of the gesture dataset."""
    files = sorted(Path(data_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV files found in {data_dir}")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    missing = [c for c in ["person", "label", "handedness"] + LANDMARK_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV files are missing columns: {missing[:5]}...")
    df["person"] = df["person"].astype(str)
    df["label"] = df["label"].astype(str)
    df["handedness"] = df["handedness"].astype(str)
    print(f"Loaded {len(files)} files, {len(df)} frames from {data_dir}")
    return df


def build_features(df: pd.DataFrame):
    """Extract features, dropping frames with unusable landmarks.

    Returns (X, y_str, groups, stats).
    """
    raw = df[LANDMARK_COLUMNS].to_numpy(dtype=float).reshape(-1, NUM_LANDMARKS, 3)
    hands = df["handedness"].to_numpy()
    feats, keep, small = [], [], 0
    for i in range(len(raw)):
        f = extract_features(raw[i], hands[i])
        if f is not None:
            feats.append(f)
            keep.append(i)
            small += int(hand_size(raw[i]) < MIN_HAND_SIZE)
    used = df.iloc[keep].reset_index(drop=True)
    X = np.vstack(feats) if feats else np.empty((0, len(FEATURE_NAMES)))
    stats = {
        "rows_total": int(len(df)),
        "rows_dropped_invalid": int(len(df) - len(keep)),
        "rows_used": int(len(keep)),
        "rows_small_hand": int(small),
        "rows_per_label": {k: int(v) for k, v in used["label"].value_counts().sort_index().items()},
        "rows_per_person": {k: int(v) for k, v in used["person"].value_counts().sort_index().items()},
        "rows_per_handedness": {k: int(v) for k, v in used["handedness"].value_counts().sort_index().items()},
    }
    print(
        f"Missing-value handling: dropped {stats['rows_dropped_invalid']} of {stats['rows_total']} "
        f"frames (invalid/degenerate landmarks); {stats['rows_used']} used, "
        f"{stats['rows_small_hand']} of them with a small (far) hand."
    )
    return X, used["label"].to_numpy(), used["person"].to_numpy(), stats


# --------------------------------------------------------------------------- models
def get_search_spaces(quick: bool, seed: int):
    scaler = lambda: StandardScaler()  # noqa: E731
    rf = Pipeline([("scaler", scaler()), ("clf", RandomForestClassifier(random_state=seed, n_jobs=1))])
    # Calibrated probabilities (replaces the deprecated SVC(probability=True)); the wrapper sizes
    # its calibration folds per fit so rare classes inside inner folds cannot crash it.
    svm = Pipeline([("scaler", scaler()), ("clf", AdaptiveCalibratedSVC(random_state=seed))])
    if quick:
        rf_grid = {"clf__n_estimators": [50], "clf__max_depth": [None, 10], "clf__min_samples_leaf": [1, 2]}
        svm_grid = {"clf__C": [1, 10], "clf__gamma": ["scale"]}
    else:
        rf_grid = {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [None, 10, 20],
            "clf__min_samples_leaf": [1, 2, 4],
        }
        svm_grid = {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1, 1]}
    return {"random_forest": (rf, rf_grid), "svm": (svm, svm_grid)}


def make_cv(groups_train, y_train, seed: int, use_groups: bool):
    if use_groups:
        n_groups = len(np.unique(groups_train))
        return GroupKFold(n_splits=min(5, n_groups))
    min_count = int(np.min(np.bincount(y_train)))
    return StratifiedKFold(n_splits=max(2, min(5, min_count)), shuffle=True, random_state=seed)


def inference_ms(pipeline, X, n: int = 100) -> float:
    """Mean latency (ms) of a single-sample ``predict_proba`` call."""
    sample = X[: min(n, len(X))]
    if len(sample) == 0:
        return float("nan")
    pipeline.predict_proba(sample[:1])  # warm-up
    t0 = time.perf_counter()
    for row in sample:
        pipeline.predict_proba(row.reshape(1, -1))
    return (time.perf_counter() - t0) / len(sample) * 1000.0


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if np.isnan(v) else v
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def love_you_victory_confusion(cm, labels, model_title, figure_path) -> dict:
    """Confusion between LOVE_YOU and VICTORY on the test split (rate None if a class is absent/unsupported)."""
    out = {"model": model_title, "figure": str(figure_path), "rate": None,
           "love_you_as_victory": None, "victory_as_love_you": None}
    if "LOVE_YOU" not in labels or "VICTORY" not in labels:
        return out
    cm = np.asarray(cm)
    li, vi = labels.index("LOVE_YOU"), labels.index("VICTORY")
    n_l, n_v = int(cm[li].sum()), int(cm[vi].sum())
    if n_l:
        out["love_you_as_victory"] = float(cm[li, vi] / n_l)
    if n_v:
        out["victory_as_love_you"] = float(cm[vi, li] / n_v)
    if n_l + n_v:
        out["rate"] = float((cm[li, vi] + cm[vi, li]) / (n_l + n_v))
    out["support"] = {"LOVE_YOU": n_l, "VICTORY": n_v}
    return out


# --------------------------------------------------------------------------- plots
def plot_confusion(cm, labels, title, path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax, cbar=True)
    ax.set_xlabel("Gesto predicho")
    ax.set_ylabel("Gesto real")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_class_distribution(df_used: pd.DataFrame, labels, path):
    counts = pd.crosstab(df_used["label"], df_used["person"]).reindex(labels).fillna(0)
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
    ax.set_xlabel("Gesto")
    ax.set_ylabel("Número de fotogramas")
    ax.set_title("Distribución de clases por persona")
    ax.legend(title="Persona")
    plt.setp(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_learning_curve(estimator, X, y, groups, cv, model_title, path, n_jobs, seed):
    sizes, train_scores, val_scores = learning_curve(
        clone(estimator),
        X,
        y,
        groups=groups,
        cv=cv,
        scoring="f1_macro",
        train_sizes=np.linspace(0.2, 1.0, 5),
        shuffle=True,
        random_state=seed,
        n_jobs=n_jobs,
        error_score=np.nan,
    )
    tr_m, tr_s = np.nanmean(train_scores, axis=1), np.nanstd(train_scores, axis=1)
    va_m, va_s = np.nanmean(val_scores, axis=1), np.nanstd(val_scores, axis=1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sizes, tr_m, "o-", color="tab:blue", label="Entrenamiento")
    ax.fill_between(sizes, tr_m - tr_s, tr_m + tr_s, alpha=0.15, color="tab:blue")
    ax.plot(sizes, va_m, "o-", color="tab:orange", label="Validación cruzada")
    ax.fill_between(sizes, va_m - va_s, va_m + va_s, alpha=0.15, color="tab:orange")
    ax.set_xlabel("Tamaño del conjunto de entrenamiento (fotogramas)")
    ax.set_ylabel("F1 macro")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Curva de aprendizaje: {model_title}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {
        "train_sizes": sizes,
        "train_f1_macro_mean": tr_m,
        "val_f1_macro_mean": va_m,
    }


def plot_model_comparison(results, path):
    names = [MODEL_TITLES[k] for k in results]
    metrics = [
        ("CV F1 macro (entrenamiento)", [results[k]["cv_f1_macro_mean"] for k in results]),
        ("Prueba: exactitud", [results[k]["test_accuracy"] for k in results]),
        ("Prueba: F1 macro", [results[k]["test_f1_macro"] for k in results]),
    ]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, vals) in enumerate(metrics):
        bars = ax.bar(x + (i - 1) * width, vals, width, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Puntaje")
    ax.set_title("Comparación de modelos de reconocimiento de gestos")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def train(args: argparse.Namespace) -> dict:
    seed = args.seed
    df = load_dataset(Path(args.data))
    X, y_str, groups, stats = build_features(df)
    if len(X) == 0:
        raise RuntimeError("no usable frames after dropping missing landmarks")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_str)
    labels = list(encoder.classes_)
    if len(labels) < 2:
        raise RuntimeError(f"need at least 2 classes, found {labels}")
    used_df = pd.DataFrame({"person": groups, "label": y_str})
    missing_labels = [l for l in LABELS if l not in labels]
    if missing_labels:
        warnings.warn(
            f"No usable data for label(s) {missing_labels}: they will not be in the model. "
            "Record them with brain.gestures.capture and retrain.",
            stacklevel=2,
        )
        print(f"WARNING: no data for labels {missing_labels}; training without them.")

    persons = sorted(set(groups))
    use_groups = len(persons) >= 3
    if use_groups:
        test_person = args.test_person or persons[-1]
        if test_person not in persons:
            raise ValueError(f"--test-person {test_person!r} not in {persons}")
        test_mask = groups == test_person
        strategy = f"GroupKFold by person + hold-out person ({test_person})"
        print(f"Persons: {persons}. Hold-out test person: {test_person}")
    else:
        warnings.warn(
            f"Only {len(persons)} person(s) found (<3): falling back to StratifiedKFold and a stratified "
            "20% hold-out. Results are optimistic because consecutive frames of the same video leak "
            "between train and test.",
            stacklevel=2,
        )
        print("WARNING: fewer than 3 persons; using stratified split (optimistic estimate).")
        idx = np.arange(len(y))
        _, test_idx = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
        test_mask = np.zeros(len(y), dtype=bool)
        test_mask[test_idx] = True
        test_person = None
        strategy = "StratifiedKFold + stratified 20% hold-out (fallback, <3 persons)"

    X_tr, y_tr, g_tr = X[~test_mask], y[~test_mask], groups[~test_mask]
    X_te, y_te = X[test_mask], y[test_mask]
    print(f"Train: {len(X_tr)} frames | Test: {len(X_te)} frames | Classes: {labels}")

    cv = make_cv(g_tr, y_tr, seed, use_groups)
    fit_kwargs = {"groups": g_tr} if use_groups else {}

    results, searches = {}, {}
    for key, (pipe, grid) in get_search_spaces(args.quick, seed).items():
        print(f"GridSearchCV [{MODEL_TITLES[key]}] ...")
        gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=args.n_jobs, refit=True)
        gs.fit(X_tr, y_tr, **fit_kwargs)
        searches[key] = gs
        best = gs.best_estimator_
        pred = best.predict(X_te)
        cv_res = gs.cv_results_
        fold_scores = [
            float(cv_res[f"split{i}_test_score"][gs.best_index_]) for i in range(cv.get_n_splits(X_tr, y_tr, g_tr))
        ]
        cm = confusion_matrix(y_te, pred, labels=range(len(labels)))
        report = classification_report(
            y_te, pred, labels=list(range(len(labels))), target_names=labels, output_dict=True, zero_division=0
        )
        results[key] = {
            "model": MODEL_TITLES[key],
            "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
            "cv_f1_macro_mean": float(gs.best_score_),
            "cv_f1_macro_std": float(cv_res["std_test_score"][gs.best_index_]),
            "cv_scores": fold_scores,
            "test_accuracy": float(accuracy_score(y_te, pred)),
            "test_f1_macro": float(f1_score(y_te, pred, average="macro", labels=list(range(len(labels))), zero_division=0)),
            "per_class_report": report,
            "confusion_matrix": cm.tolist(),
            "inference_ms_per_sample": inference_ms(best, X_te),
        }
        print(
            f"  best={results[key]['best_params']} cv_f1={results[key]['cv_f1_macro_mean']:.3f} "
            f"test_acc={results[key]['test_accuracy']:.3f} test_f1={results[key]['test_f1_macro']:.3f}"
        )

    best_key = max(results, key=lambda k: results[k]["cv_f1_macro_mean"])
    best_res = results[best_key]
    print(f"Best model (by CV f1_macro): {MODEL_TITLES[best_key]}")

    # Figures
    fig_dir = Path(args.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion(
        np.array(results["random_forest"]["confusion_matrix"]), labels,
        "Matriz de confusión: Random Forest (persona de prueba)", fig_dir / "gesture_cm_rf.png",
    )
    plot_confusion(
        np.array(results["svm"]["confusion_matrix"]), labels,
        "Matriz de confusión: SVM (persona de prueba)", fig_dir / "gesture_cm_svm.png",
    )
    plot_class_distribution(used_df, labels, fig_dir / "gesture_class_distribution.png")
    plot_model_comparison(results, fig_dir / "gesture_model_comparison.png")
    lc = plot_learning_curve(
        searches[best_key].best_estimator_, X_tr, y_tr, g_tr if use_groups else None, cv,
        MODEL_TITLES[best_key], fig_dir / "gesture_learning_curve.png", args.n_jobs, seed,
    )

    cm_fig = fig_dir / ("gesture_cm_rf.png" if best_key == "random_forest" else "gesture_cm_svm.png")
    ly_v = love_you_victory_confusion(best_res["confusion_matrix"], labels, best_res["model"], cm_fig)
    if ly_v["rate"] is None:
        print("LOVE_YOU<->VICTORY confusion: n/a (a class has no test samples or no data).")
    else:
        print(
            f"LOVE_YOU<->VICTORY confusion ({best_res['model']}, test person): rate={ly_v['rate']:.3f} "
            f"(LOVE_YOU->VICTORY {ly_v['love_you_as_victory']}, VICTORY->LOVE_YOU {ly_v['victory_as_love_you']}); "
            f"matrix: {cm_fig}"
        )

    # Final model: fitted on the training split (so reported metrics describe it),
    # unless --refit-all asks for training on every person for deployment.
    final = clone(searches[best_key].best_estimator_)
    if args.refit_all:
        final.fit(X, y)
        print("Final model refit on ALL data (--refit-all); reported test metrics belong to the train-split model.")
    else:
        final = searches[best_key].best_estimator_
    artifact = {
        "pipeline": final,
        "labels": labels,
        "feature_names": list(FEATURE_NAMES),
        "model_name": MODEL_TITLES[best_key],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_path)
    print(f"Saved model to {out_path}")

    metrics = {
        "model_name": best_res["model"],
        "accuracy": best_res["test_accuracy"],
        "f1_macro": best_res["test_f1_macro"],
        "per_class_report": best_res["per_class_report"],
        "best_params": best_res["best_params"],
        "cv_scores": best_res["cv_scores"],
        "cv_f1_macro_mean": best_res["cv_f1_macro_mean"],
        "train_size": int(len(X_tr)),
        "test_size": int(len(X_te)),
        "inference_ms_per_sample": best_res["inference_ms_per_sample"],
        "labels": labels,
        "missing_labels": missing_labels,
        "love_you_victory_confusion": ly_v,
        "split": {
            "strategy": strategy,
            "persons": persons,
            "test_person": test_person,
            "cv_splits": int(cv.get_n_splits(X_tr, y_tr, g_tr)),
            "refit_all": bool(args.refit_all),
        },
        "preprocessing": stats,
        "models": results,
        "learning_curve": lc,
    }
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "gesture_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(_jsonable(metrics), fh, indent=2, ensure_ascii=False)
    print(f"Saved metrics to {metrics_dir / 'gesture_metrics.json'}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m brain.gestures.train",
        description="Train and compare RandomForest vs SVM hand-gesture classifiers.",
    )
    p.add_argument("--data", default="data/gestures", help="Directory with gesture CSV files.")
    p.add_argument("--out", default="models/gesture_model.joblib", help="Output model path.")
    p.add_argument("--figures-dir", default="report/figures", help="Directory for PNG figures.")
    p.add_argument("--metrics-dir", default="report/metrics", help="Directory for the metrics JSON.")
    p.add_argument("--test-person", default=None, help="Person held out as test (default: last alphabetically).")
    p.add_argument("--quick", action="store_true", help="Use a tiny hyperparameter grid (for smoke tests).")
    p.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs for GridSearchCV (default -1).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--refit-all", action="store_true", help="Refit the final model on all persons before saving.")
    return p


def main(argv: list[str] | None = None) -> int:
    train(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
