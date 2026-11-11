"""Train and compare DecisionTree vs RandomForest for the contextual action.

Usage:
    python -m brain.context.train --data data/context/scenarios.csv
        [--extra data/context/synthetic.csv] --out models/context_model.joblib
        --figures-dir report/figures --metrics-dir report/metrics

Method (spec S5):
  * preprocessing: one-hot of ``gesture``; numeric features pass through
    (tree-based models are scale invariant and raw thresholds stay readable
    in the plotted tree, so no scaler is needed);
  * split by ``scenario_id`` groups (GroupShuffleSplit) so rows of one scene
    never appear in both train and test (no leakage);
  * GridSearchCV with GroupKFold, scoring f1_macro, for both algorithms;
  * metrics: accuracy, f1_macro, per-class report, confusion matrices;
  * safety override rate on the test set (how often the hard safety layer
    changes the model output) and the metrics after the safety layer;
  * the saved artifact is refit on ALL rows with the best hyperparameters.

The primary data is the real, human-labeled CSV; ``--extra`` rows
(``source=synthetic``) only augment it. Metrics are also reported on the real
rows of the test split so the report can separate real from synthetic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402
from sklearn.tree import DecisionTreeClassifier, plot_tree  # noqa: E402

from . import ACTIONS, FEATURE_COLUMNS, GESTURES  # noqa: E402
from .safety import apply_safety  # noqa: E402

NUMERIC = [c for c in FEATURE_COLUMNS if c != "gesture"]


def load_data(data_path, extra_path=None) -> pd.DataFrame:
    """Load the real CSV (+ optional synthetic CSV) and validate the schema."""
    df = pd.read_csv(data_path)
    if "source" not in df.columns:
        df["source"] = "real"
    if extra_path:
        extra = pd.read_csv(extra_path)
        if "source" not in extra.columns:
            extra["source"] = "synthetic"
        df = pd.concat([df, extra], ignore_index=True)
    needed = [*FEATURE_COLUMNS, "action", "scenario_id"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    df = df.dropna(subset=needed).reset_index(drop=True)
    df["moving"] = df["moving"].astype(int)
    df["scenario_id"] = df["scenario_id"].astype(str)
    return df


def build_pipeline(clf) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("gesture", OneHotEncoder(categories=[list(GESTURES)], handle_unknown="ignore"), ["gesture"]),
            ("num", "passthrough", NUMERIC),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _metrics(y_true, y_pred) -> dict:
    labels = list(ACTIONS)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
        "n": int(len(y_true)),
    }


def _plot_cm(y_true, y_pred, title, path):
    labels = list(ACTIONS)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Accion predicha")
    ax.set_ylabel("Accion real (etiqueta humana)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train(data, extra, out, figures_dir, metrics_dir, seed=42, test_size=0.25) -> dict:
    df = load_data(data, extra)
    X, y, groups = df[list(FEATURE_COLUMNS)], df["action"], df["scenario_id"]
    n_groups = groups.nunique()
    if n_groups < 4:
        raise ValueError(f"need at least 4 distinct scenario_id groups, got {n_groups}")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_tr, y_tr, g_tr = X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]
    X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
    if not set(g_tr).isdisjoint(set(groups.iloc[test_idx])):
        raise RuntimeError("group leakage between train and test")
    cv = GroupKFold(n_splits=max(2, min(5, g_tr.nunique())))

    candidates = {
        "DecisionTree": (
            DecisionTreeClassifier(random_state=seed),
            {
                "clf__max_depth": [3, 5, 8, None],
                "clf__min_samples_leaf": [1, 3, 5],
                "clf__criterion": ["gini", "entropy"],
            },
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=seed, n_jobs=1),
            {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 6, 10],
                "clf__min_samples_leaf": [1, 3],
            },
        ),
    }

    results, fitted, preds = {}, {}, {}
    is_real = (df["source"].iloc[test_idx] == "real").to_numpy()
    for name, (clf, grid) in candidates.items():
        gs = GridSearchCV(build_pipeline(clf), grid, cv=cv, scoring="f1_macro", n_jobs=1)
        gs.fit(X_tr, y_tr, groups=g_tr)
        pred = gs.predict(X_te)
        fitted[name], preds[name] = gs, pred
        results[name] = {
            "best_params": {k.replace("clf__", ""): v for k, v in gs.best_params_.items()},
            "cv_f1_macro": float(gs.best_score_),
            "test": _metrics(y_te, pred),
        }
        if is_real.any():
            results[name]["test_real_only"] = _metrics(y_te[is_real], pred[is_real])

    best_name = max(results, key=lambda n: results[n]["cv_f1_macro"])

    # hard safety layer on the test set, using the best model
    best_pred = preds[best_name]
    safe_pred = [
        apply_safety(a, left, right)
        for a, left, right in zip(best_pred, X_te["sonar_left"], X_te["sonar_right"])
    ]
    n_override = int(sum(a != b for a, b in zip(best_pred, safe_pred)))
    n_adv = int(sum(a == "ADVANCE" for a in best_pred))
    safety = {
        "override_count": n_override,
        "override_rate_test": n_override / len(best_pred),
        "override_rate_among_advance_predictions": (n_override / n_adv) if n_adv else 0.0,
        "metrics_after_safety": _metrics(y_te, safe_pred),
    }

    # figures
    figures_dir, metrics_dir = Path(figures_dir), Path(metrics_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    _plot_cm(y_te, preds["DecisionTree"], "Matriz de confusion - Arbol de Decision (contexto)",
             figures_dir / "context_cm_dt.png")
    _plot_cm(y_te, preds["RandomForest"], "Matriz de confusion - Random Forest (contexto)",
             figures_dir / "context_cm_rf.png")

    dt_pipe = fitted["DecisionTree"].best_estimator_
    feat_names = list(dt_pipe.named_steps["pre"].get_feature_names_out())
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(dt_pipe.named_steps["clf"], max_depth=3, feature_names=feat_names,
              class_names=list(dt_pipe.named_steps["clf"].classes_), filled=True,
              rounded=True, fontsize=7, ax=ax)
    ax.set_title("Arbol de decision para la accion contextual (primeros 3 niveles)")
    fig.tight_layout()
    fig.savefig(figures_dir / "context_tree.png", dpi=120)
    plt.close(fig)

    rf_pipe = fitted["RandomForest"].best_estimator_
    imp = pd.DataFrame({
        "feature": feat_names,
        "Arbol de Decision": dt_pipe.named_steps["clf"].feature_importances_,
        "Random Forest": rf_pipe.named_steps["clf"].feature_importances_,
    }).set_index("feature").sort_values("Random Forest")
    ax = imp.plot.barh(figsize=(8, 5))
    ax.set_title("Importancia de variables - decision contextual")
    ax.set_xlabel("Importancia")
    ax.figure.tight_layout()
    ax.figure.savefig(figures_dir / "context_feature_importance.png", dpi=150)
    plt.close(ax.figure)

    comp = pd.DataFrame({
        n: {
            "CV F1-macro": results[n]["cv_f1_macro"],
            "Test accuracy": results[n]["test"]["accuracy"],
            "Test F1-macro": results[n]["test"]["f1_macro"],
        }
        for n in results
    })
    ax = comp.T.plot.bar(figsize=(7, 5), rot=0)
    ax.set_ylim(0, 1.05)
    ax.set_title("Comparacion de modelos - decision contextual")
    ax.set_ylabel("Puntaje")
    ax.figure.tight_layout()
    ax.figure.savefig(figures_dir / "context_model_comparison.png", dpi=150)
    plt.close(ax.figure)

    # final artifact: best configuration refit on all rows
    final = clone(fitted[best_name].best_estimator_).fit(X, y)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline": final,
        "feature_columns": list(FEATURE_COLUMNS),
        "labels": list(ACTIONS),
        "model_name": best_name,
    }, out)

    report = {
        "n_rows": int(len(df)),
        "n_groups": int(n_groups),
        "rows_by_source": {k: int(v) for k, v in df["source"].value_counts().items()},
        "class_counts": {k: int(v) for k, v in y.value_counts().items()},
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "split": f"GroupShuffleSplit by scenario_id, test_size={test_size}, seed={seed}",
        "cv": f"GroupKFold n_splits={cv.n_splits}, scoring=f1_macro",
        "models": results,
        "best_model": best_name,
        "safety": safety,
        "artifact_fit": "best model refit on all rows",
    }
    with open(metrics_dir / "context_metrics.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train the contextual action model.")
    p.add_argument("--data", required=True, help="real, human-labeled scenarios CSV")
    p.add_argument("--extra", default=None, help="optional synthetic augmentation CSV")
    p.add_argument("--out", default="models/context_model.joblib")
    p.add_argument("--figures-dir", default="report/figures")
    p.add_argument("--metrics-dir", default="report/metrics")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    rep = train(args.data, args.extra, args.out, args.figures_dir, args.metrics_dir, args.seed)
    for name, r in rep["models"].items():
        print(f"{name}: cv_f1={r['cv_f1_macro']:.3f} test_acc={r['test']['accuracy']:.3f} "
              f"test_f1={r['test']['f1_macro']:.3f} params={r['best_params']}")
    print(f"best={rep['best_model']} safety_override_rate={rep['safety']['override_rate_test']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
