"""Runtime decision: learned model (or gesture rules) first, then the safety layer."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .intents import intent_for
from .safety import Decision, validate

_NO_TREE_NOTE = (
    "Explicación no disponible: la decisión no viene de un árbol de decisión "
    "(se usa Random Forest o las reglas por gesto)."
)


class ContextDecider:
    """Decides what NAO should do for a sensed context.

    * With a model (``model_path`` to ``models/context_model.joblib`` or an
      in-memory ``bundle`` ``{pipeline, feature_columns, labels, model_name}``)
      the requested action is predicted from ``FEATURE_COLUMNS``.
    * Without a model the requested action is the gesture intent.
    * In both cases the result ALWAYS goes through ``safety.validate``.
    """

    def __init__(self, model_path=None, bundle: dict | None = None):
        if bundle is None and model_path is not None:
            bundle = joblib.load(Path(model_path))
        self.pipeline = bundle["pipeline"] if bundle else None
        self.feature_columns = list(bundle["feature_columns"]) if bundle else []
        self.labels = list(bundle.get("labels", [])) if bundle else []
        self.model_name = bundle.get("model_name", "unknown") if bundle else "rules"

    @property
    def has_model(self) -> bool:
        return self.pipeline is not None

    def _row(self, ctx: dict) -> pd.DataFrame:
        missing = [c for c in self.feature_columns if c not in ctx]
        if missing:
            raise KeyError(f"missing features: {missing}")
        row = {c: ctx[c] for c in self.feature_columns}
        for key in ("moving", "standing"):
            if key in row:
                row[key] = int(bool(row[key]))
        return pd.DataFrame([row])

    def requested_action(self, ctx: dict) -> str:
        """The action asked for BEFORE validation (model prediction or intent)."""
        if self.has_model:
            return str(self.pipeline.predict(self._row(ctx))[0])
        return intent_for(ctx.get("gesture", "NONE"), bool(ctx.get("standing", False))) or "WAIT"

    def decide(self, ctx: dict) -> Decision:
        return validate(self.requested_action(ctx), ctx)

    def explain(self, ctx: dict) -> str:
        """Spanish text with the decision path of a DecisionTree model.

        Returns a short note when the model is a RandomForest or there is none.
        """
        clf = self.pipeline[-1] if self.has_model else None
        if clf is None or not hasattr(clf, "decision_path") or hasattr(clf, "estimators_"):
            return _NO_TREE_NOTE
        row = self._row(ctx)
        pre = self.pipeline[:-1] if len(self.pipeline.steps) > 1 else None
        x = pre.transform(row) if pre is not None else row.to_numpy()
        if hasattr(x, "toarray"):  # ColumnTransformer may return a sparse matrix
            x = x.toarray()
        names = _feature_names(pre, row, x.shape[1])
        tree = clf.tree_
        lines = []
        for node in clf.decision_path(x).indices:
            if tree.children_left[node] == tree.children_right[node]:  # leaf
                continue
            value = float(x[0, tree.feature[node]])
            threshold = float(tree.threshold[node])
            name = names[tree.feature[node]]
            if value <= threshold:
                lines.append(f"- {name} = {value:.2f} <= {threshold:.2f}")
            else:
                lines.append(f"- {name} = {value:.2f} > {threshold:.2f}")
        predicted = str(clf.predict(x)[0])
        return "\n".join([*lines, f"Resultado del árbol: {predicted}"])


def _feature_names(pre, row: pd.DataFrame, n: int) -> list[str]:
    try:
        return [str(name) for name in pre.get_feature_names_out()]
    except Exception:  # noqa: BLE001 - pipelines without named outputs
        return list(row.columns) if len(row.columns) == n else [f"x{i}" for i in range(n)]
