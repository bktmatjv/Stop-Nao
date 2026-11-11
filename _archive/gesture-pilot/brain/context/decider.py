"""Runtime decision: learned model first, then the hard safety layer."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .safety import apply_safety


class ContextDecider:
    """Loads ``models/context_model.joblib`` and decides the NAO action.

    The artifact is a dict ``{pipeline, feature_columns, labels, model_name}``.
    ``bundle`` may be passed instead of ``model_path`` (used by tests).
    """

    def __init__(self, model_path=None, bundle: dict | None = None, threshold: float = 0.30):
        if bundle is None:
            if model_path is None:
                raise ValueError("model_path or bundle is required")
            bundle = joblib.load(Path(model_path))
        self.pipeline = bundle["pipeline"]
        self.feature_columns = list(bundle["feature_columns"])
        self.labels = list(bundle.get("labels", []))
        self.model_name = bundle.get("model_name", "unknown")
        self.threshold = threshold

    def _predict(self, features: dict) -> str:
        missing = [c for c in self.feature_columns if c not in features]
        if missing:
            raise KeyError(f"missing features: {missing}")
        row = pd.DataFrame([{c: features[c] for c in self.feature_columns}])
        return str(self.pipeline.predict(row)[0])

    def decide_with_reason(self, features: dict) -> tuple[str, str]:
        """Return ``(action, reason)``; reason says whether safety overrode the model."""
        model_action = self._predict(features)
        final = apply_safety(
            model_action,
            features.get("sonar_left"),
            features.get("sonar_right"),
            self.threshold,
        )
        if final != model_action:
            return final, f"safety override: model said {model_action} but obstacle/invalid sonar"
        return final, f"model ({self.model_name})"

    def decide(self, features: dict) -> str:
        return self.decide_with_reason(features)[0]
