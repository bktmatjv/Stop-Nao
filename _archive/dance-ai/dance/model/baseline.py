"""Rule-based BASELINE for comparison with the ML models.

This estimator is ONLY used in the offline evaluation (``dance.model.train``)
to show what a simple "loud and fast => energetic" rule achieves. It is never
used for inference on the robot: the choreography style always comes from the
trained RF/SVM model.

The rule combines the two features ``bpm`` and ``rms_mean`` into one intensity
score (mean of their z-scores learned on the training fold) and learns
``n_classes - 1`` ordered thresholds on that score by grid search maximising
macro-F1 on the training data. Classes are ordered by ``dance.STYLES``
(SOFT < MODERATE < ENERGETIC).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score

from dance import STYLES

BPM_COLUMN = "bpm"
RMS_COLUMN = "rms_mean"
_MAX_CANDIDATES = 60


def _ordered_classes(y) -> np.ndarray:
    present = set(np.asarray(y).tolist())
    known = [c for c in STYLES if c in present]
    extra = sorted(present - set(STYLES))
    return np.array(known + extra, dtype=object)


class BpmRuleBaseline(ClassifierMixin, BaseEstimator):
    """Threshold rule on bpm and rms_mean (baseline, not used on the robot).

    ``X`` may be a DataFrame with ``bpm`` and ``rms_mean`` columns, or an
    ndarray together with ``feature_names`` naming its columns.
    """

    def __init__(self, feature_names=None):
        self.feature_names = feature_names

    # -- helpers ---------------------------------------------------------
    def _columns(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            names = list(X.columns)
            values = X.to_numpy(dtype=float)
        else:
            if self.feature_names is None:
                raise ValueError("feature_names is required when X is not a DataFrame (needs bpm and rms_mean)")
            names = list(self.feature_names)
            values = np.asarray(X, dtype=float)
        for col in (BPM_COLUMN, RMS_COLUMN):
            if col not in names:
                raise ValueError(f"column '{col}' is required by the baseline (got {names[:6]}...)")
        return values[:, [names.index(BPM_COLUMN), names.index(RMS_COLUMN)]]

    def _score(self, X) -> np.ndarray:
        cols = self._columns(X)
        z = (cols - self.mean_) / self.scale_
        return z.mean(axis=1)

    # -- estimator API ---------------------------------------------------
    def fit(self, X, y):
        cols = self._columns(X)
        y = np.asarray(y, dtype=object)
        self.classes_ = _ordered_classes(y)
        self.mean_ = cols.mean(axis=0)
        scale = cols.std(axis=0)
        self.scale_ = np.where(scale > 0, scale, 1.0)
        score = ((cols - self.mean_) / self.scale_).mean(axis=1)

        k = len(self.classes_)
        self.class_scores_ = np.array([score[y == c].mean() for c in self.classes_])
        if k == 1:
            self.thresholds_ = np.array([])
        else:
            self.thresholds_ = self._search_thresholds(score, y, k)
        spread = float(np.std(score)) or 1.0
        self._temperature_ = spread / 2.0
        return self

    def _search_thresholds(self, score, y, k) -> np.ndarray:
        uniq = np.unique(score)
        if len(uniq) > _MAX_CANDIDATES:
            uniq = np.quantile(score, np.linspace(0.02, 0.98, _MAX_CANDIDATES))
            uniq = np.unique(uniq)
        mids = (uniq[:-1] + uniq[1:]) / 2.0 if len(uniq) > 1 else uniq
        if len(mids) == 0:
            mids = uniq
        best, best_f1 = None, -1.0
        if k == 2:
            combos = [(t,) for t in mids]
        else:
            combos = [(a, b) for i, a in enumerate(mids) for b in mids[i + 1:]]
            if k > 3:  # not needed for 3 styles; fall back to quantiles
                qs = np.quantile(score, np.linspace(0, 1, k + 1)[1:-1])
                combos = [tuple(qs)]
        for combo in combos:
            idx = np.searchsorted(np.asarray(combo), score)
            pred = self.classes_[idx]
            f1 = f1_score(y, pred, average="macro", labels=self.classes_, zero_division=0)
            if f1 > best_f1:
                best, best_f1 = combo, f1
        if best is None:  # degenerate (single unique score)
            best = tuple(np.quantile(score, np.linspace(0, 1, k + 1)[1:-1]))
        return np.asarray(best, dtype=float)

    def predict(self, X):
        self._check_fitted()
        idx = np.searchsorted(self.thresholds_, self._score(X))
        return self.classes_[idx]

    def predict_proba(self, X):
        """Soft assignment: softmax of negative distance to each class mean score."""
        self._check_fitted()
        score = self._score(X)[:, None]
        logits = -((score - self.class_scores_[None, :]) ** 2) / (2.0 * self._temperature_**2)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        return p / p.sum(axis=1, keepdims=True)

    def _check_fitted(self):
        if not hasattr(self, "thresholds_"):
            raise RuntimeError("BpmRuleBaseline is not fitted")
