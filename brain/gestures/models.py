"""Estimators shared by training and inference.

They live in their own module (not in ``train.py``) so that a model saved by
``python -m brain.gestures.train`` can be unpickled by the predictor: classes
defined in a ``__main__`` script cannot be.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted


class AdaptiveCalibratedSVC(ClassifierMixin, BaseEstimator):
    """RBF SVM with calibrated probabilities that tolerates tiny classes.

    ``SVC(probability=True)`` is deprecated in scikit-learn 1.9, and
    ``CalibratedClassifierCV`` raises when a class has fewer samples than the
    number of calibration folds (this happens inside GroupKFold / learning-curve
    fits with rare classes). The calibration folds are therefore sized per fit:
    ``cv = min(max_cv, smallest class count)``. If some class has a single
    sample (``cv < 2``) calibration is impossible and probabilities fall back
    to a softmax over the SVM decision function.
    """

    def __init__(self, C: float = 1.0, gamma="scale", kernel: str = "rbf", max_cv: int = 3,
                 random_state: int | None = None):
        self.C = C
        self.gamma = gamma
        self.kernel = kernel
        self.max_cv = max_cv
        self.random_state = random_state

    def _base(self) -> SVC:
        return SVC(C=self.C, gamma=self.gamma, kernel=self.kernel, random_state=self.random_state)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        self.classes_ = classes
        self.n_features_in_ = X.shape[1]
        self.calibration_cv_ = int(min(self.max_cv, counts.min()))
        if len(classes) >= 2 and self.calibration_cv_ >= 2:
            self.model_ = CalibratedClassifierCV(self._base(), ensemble=False, cv=self.calibration_cv_)
        else:
            self.model_ = self._base()
        self.model_.fit(X, y)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "model_")
        X = np.asarray(X, dtype=float)
        if hasattr(self.model_, "predict_proba"):
            return self.model_.predict_proba(X)
        scores = np.asarray(self.model_.decision_function(X), dtype=float)
        if scores.ndim == 1:  # binary: score of the positive class
            scores = np.column_stack([-scores, scores])
        scores = scores - scores.max(axis=1, keepdims=True)
        e = np.exp(scores)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X):
        check_is_fitted(self, "model_")
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

