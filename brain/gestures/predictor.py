"""Real-time hand-gesture prediction with a sliding-window majority vote."""

from __future__ import annotations

from collections import Counter, deque

import joblib
import numpy as np

from .features import extract_features

NONE_LABEL = "NONE"


class GesturePredictor:
    """Wraps the saved gesture model (dict {pipeline, labels, feature_names, model_name}).

    ``update(landmarks, handedness)`` is called once per camera frame (with the
    (21, 3) hand landmarks and MediaPipe's ``"Left"``/``"Right"`` label, or
    ``landmarks=None`` when no hand is detected) and returns
    ``(label, confidence)``. The label is the majority vote over the last
    ``window`` frames; confidence is the mean probability of the frames that
    voted for the winner. ``("NONE", conf)`` is returned when there are too few
    frames, no usable landmarks, or the confidence is below ``min_conf``; with
    no hand at all the result is always ``("NONE", 0.0)``.
    """

    def __init__(self, model_path: str, window: int = 7, min_conf: float = 0.6):
        if window < 1:
            raise ValueError("window must be >= 1")
        artifact = joblib.load(model_path)
        self.pipeline = artifact["pipeline"]
        self.labels = list(artifact["labels"])
        self.feature_names = list(artifact["feature_names"])
        self.model_name = artifact.get("model_name", "unknown")
        self.window = window
        self.min_conf = min_conf
        self._classes = [int(c) for c in self.pipeline.classes_]
        self._votes: deque[tuple[str, float]] = deque(maxlen=window)

    def reset(self) -> None:
        self._votes.clear()

    def _predict_frame(self, landmarks: np.ndarray, handedness: str) -> tuple[str, float]:
        feats = extract_features(landmarks, handedness)
        if feats is None:
            return NONE_LABEL, 0.0
        proba = self.pipeline.predict_proba(feats.reshape(1, -1))[0]
        best = int(np.argmax(proba))
        return self.labels[self._classes[best]], float(proba[best])

    def update(self, landmarks: np.ndarray | None, handedness: str | None) -> tuple[str, float]:
        if landmarks is None:  # no hand detected: record the absence and answer NONE right away
            self._votes.append((NONE_LABEL, 0.0))
            return NONE_LABEL, 0.0
        self._votes.append(self._predict_frame(landmarks, handedness))

        min_frames = self.window // 2 + 1
        if len(self._votes) < min_frames:
            return NONE_LABEL, 0.0

        counts = Counter(label for label, _ in self._votes)
        conf_sum: dict[str, float] = {}
        for label, conf in self._votes:
            conf_sum[label] = conf_sum.get(label, 0.0) + conf
        # Majority; ties are broken by the larger accumulated confidence.
        winner = max(counts, key=lambda lb: (counts[lb], conf_sum[lb]))
        confidence = conf_sum[winner] / counts[winner]

        if confidence < self.min_conf:
            return NONE_LABEL, confidence
        return winner, confidence
