"""Inference: predict the dance style of audio segments with the TRAINED model.

The style always comes from the saved ML pipeline; BPM rules are never used as
a substitute (they only exist as a comparison baseline in ``baseline.py``).
Features are extracted with the same ``dance.audio.features`` functions used to
build the training dataset, and the feature names stored in the artifact are
checked against ``FEATURE_NAMES`` so train and inference cannot silently drift.

CLI::

    python -m dance.model.infer --song path/to/song.wav [--model models/dance_model.joblib]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from dance import config

DEFAULT_MODEL_PATH = Path("models/dance_model.joblib")
REQUIRED_KEYS = {"pipeline", "labels", "feature_names", "model_name", "segment_seconds"}


@dataclass(frozen=True)
class Prediction:
    style: str
    confidence: Optional[float]
    bpm: float
    probabilities: dict
    #: name -> value of the features fed to the model (kept for reports and logs;
    #: they are already computed here, so callers never need to extract them again)
    features: Optional[dict] = None


@dataclass(frozen=True)
class SegmentPrediction:
    start_s: float
    end_s: float
    style: str
    confidence: Optional[float]
    bpm: float
    features: Optional[dict] = None


def _load_features_module():
    """Import the audio feature module lazily (keeps training independent of librosa)."""
    from dance.audio import features

    return features


class DancePredictor:
    """Wraps the saved artifact ``{pipeline, labels, feature_names, model_name, segment_seconds}``."""

    def __init__(self, model_path=None, *, artifact: Optional[dict] = None):
        if artifact is None:
            path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
            if not path.is_file():
                raise FileNotFoundError(f"Model file not found: {path} (run python -m dance.model.train first)")
            artifact = joblib.load(path)
        missing = REQUIRED_KEYS - set(artifact)
        if missing:
            raise ValueError(f"Invalid model artifact, missing keys: {sorted(missing)}")
        self.artifact = artifact
        self.pipeline = artifact["pipeline"]
        self.labels = [str(x) for x in artifact["labels"]]
        self.feature_names = list(artifact["feature_names"])
        self.model_name = artifact["model_name"]
        self.segment_seconds = float(artifact.get("segment_seconds", config.SEGMENT_SECONDS))

    # ------------------------------------------------------------------
    def _features(self):
        feats = _load_features_module()
        names = getattr(feats, "FEATURE_NAMES", None)
        if names is not None and list(names) != self.feature_names:
            raise ValueError(
                "The model was trained with different features than dance.audio.features.FEATURE_NAMES; "
                "retrain the model (features must be exactly the same)."
            )
        return feats

    def predict_segment(self, y, sr) -> Prediction:
        feats = self._features()
        x = np.asarray(feats.extract_features(y, sr), dtype=float).reshape(1, -1)
        if x.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {x.shape[1]}")
        feature_values = {n: float(v) for n, v in zip(self.feature_names, x[0])}
        bpm = float(feats.estimate_bpm(y, sr))
        if hasattr(self.pipeline, "predict_proba"):
            proba = np.asarray(self.pipeline.predict_proba(x)[0], dtype=float)
            classes = [str(c) for c in getattr(self.pipeline, "classes_", self.labels)]
            probabilities = {c: float(p) for c, p in zip(classes, proba)}
            style = classes[int(np.argmax(proba))]
            confidence = float(proba.max())
        else:
            style = str(self.pipeline.predict(x)[0])
            probabilities = {style: 1.0}
            confidence = None
        return Prediction(style=style, confidence=confidence, bpm=bpm,
                          probabilities=probabilities, features=feature_values)

    def analyze_file(self, path) -> list:
        feats = self._features()
        y, sr = feats.load_audio(str(path))
        out = []
        for start_s, end_s, samples in feats.segment_audio(y, sr, seconds=self.segment_seconds):
            p = self.predict_segment(samples, sr)
            out.append(
                SegmentPrediction(
                    start_s=float(start_s), end_s=float(end_s), style=p.style,
                    confidence=p.confidence, bpm=p.bpm, features=p.features,
                )
            )
        return out


# ---------------------------------------------------------------------- smoothing


def smooth_predictions(preds, min_conf: float = 0.5) -> list:
    """Avoid abrupt style changes.

    (a) A prediction whose confidence is below ``min_conf`` keeps the previous
        (already smoothed) style. A missing confidence (None) is trusted.
    (b) A 3-wide majority filter then removes one-segment flips; on ties the
        segment keeps its own style. BPM, times and confidence are preserved.
    The input list is not modified.
    """
    preds = list(preds)
    if not preds:
        return []

    held = []
    for i, p in enumerate(preds):
        if i > 0 and p.confidence is not None and p.confidence < min_conf:
            held.append(replace(p, style=held[-1].style))
        else:
            held.append(p)

    styles = [p.style for p in held]
    out = []
    for i, p in enumerate(held):
        window = styles[max(0, i - 1): i + 2]
        counts = Counter(window)
        top = max(counts.values())
        winners = {s for s, c in counts.items() if c == top}
        chosen = p.style if p.style in winners else sorted(winners, key=window.index)[0]
        out.append(p if chosen == p.style else replace(p, style=chosen))
    return out


# ---------------------------------------------------------------------- CLI


def _fmt_conf(conf) -> str:
    return "  n/a" if conf is None else f"{conf:5.2f}"


def format_table(raw, smoothed) -> str:
    header = f"{'start_s':>8} {'end_s':>8} {'bpm':>7} {'raw style':<10} {'conf':>5} {'smoothed':<10}"
    lines = [header, "-" * len(header)]
    for r, s in zip(raw, smoothed):
        lines.append(
            f"{r.start_s:8.1f} {r.end_s:8.1f} {r.bpm:7.1f} {r.style:<10} {_fmt_conf(r.confidence):>5} {s.style:<10}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m dance.model.infer",
        description="Predict the dance style of every segment of an audio file with the trained model.",
    )
    p.add_argument("--song", required=True, help="path to an audio file (wav/mp3/...)")
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="trained model artifact (joblib)")
    p.add_argument("--min-conf", type=float, default=0.5, help="confidence below which the previous style is kept")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        predictor = DancePredictor(args.model)
        raw = predictor.analyze_file(args.song)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    smoothed = smooth_predictions(raw, min_conf=args.min_conf)
    print(f"Model: {predictor.model_name} | segment: {predictor.segment_seconds:g}s | song: {args.song}")
    print(format_table(raw, smoothed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
