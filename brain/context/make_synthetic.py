"""Synthetic augmentation dataset for the contextual action model.

IMPORTANT (report): this data is SYNTHETIC. The primary dataset is the one the
team records with the robot (``collect.py``) where a human labels what NAO
SHOULD do. This generator only augments it, and every row carries
``source=synthetic``. The report MUST state clearly what is real vs synthetic
(row counts by source, metrics on the real rows only) and must not present
synthetic accuracy as evidence about the real robot.

How it avoids "memorizing the rules": feature values are sampled from
realistic distributions (sonar range of the NAO, hand sizes seen by a webcam),
the label comes from a reference *human policy* written as how an operator
would judge each situation (with its own, slightly different tolerances than
the hard safety layer), and then label noise is injected (``--noise``) to
simulate operator disagreement/mistakes. The model therefore has to
generalize instead of reproducing a deterministic function.

Usage:
    python -m brain.context.make_synthetic --n 600 --out data/context/synthetic.csv --noise 0.1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import ACTIONS, GESTURES, SCENARIO_COLUMNS
from .intents import intent_for

LOW_CONFIDENCE = 0.55  # an operator is a bit more lenient than the 0.6 safety limit
FRONT_BLOCKED = 0.30  # operator refuses to advance below this front distance
FRONT_FREE = 0.50  # ... and advances a full step above this one
MIN_STEP = 0.05
SIDE_BLOCKED = 0.22  # a turn towards a blocked side is judged as an obstacle
HAND_TOO_CLOSE = 0.26


def reference_policy(row: dict) -> str:
    """How a careful human operator would label the situation."""
    gesture = row["gesture"]
    standing = bool(row["standing"])
    if gesture == "NONE":
        return "WAIT"
    intent = intent_for(gesture, standing)
    if intent == "STOP":
        return "STOP"
    if row["gesture_confidence"] < LOW_CONFIDENCE:
        return "ASK_CONFIRM"
    if intent == "ADVANCE":
        front = min(row["sonar_left"], row["sonar_right"])
        if front < FRONT_BLOCKED:
            return "WARN_OBSTACLE"
        if not standing:
            return "WARN_MUST_STAND"
        if front < FRONT_FREE:
            return "ADVANCE_SHORT" if front - FRONT_BLOCKED >= MIN_STEP else "WARN_OBSTACLE"
        return "ADVANCE"
    if intent in ("TURN_LEFT", "TURN_RIGHT"):
        side = row["sonar_left"] if intent == "TURN_LEFT" else row["sonar_right"]
        if side < SIDE_BLOCKED:
            return "WARN_OBSTACLE"
        return intent if standing else "WARN_MUST_STAND"
    if intent == "WAVE":
        if row["moving"]:
            return "WARN_MUST_STOP"
        if row["hand_size"] > HAND_TOO_CLOSE:
            return "WARN_TOO_CLOSE"
        return "WAVE" if standing else "WARN_MUST_STAND"
    if intent == "GREET":
        return "WARN_MUST_STOP" if row["moving"] else "GREET"
    if intent in ("SIT", "STAND"):
        return "WARN_MUST_STOP" if row["moving"] else intent
    return "WAIT"


def generate(n: int, noise: float = 0.1, seed: int = 0) -> pd.DataFrame:
    """Return ``n`` synthetic rows with the scenario schema plus ``source``."""
    if not 0.0 <= noise <= 1.0:
        raise ValueError("noise must be in [0, 1]")
    rng = np.random.default_rng(seed)
    gesture_p = np.array([0.20, 0.11, 0.12, 0.12, 0.12, 0.11, 0.11, 0.11])
    rows = []
    for i in range(n):
        gesture = str(rng.choice(GESTURES, p=gesture_p / gesture_p.sum()))
        confidence = 0.0 if gesture == "NONE" else float(np.clip(rng.beta(6, 1.5), 0.2, 1.0))
        # sonar: mostly free space, sometimes close obstacles; NAO range ~0.25-2.5 m
        sonars = []
        for _ in range(2):
            if rng.random() < 0.30:
                sonars.append(float(rng.uniform(0.10, 0.60)))
            else:
                sonars.append(float(np.clip(rng.gamma(4.0, 0.35), 0.45, 2.5)))
        hand = float(np.clip(rng.normal(0.17, 0.08), 0.04, 0.45))
        row = {
            "gesture": gesture,
            "gesture_confidence": round(confidence, 3),
            "sonar_left": round(sonars[0], 3),
            "sonar_right": round(sonars[1], 3),
            "hand_size": round(hand, 3),
            "moving": int(rng.random() < 0.25),
            "standing": int(rng.random() < 0.85),
        }
        action = reference_policy(row)
        if rng.random() < noise:  # simulated operator disagreement
            action = str(rng.choice([a for a in ACTIONS if a != action]))
        row["action"] = action
        row["scenario_id"] = f"syn_{i:05d}"
        row["source"] = "synthetic"
        rows.append(row)
    return pd.DataFrame(rows, columns=[*SCENARIO_COLUMNS, "source"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate a SYNTHETIC context dataset (augmentation only).")
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--out", required=True)
    parser.add_argument("--noise", type=float, default=0.1, help="label noise probability")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    df = generate(args.n, args.noise, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} synthetic rows to {out}")
    print(df["action"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
