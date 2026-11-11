"""Synthetic augmentation dataset for the contextual action model.

IMPORTANT (report): this data is SYNTHETIC. The primary dataset is the one the
team records with the robot (``collect.py``) where a human labels what NAO
SHOULD do. This generator only augments it, and every row carries
``source=synthetic`` so the report can state clearly what is real vs synthetic
and so results can be broken down by source.

How it avoids "memorizing the rules": feature values are sampled from
realistic distributions (sonar range of the NAO, shoulder-width ratios), the
label comes from a reference *human policy* written as how an operator would
judge each situation, and then label noise is injected (``--noise``) to
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

FAR_PERSON = 0.08  # shoulder width / frame width below this: person too far
LOW_CONFIDENCE = 0.55
SIDE_BLOCKED = 0.20  # a turn towards a blocked side is judged as an obstacle


def reference_policy(row: dict) -> str:
    """How a careful human operator would label the situation."""
    gesture = row["gesture"]
    conf = row["gesture_confidence"]
    left, right = row["sonar_left"], row["sonar_right"]
    if gesture == "NONE":
        return "STOP" if row["moving"] else "WAIT"
    if conf < LOW_CONFIDENCE:
        return "WAIT"
    if row["person_distance"] < FAR_PERSON:
        return "ASK_CLOSER"
    if gesture == "STOP":
        return "STOP"
    if gesture == "ADVANCE":
        return "WARN_OBSTACLE" if min(left, right) < 0.30 else "ADVANCE"
    if gesture == "TURN_LEFT":
        return "WARN_OBSTACLE" if left < SIDE_BLOCKED else "TURN_LEFT"
    if gesture == "TURN_RIGHT":
        return "WARN_OBSTACLE" if right < SIDE_BLOCKED else "TURN_RIGHT"
    return "WAIT"


def generate(n: int, noise: float = 0.1, seed: int = 0) -> pd.DataFrame:
    """Return ``n`` synthetic rows with the scenario schema plus ``source``."""
    if not 0.0 <= noise <= 1.0:
        raise ValueError("noise must be in [0, 1]")
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        gesture = str(rng.choice(GESTURES, p=[0.28, 0.18, 0.17, 0.17, 0.20]))
        confidence = float(np.clip(rng.beta(6, 1.5), 0.2, 1.0))
        # sonar: mostly free space, sometimes close obstacles; NAO range ~0.25-2.5 m
        sonars = []
        for _ in range(2):
            if rng.random() < 0.30:
                sonars.append(float(rng.uniform(0.10, 0.45)))
            else:
                sonars.append(float(np.clip(rng.gamma(4.0, 0.35), 0.45, 2.5)))
        person_distance = float(np.clip(rng.normal(0.16, 0.07), 0.03, 0.45))
        row = {
            "gesture": gesture,
            "gesture_confidence": round(confidence, 3),
            "sonar_left": round(sonars[0], 3),
            "sonar_right": round(sonars[1], 3),
            "person_distance": round(person_distance, 3),
            "moving": int(rng.random() < 0.25),
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
