"""Hard safety layer (spec S3).

"si recibe la orden gestual de avanzar, pero detecta un obstaculo, decide
permanecer detenido y emitir una advertencia."

This layer runs AFTER the learned model and cannot be bypassed by it:
if the action is ``ADVANCE`` and the closest sonar reading is below the
threshold (or a reading is invalid) the action becomes ``WARN_OBSTACLE``.
Every other action is returned unchanged.
"""

from __future__ import annotations

import math

DEFAULT_THRESHOLD = 0.30  # meters


def _is_valid(reading) -> bool:
    """A sonar reading is valid when it is a finite number > 0."""
    try:
        value = float(reading)
    except (TypeError, ValueError):
        return False
    return not math.isnan(value) and not math.isinf(value) and value > 0.0


def is_blocked(sonar_left, sonar_right, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """True when advancing is unsafe: an invalid reading or a reading < threshold."""
    if not (_is_valid(sonar_left) and _is_valid(sonar_right)):
        return True
    return min(float(sonar_left), float(sonar_right)) < threshold


def apply_safety(action: str, sonar_left, sonar_right, threshold: float = DEFAULT_THRESHOLD) -> str:
    """Return ``WARN_OBSTACLE`` for an unsafe ADVANCE, else ``action`` unchanged."""
    if action == "ADVANCE" and is_blocked(sonar_left, sonar_right, threshold):
        return "WARN_OBSTACLE"
    return action
