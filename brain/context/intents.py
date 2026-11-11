"""Gesture -> intended movement (spec S2).

Pointing semantics: the person faces the robot, so the person's left is the
robot's right. With ``POINT_TURNS_TOWARD_POINTED_SIDE = True`` the robot turns
TOWARD where the person points (POINT_LEFT -> TURN_RIGHT in the robot frame).
Set it to False to map the literal side instead.
"""

from __future__ import annotations

POINT_TURNS_TOWARD_POINTED_SIDE = True

# FIST is resolved at runtime (SIT if standing, else STAND); NONE means "no order".
GESTURE_TO_INTENT = {
    "THUMBS_UP": "ADVANCE",
    "OPEN_PALM": "STOP",
    "POINT_LEFT": "TURN_LEFT",
    "POINT_RIGHT": "TURN_RIGHT",
    "VICTORY": "WAVE",
    "FIST": "TOGGLE_POSTURE",
    "TWO_FINGERS": "GREET",
    "NONE": None,
}

_MIRROR = {"TURN_LEFT": "TURN_RIGHT", "TURN_RIGHT": "TURN_LEFT"}


def intent_for(gesture: str, standing: bool) -> str | None:
    """Return the requested action for ``gesture`` (None = no order)."""
    intent = GESTURE_TO_INTENT.get(gesture)
    if intent == "TOGGLE_POSTURE":
        return "SIT" if standing else "STAND"
    if intent in _MIRROR and POINT_TURNS_TOWARD_POINTED_SIDE:
        return _MIRROR[intent]
    return intent
