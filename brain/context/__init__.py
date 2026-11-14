"""Contextual action decision for the NAO hand-gesture pilot.

The robot does NOT execute a gesture blindly: it first validates its
surroundings (sonars, motion state, posture, person proximity).

Modules:
    intents         gesture -> intended movement (spec S2)
    safety          hard validation layer, ``validate(action, ctx) -> Decision`` (spec S3/S4)
    decider         runtime ``ContextDecider`` (learned model or rules, then safety)
    collect         CLI used by the team to record real, human-labeled scenarios
    make_synthetic  synthetic augmentation dataset (NOT real data)
    train           trains/compares DecisionTree vs RandomForest vs rule-only policy
"""

GESTURES = (
    "THUMBS_UP",
    "OPEN_PALM",
    "POINT_LEFT",
    "POINT_RIGHT",
    "VICTORY",
    "FIST",
    "LOVE_YOU",
    "NONE",
)
ACTIONS = (
    "ADVANCE",
    "ADVANCE_SHORT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "STOP",
    "WAVE",
    "GREET",
    "SIT",
    "STAND",
    "WARN_OBSTACLE",
    "WARN_TOO_CLOSE",
    "WARN_MUST_STOP",
    "WARN_MUST_STAND",
    "ASK_CONFIRM",
    "WAIT",
)
FEATURE_COLUMNS = (
    "gesture",
    "gesture_confidence",
    "sonar_left",
    "sonar_right",
    "hand_size",
    "moving",
    "standing",
)
SCENARIO_COLUMNS = (*FEATURE_COLUMNS, "action", "scenario_id")
