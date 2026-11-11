"""Contextual action decision for the NAO gesture pilot.

Modules:
    safety          hard safety override (ADVANCE + obstacle -> WARN_OBSTACLE)
    collect         CLI used by the team to record real, human-labeled scenarios
    make_synthetic  synthetic augmentation dataset (NOT real data)
    train           trains/compares DecisionTree vs RandomForest
    decider         runtime ``ContextDecider`` (model + safety layer)
"""

ACTIONS = (
    "ADVANCE",
    "TURN_LEFT",
    "TURN_RIGHT",
    "STOP",
    "WARN_OBSTACLE",
    "ASK_CLOSER",
    "WAIT",
)
GESTURES = ("ADVANCE", "STOP", "TURN_LEFT", "TURN_RIGHT", "NONE")
SCENARIO_COLUMNS = (
    "gesture",
    "gesture_confidence",
    "sonar_left",
    "sonar_right",
    "person_distance",
    "moving",
    "action",
    "scenario_id",
)
FEATURE_COLUMNS = (
    "gesture",
    "gesture_confidence",
    "sonar_left",
    "sonar_right",
    "person_distance",
    "moving",
)
