"""Shared vocabulary for modes, input sources and intents (gesture and voice both emit these)."""

from __future__ import annotations

# Two independent axes.
MODE_MOVEMENT = "MOVEMENT"
MODE_MUSIC = "MUSIC"
INPUT_GESTURES = "GESTURES"
INPUT_VOICE = "VOICE"

# Movement intents (already used by brain/context).
ADVANCE, STOP, TURN_LEFT, TURN_RIGHT = "ADVANCE", "STOP", "TURN_LEFT", "TURN_RIGHT"
SIT, STAND, TOGGLE_POSTURE, WAVE = "SIT", "STAND", "TOGGLE_POSTURE", "WAVE"
# New intents.
CELEBRATE = "CELEBRATE"            # We Are the Champions + special dance
TOGGLE_INPUT = "TOGGLE_INPUT"      # GESTURES <-> VOICE
TOGGLE_MODE = "TOGGLE_MODE"        # MOVEMENT <-> MUSIC (two thumbs up)
SET_MODE_MUSIC = "SET_MODE_MUSIC"
SET_MODE_MOVEMENT = "SET_MODE_MOVEMENT"
SET_INPUT_GESTURES = "SET_INPUT_GESTURES"
SET_INPUT_VOICE = "SET_INPUT_VOICE"
MUSIC_PLAY, MUSIC_PAUSE = "MUSIC_PLAY", "MUSIC_PAUSE"
MUSIC_NEXT, MUSIC_PREV, MUSIC_STOP = "MUSIC_NEXT", "MUSIC_PREV", "MUSIC_STOP"
UNKNOWN = "UNKNOWN"

# Intents the voice classifier may return (JSON-constrained to exactly this set).
VOICE_INTENTS = (
    ADVANCE, STOP, TURN_LEFT, TURN_RIGHT, SIT, STAND, WAVE, CELEBRATE,
    MUSIC_PLAY, MUSIC_PAUSE, MUSIC_NEXT, MUSIC_PREV, MUSIC_STOP,
    SET_MODE_MUSIC, SET_MODE_MOVEMENT, SET_INPUT_GESTURES, UNKNOWN,
)

# Honoured even while the other input source is active (S: never trap the user, never disable e-stop).
ALWAYS_ACTIVE_GESTURES = ("VICTORY", "OPEN_PALM")  # V = toggle input, palm = emergency stop

# Gesture meaning per mode. Same gesture, different intent.
MOVEMENT_GESTURES = {
    "THUMBS_UP": ADVANCE, "OPEN_PALM": STOP, "POINT_LEFT": TURN_LEFT, "POINT_RIGHT": TURN_RIGHT,
    "FIST": TOGGLE_POSTURE, "LOVE_YOU": CELEBRATE, "VICTORY": TOGGLE_INPUT,
}
MUSIC_GESTURES = {
    "THUMBS_UP": MUSIC_PLAY, "OPEN_PALM": MUSIC_PAUSE, "POINT_LEFT": MUSIC_PREV,
    "POINT_RIGHT": MUSIC_NEXT, "FIST": MUSIC_STOP, "LOVE_YOU": CELEBRATE, "VICTORY": TOGGLE_INPUT,
}
