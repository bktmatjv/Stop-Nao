"""Feature extraction from MediaPipe Hands landmarks.

Input
-----
``landmarks`` is an array of shape (21, 3): ``x, y, z`` per landmark as produced
by ``mp.solutions.hands`` (x/y normalized to [0, 1], y grows downward) computed
on the RAW (non-mirrored) camera frame. ``handedness`` is the label MediaPipe
reports for that frame, ``"Left"`` or ``"Right"``. Note that MediaPipe assumes a
mirrored (selfie) image, so on a raw frame the label is the opposite of the
physical hand; this does not matter here because the label is only used to
canonicalize the hand shape and the same raw convention is used for training
and inference.

Landmark indices: 0 wrist; thumb 1-4 (CMC, MCP, IP, TIP); index 5-8; middle
9-12; ring 13-16; pinky 17-20 (each MCP, PIP, DIP, TIP).

Feature design (documented here because it goes into the report)
----------------------------------------------------------------
* Translation and scale invariance: every coordinate is expressed relative to
  the wrist and divided by the PALM SIZE (distance wrist -> middle-finger MCP,
  landmarks 0 and 9). The z coordinate is ignored on purpose: it is noisy for
  monocular webcams and the gestures of this project do not need it. x/y are
  normalized per axis by MediaPipe, so the image aspect ratio is not corrected
  (it is the same camera at training and at inference time).
* The handedness label is NOT used by any feature. MediaPipe's Left/Right
  classification is unreliable for this task: in the team's first real
  recordings the same physical hand was labeled "Right" only while pointing to
  the right and "Left" for every other gesture, so a model that used the label
  (or mirrored the hand according to it) learned a shortcut that breaks as soon
  as the label flips. Coordinates are therefore kept in the raw image
  orientation; mirror-invariant features (finger extension, bend angles,
  fingertip distances and spreads, thumb_up_score) describe the hand shape
  regardless of which hand is used, while the coordinates still differ between
  hands, so the team records with the hand they will use in the demo.
* Pointing direction is in ABSOLUTE IMAGE terms. ``index_dir_x/y`` is the unit vector from the index MCP
  (5) to the index tip (8) and ``palm_dir_x/y`` the unit vector wrist -> middle
  MCP, both in raw image coordinates (x to the right, y downward). The camera
  faces the person, so the person's left side appears on the image's RIGHT:
    - ``POINT_LEFT``  = the person points to THEIR left  -> ``index_dir_x > 0``.
    - ``POINT_RIGHT`` = the person points to THEIR right -> ``index_dir_x < 0``.
  (The capture preview is mirrored like a mirror, so for the person it feels
  natural: pointing to your left looks like pointing to the left on screen.)
* Finger extension: per finger, ``ext_<f>`` = distance(wrist, tip) /
  distance(wrist, PIP) (for the thumb the IP joint is used); > 1 means
  extended, < 1 curled. ``bend_<f>`` is the joint angle in degrees at the PIP
  (IP for the thumb) between the two adjacent segments (180 = straight).
  ``tipdist_<f>`` is the tip distance from the wrist in palm units and
  ``n_fingers_extended`` counts index..pinky with ``ext > 1``.
* Fingertip spread: distances between adjacent fingertips (thumb-index,
  index-middle, middle-ring, ring-pinky) in palm units plus the angle between
  the index and middle finger directions (separates VICTORY from a pointing
  finger or a closed hand).
* Thumb: ``thumb_up_score`` = (thumb MCP y - thumb tip y) / palm size, so it is
  positive when the thumb points up (THUMBS_UP) and negative when it points
  down; ``thumb_out`` = distance(thumb tip, index MCP) / palm size, i.e. how far
  the thumb tip is from the base of the index finger (small when the thumb
  rests against the index, large when it is separated or folded across).

``extract_features`` returns None for invalid or degenerate input (wrong shape,
non-finite values, a handedness value that is not "Left"/"Right" (input
validation only) or a palm collapsed to a point).
"""

from __future__ import annotations

import numpy as np

NUM_LANDMARKS = 21
WRIST = 0
MIDDLE_MCP = 9
_EPS = 1e-6

# Below this palm size (fraction of the frame width) the hand is too far away
# for reliable landmarks (roughly 1.5 m or more from a laptop webcam).
MIN_HAND_SIZE = 0.05

FINGERS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
_FINGER_NAMES = list(FINGERS)

# Features that depend on the absolute image direction (not mirror-canonical).
ABSOLUTE_FEATURES = ["index_dir_x", "index_dir_y", "palm_dir_x", "palm_dir_y"]

FEATURE_NAMES = (
    [f"lm{i}_{axis}" for i in range(1, NUM_LANDMARKS) for axis in ("x", "y")]
    + [f"ext_{f}" for f in _FINGER_NAMES]
    + [f"bend_{f}" for f in _FINGER_NAMES]
    + [f"tipdist_{f}" for f in _FINGER_NAMES]
    + ["n_fingers_extended"]
    + ["spread_thumb_index", "spread_index_middle", "spread_middle_ring", "spread_ring_pinky",
       "angle_index_middle"]
    + ["thumb_up_score", "thumb_out"]
    + ABSOLUTE_FEATURES
)

# CSV layout of the gesture dataset: person,label,handedness,x0,y0,z0,...,x20,y20,z20
LANDMARK_COLUMNS = [f"{c}{i}" for i in range(NUM_LANDMARKS) for c in ("x", "y", "z")]
CSV_COLUMNS = ["person", "label", "handedness"] + LANDMARK_COLUMNS


def _as_landmarks(landmarks) -> np.ndarray | None:
    try:
        arr = np.asarray(landmarks, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.shape != (NUM_LANDMARKS, 3) or not np.all(np.isfinite(arr)):
        return None
    return arr


def hand_size(landmarks) -> float:
    """Palm size: wrist -> middle-finger MCP distance in normalized image coords.

    Bigger value = hand closer to the camera. Returns 0.0 for invalid input.
    """
    arr = _as_landmarks(landmarks)
    if arr is None:
        return 0.0
    return float(np.hypot(*(arr[MIDDLE_MCP, :2] - arr[WRIST, :2])))


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in degrees between two 2D vectors (0 if either is degenerate)."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < _EPS or n2 < _EPS:
        return 0.0
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n >= _EPS else np.zeros(2)


def _normalize_handedness(handedness) -> str | None:
    name = str(handedness).strip().capitalize()
    return name if name in ("Left", "Right") else None


def extract_features(landmarks: np.ndarray, handedness: str) -> np.ndarray | None:
    """Return the feature vector (len == len(FEATURE_NAMES)) or None if invalid."""
    arr = _as_landmarks(landmarks)
    side = _normalize_handedness(handedness)
    if arr is None or side is None:
        return None
    xy = arr[:, :2]
    palm = float(np.linalg.norm(xy[MIDDLE_MCP] - xy[WRIST]))
    if palm < _EPS:
        return None

    # Absolute (image) direction features.
    index_dir = _unit(xy[8] - xy[5])
    palm_dir = _unit(xy[MIDDLE_MCP] - xy[WRIST])

    # Wrist-centered, palm-scaled frame. The handedness label is deliberately
    # ignored (see module docstring): no mirroring, no label feature.
    pts = (xy - xy[WRIST]) / palm

    coords = pts[1:].reshape(-1)  # lm1_x, lm1_y, lm2_x, ...
    wdist = np.linalg.norm(pts, axis=1)  # distance from the wrist in palm units

    ext, bend, tipdist = [], [], []
    for name in _FINGER_NAMES:
        mcp, pip, dip, tip = FINGERS[name]
        ref = dip if name == "thumb" else pip  # thumb: landmark 3 is the IP joint
        ext.append(wdist[tip] / max(wdist[ref], _EPS))
        if name == "thumb":
            bend.append(_angle_deg(pts[2] - pts[3], pts[4] - pts[3]))
        else:
            bend.append(_angle_deg(pts[mcp] - pts[pip], pts[dip] - pts[pip]))
        tipdist.append(wdist[tip])
    n_extended = float(sum(e > 1.0 for e in ext[1:]))

    tips = {name: pts[FINGERS[name][3]] for name in _FINGER_NAMES}
    spreads = [
        np.linalg.norm(tips["thumb"] - tips["index"]),
        np.linalg.norm(tips["index"] - tips["middle"]),
        np.linalg.norm(tips["middle"] - tips["ring"]),
        np.linalg.norm(tips["ring"] - tips["pinky"]),
    ]
    angle_im = _angle_deg(pts[8] - pts[5], pts[12] - pts[9])

    thumb_up_score = float(pts[2, 1] - pts[4, 1])  # image y grows downward: up => tip y smaller
    thumb_out = float(np.linalg.norm(pts[4] - pts[5]))

    feats = np.concatenate(
        [
            coords,
            ext,
            bend,
            tipdist,
            [n_extended],
            spreads,
            [angle_im],
            [thumb_up_score, thumb_out],
            index_dir,
            palm_dir,
        ]
    )
    return feats.astype(np.float64)
