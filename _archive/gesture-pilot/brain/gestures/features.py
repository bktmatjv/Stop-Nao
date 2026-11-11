"""Feature extraction from MediaPipe Pose landmarks.

Conventions
-----------
* Input is an array of shape (33, 4): ``x, y, z, visibility`` per landmark, as
  produced by ``mp.solutions.pose`` (x/y normalized to [0, 1], y grows downward).
* Landmarks are used exactly as MediaPipe reports them, with NO horizontal flip.
  MediaPipe's LEFT landmarks (11, 13, 15, 23) belong to the PERSON's left side.
  The webcam image shown on screen is usually mirrored for the user, but the
  features are computed on the raw frame. Therefore:
    - ``TURN_LEFT``  = the person extends THEIR left arm  (MediaPipe left landmarks).
    - ``TURN_RIGHT`` = the person extends THEIR right arm (MediaPipe right landmarks).
* Features are translation invariant (hip-centered) and scale invariant
  (divided by shoulder width). The z coordinate is ignored on purpose: it is
  noisy for monocular webcams and not needed for the gestures of this project.
* Missing values: both shoulders are required (they define the scale), so a
  frame without them yields None. An elbow, wrist or the nose with visibility
  < 0.5 (typically out of frame) does NOT discard the frame: its coordinates
  are imputed as 0 (the body center), every feature derived from it is set to
  0, and a binary ``<joint>_visible`` flag tells the classifier it is missing.
"""

from __future__ import annotations

import numpy as np

# MediaPipe Pose landmark indices used by this project.
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24

NUM_LANDMARKS = 33
MIN_VISIBILITY = 0.5
_EPS = 1e-6

# Landmarks whose normalized coordinates are fed to the classifier.
_COORD_POINTS = [
    ("nose", NOSE),
    ("l_shoulder", L_SHOULDER),
    ("r_shoulder", R_SHOULDER),
    ("l_elbow", L_ELBOW),
    ("r_elbow", R_ELBOW),
    ("l_wrist", L_WRIST),
    ("r_wrist", R_WRIST),
]

FEATURE_NAMES = (
    [f"{name}_{axis}" for name, _ in _COORD_POINTS for axis in ("x", "y")]
    + [
        "l_elbow_angle",
        "r_elbow_angle",
        "l_shoulder_angle",
        "r_shoulder_angle",
        "l_wrist_above_shoulder",
        "r_wrist_above_shoulder",
        "l_wrist_above_nose",
        "r_wrist_above_nose",
        "wrist_distance",
    ]
    + ["nose_visible", "l_elbow_visible", "r_elbow_visible", "l_wrist_visible", "r_wrist_visible"]
)

# Joints that may be missing (imputed + flagged) and joints checked for framing.
_OPTIONAL_POINTS = [("nose", NOSE), ("l_elbow", L_ELBOW), ("r_elbow", R_ELBOW),
                    ("l_wrist", L_WRIST), ("r_wrist", R_WRIST)]
KEY_POINTS = [("l_shoulder", L_SHOULDER), ("r_shoulder", R_SHOULDER), ("l_elbow", L_ELBOW),
              ("r_elbow", R_ELBOW), ("l_wrist", L_WRIST), ("r_wrist", R_WRIST)]

# CSV layout of the gesture dataset: person,label,x0,y0,z0,v0,...,x32,y32,z32,v32
LANDMARK_COLUMNS = [f"{c}{i}" for i in range(NUM_LANDMARKS) for c in ("x", "y", "z", "v")]
CSV_COLUMNS = ["person", "label"] + LANDMARK_COLUMNS


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Angle in degrees between two 2D vectors (0 if either is degenerate)."""
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < _EPS or n2 < _EPS:
        return 0.0
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _validate(landmarks) -> np.ndarray:
    arr = np.asarray(landmarks, dtype=float)
    if arr.shape != (NUM_LANDMARKS, 4):
        raise ValueError(f"landmarks must have shape (33, 4), got {arr.shape}")
    return arr


def missing_key_landmarks(landmarks: np.ndarray) -> list[str]:
    """Names of shoulders/elbows/wrists that are not reliably visible (in KEY_POINTS order)."""
    lm = _validate(landmarks)
    return [name for name, idx in KEY_POINTS
            if lm[idx, 3] < MIN_VISIBILITY or not np.all(np.isfinite(lm[idx, :2]))]


def extract_features(landmarks: np.ndarray) -> np.ndarray | None:
    """Return the feature vector (len == len(FEATURE_NAMES)) or None.

    None is returned only when a shoulder has visibility < 0.5 or the shoulders
    collapse to a single point (the scale is undefined). Missing elbows, wrists
    or nose are imputed and flagged (see module docstring).
    """
    lm = _validate(landmarks)
    shoulders = [L_SHOULDER, R_SHOULDER]
    if np.any(lm[shoulders, 3] < MIN_VISIBILITY) or not np.all(np.isfinite(lm[shoulders, :2])):
        return None
    visible = {idx: bool(lm[idx, 3] >= MIN_VISIBILITY and np.all(np.isfinite(lm[idx, :2])))
               for _, idx in _OPTIONAL_POINTS}
    visible[L_SHOULDER] = visible[R_SHOULDER] = True

    xy = np.nan_to_num(lm[:, :2].copy())
    shoulder_mid = (xy[L_SHOULDER] + xy[R_SHOULDER]) / 2.0
    scale = float(np.linalg.norm(xy[L_SHOULDER] - xy[R_SHOULDER]))
    if scale < _EPS:
        return None

    hips_visible = lm[L_HIP, 3] >= MIN_VISIBILITY and lm[R_HIP, 3] >= MIN_VISIBILITY
    if hips_visible:
        center = (xy[L_HIP] + xy[R_HIP]) / 2.0
        torso = center - shoulder_mid  # shoulders -> hips, points "down" in the image
        if np.linalg.norm(torso) < _EPS:
            torso = np.array([0.0, 1.0])
    else:
        center = shoulder_mid
        torso = np.array([0.0, 1.0])

    norm = (xy - center) / scale
    for idx, ok in visible.items():
        if not ok:
            norm[idx] = 0.0  # imputed: body center

    coords = np.concatenate([norm[idx] for _, idx in _COORD_POINTS])

    def vis(*idxs: int) -> bool:
        return all(visible[i] for i in idxs)

    l_elbow_angle = (_angle_deg(xy[L_SHOULDER] - xy[L_ELBOW], xy[L_WRIST] - xy[L_ELBOW])
                     if vis(L_ELBOW, L_WRIST) else 0.0)
    r_elbow_angle = (_angle_deg(xy[R_SHOULDER] - xy[R_ELBOW], xy[R_WRIST] - xy[R_ELBOW])
                     if vis(R_ELBOW, R_WRIST) else 0.0)
    # Arm vs torso: 0 = arm hanging along the body, 90 = horizontal, 180 = straight up.
    l_shoulder_angle = _angle_deg(xy[L_ELBOW] - xy[L_SHOULDER], torso) if vis(L_ELBOW) else 0.0
    r_shoulder_angle = _angle_deg(xy[R_ELBOW] - xy[R_SHOULDER], torso) if vis(R_ELBOW) else 0.0

    # Image y grows downward, so "above" means a smaller y.
    l_above_sh = float(vis(L_WRIST) and xy[L_WRIST, 1] < xy[L_SHOULDER, 1])
    r_above_sh = float(vis(R_WRIST) and xy[R_WRIST, 1] < xy[R_SHOULDER, 1])
    l_above_nose = float(vis(L_WRIST, NOSE) and xy[L_WRIST, 1] < xy[NOSE, 1])
    r_above_nose = float(vis(R_WRIST, NOSE) and xy[R_WRIST, 1] < xy[NOSE, 1])

    wrist_distance = (float(np.linalg.norm(xy[L_WRIST] - xy[R_WRIST]) / scale)
                      if vis(L_WRIST, R_WRIST) else 0.0)
    flags = [float(visible[idx]) for _, idx in _OPTIONAL_POINTS]

    feats = np.concatenate(
        [
            coords,
            [
                l_elbow_angle,
                r_elbow_angle,
                l_shoulder_angle,
                r_shoulder_angle,
                l_above_sh,
                r_above_sh,
                l_above_nose,
                r_above_nose,
                wrist_distance,
            ],
            flags,
        ]
    )
    return feats.astype(np.float64)


def person_distance(landmarks: np.ndarray) -> float:
    """Shoulder width in normalized image coordinates (frame width = 1.0).

    Bigger value = person closer to the camera. Returns 0.0 if the shoulders
    are not visible.
    """
    lm = _validate(landmarks)
    if lm[L_SHOULDER, 3] < MIN_VISIBILITY or lm[R_SHOULDER, 3] < MIN_VISIBILITY:
        return 0.0
    return float(abs(lm[L_SHOULDER, 0] - lm[R_SHOULDER, 0]))
