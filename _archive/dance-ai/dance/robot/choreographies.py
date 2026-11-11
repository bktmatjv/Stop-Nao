"""Arm + head choreographies for NAO (SOFT / MODERATE / ENERGETIC).

Only arms and head are used: no legs, no hip, no walking, so the robot keeps
its balance in StandInit. A choreography is a cyclic list of steps
``(duration_beats, {joint: radians})``; ``build_keyframes`` scales it to the
musical tempo, repeats it to fill a time span, enforces the safety limits and
emits the per-joint ``(names, angles, times)`` triple that
``ALMotion.angleInterpolation`` expects.

Safety rules enforced here (the bridge re-checks limits and a hard velocity cap):
  * joint angular velocity <= MAX_JOINT_VELOCITY (steps that would exceed it are
    stretched in half-beat units, so they stay on the musical grid),
  * every step lasts at least MIN_STEP_SECONDS,
  * the first transition (from the current pose to the first step) lasts at
    least FIRST_TRANSITION_SECONDS,
  * ``validate_keyframes`` rejects any angle outside the limits reported by
    the robot (ALMotion.getLimits) before anything is sent.
"""
import math

from dance import STYLES
from dance.config import BPM_MAX, BPM_MIN

MAX_JOINT_VELOCITY = 2.0       # rad/s (NAO hardware allows more; we stay gentle)
MIN_STEP_SECONDS = 0.25
FIRST_TRANSITION_SECONDS = 0.8
STRETCH_BEATS = 0.5            # granularity used to lengthen a too-fast step

JOINTS = (
    "HeadYaw", "HeadPitch",
    "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll",
    "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll",
)

# Arms down at the sides, head forward (values from the professor's
# naoooo/nao_show.py "Bajar el brazo" step, mirrored for the left side).
NEUTRAL_POSE = {
    "HeadYaw": 0.0, "HeadPitch": 0.0,
    "LShoulderPitch": 1.40, "LShoulderRoll": 0.15,
    "LElbowYaw": -1.20, "LElbowRoll": -0.45,
    "RShoulderPitch": 1.40, "RShoulderRoll": -0.15,
    "RElbowYaw": 1.20, "RElbowRoll": 0.45,
}


def _arm(side, pitch, out, yaw, bend):
    """Arm joints from side-independent values.

    pitch: shoulder pitch (0 = forward horizontal, 1.4 = down at the side).
    out:   how far the arm is lifted away from the body (>= 0).
    yaw:   elbow yaw magnitude; bend: elbow bend magnitude (>= 0).
    The left side is the mirror image of the right side.
    """
    sign = 1.0 if side == "R" else -1.0
    return {
        side + "ShoulderPitch": pitch,
        side + "ShoulderRoll": -sign * out,
        side + "ElbowYaw": sign * yaw,
        side + "ElbowRoll": sign * bend,
    }


def _pose(right, left, head):
    pose = {"HeadYaw": head[0], "HeadPitch": head[1]}
    pose.update(_arm("R", *right))
    pose.update(_arm("L", *left))
    return pose


DOWN = (1.40, 0.15, 1.20, 0.45)   # (pitch, out, yaw, bend): arm at the side
FRONT = (0.0, 0.0)                # (head yaw, head pitch)

# --- SOFT: wide, slow waves, 2 beats per step, small head sway --------------
_SOFT = [
    (2.0, _pose((0.80, 0.90, 1.20, 0.50), DOWN, (-0.20, 0.0))),
    (2.0, _pose((0.90, 0.70, 1.20, 0.80), (0.90, 0.70, 1.20, 0.80), (0.0, -0.10))),
    (2.0, _pose(DOWN, (0.80, 0.90, 1.20, 0.50), (0.20, 0.0))),
    (2.0, _pose(DOWN, DOWN, FRONT)),
]

# --- MODERATE: rhythmic alternating arm raises, 1 beat per step -------------
_R_UP = (0.70, 0.50, 1.20, 1.00)
_MODERATE = [
    (1.0, _pose(_R_UP, DOWN, (-0.15, 0.0))),
    (1.0, _pose(DOWN, _R_UP, (0.15, 0.0))),
    (1.0, _pose(_R_UP, DOWN, (-0.15, 0.0))),
    (1.0, _pose(DOWN, _R_UP, (0.15, 0.0))),
    (1.0, _pose((0.90, 0.60, 1.20, 0.90), (0.90, 0.60, 1.20, 0.90), (0.0, -0.10))),
    (1.0, _pose(DOWN, DOWN, FRONT)),
]

# --- ENERGETIC: fast pumping, alternating arms, 0.5-1 beat per step ---------
_READY = (0.40, 0.60, 1.20, 1.00)
_PUMP_UP = (0.15, 0.70, 1.20, 1.20)
_PUMP_LOW = (0.55, 0.50, 1.20, 0.90)
_HIGH = (0.0, 0.80, 1.20, 1.10)
_MID = (0.40, 0.60, 1.20, 0.95)
_ENERGETIC = [
    (0.5, _pose(_PUMP_UP, _PUMP_LOW, (-0.20, -0.15))),
    (0.5, _pose(_PUMP_LOW, _PUMP_UP, (0.20, -0.15))),
    (0.5, _pose(_PUMP_UP, _PUMP_LOW, (-0.20, -0.15))),
    (0.5, _pose(_PUMP_LOW, _PUMP_UP, (0.20, -0.15))),
    (1.0, _pose(_HIGH, _HIGH, (0.0, -0.20))),
    (0.5, _pose(_MID, _MID, (0.0, 0.0))),
    (0.5, _pose(_HIGH, _HIGH, (0.0, -0.20))),
    (1.0, _pose(_READY, _READY, (0.0, -0.15))),
]

CHOREOGRAPHIES = {"SOFT": _SOFT, "MODERATE": _MODERATE, "ENERGETIC": _ENERGETIC}
assert sorted(CHOREOGRAPHIES) == sorted(STYLES)


def fold_bpm(bpm):
    """Fold a tempo into [BPM_MIN, BPM_MAX] by doubling/halving (240 -> 120)."""
    bpm = float(bpm)
    if not math.isfinite(bpm) or bpm <= 0:
        raise ValueError("bpm must be a positive finite number, got %r" % (bpm,))
    while bpm < BPM_MIN:
        bpm *= 2.0
    while bpm > BPM_MAX:
        bpm /= 2.0
    return bpm


def _full_pose(pose):
    full = dict(NEUTRAL_POSE)
    if pose:
        unknown = set(pose) - set(NEUTRAL_POSE)
        if unknown:
            raise ValueError("unknown joint(s) in pose: %s" % ", ".join(sorted(unknown)))
        full.update(pose)
    return full


def _min_travel_time(prev, nxt):
    return max(abs(nxt[j] - prev[j]) for j in JOINTS) / MAX_JOINT_VELOCITY


def build_keyframes(style, bpm, duration_s, start_pose=None):
    """Return ``(names, angles, times)`` for ``angleInterpolation``.

    ``angles[i]`` / ``times[i]`` belong to ``names[i]``; times are cumulative
    seconds, strictly increasing. The motion covers at least ``duration_s``
    seconds (it ends when the step that crosses it ends) and starts from
    ``start_pose`` (partial dict allowed) or the neutral pose.
    """
    if style not in CHOREOGRAPHIES:
        raise ValueError("unknown style %r; expected one of %s" % (style, ", ".join(STYLES)))
    duration_s = float(duration_s)
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be positive, got %r" % (duration_s,))
    beat = 60.0 / fold_bpm(bpm)
    cycle = CHOREOGRAPHIES[style]

    current = _full_pose(start_pose)
    poses, times = [], []
    t, index = 0.0, 0
    while not times or t < duration_s:
        beats, target = cycle[index % len(cycle)]
        target = _full_pose(target)
        need = _min_travel_time(current, target)
        if not times:
            # Lead-in: smooth move from wherever we are to the first step.
            step = max(FIRST_TRANSITION_SECONDS, need)
        else:
            step = beats * beat
            required = max(MIN_STEP_SECONDS, need)
            if step < required:
                unit = STRETCH_BEATS * beat
                step += math.ceil((required - step) / unit - 1e-9) * unit
        t += step
        times.append(t)
        poses.append(target)
        current = target
        index += 1

    names = list(JOINTS)
    angles = [[p[j] for p in poses] for j in names]
    return names, angles, [list(times) for _ in names]


def final_pose(names, angles):
    """Pose (joint -> radians) reached at the end of a keyframe set."""
    return {n: a[-1] for n, a in zip(names, angles)}


def validate_keyframes(names, angles, limits):
    """Raise ValueError listing every angle outside ``limits`` ({joint: (lo, hi)})."""
    problems = []
    for name, row in zip(names, angles):
        if name not in limits:
            problems.append("%s: no limits known" % name)
            continue
        lo, hi = limits[name]
        for i, value in enumerate(row):
            if not lo <= value <= hi:
                problems.append("%s=%s (keyframe %d) outside [%s, %s]" % (name, value, i, lo, hi))
    if problems:
        raise ValueError("keyframes violate joint limits: " + "; ".join(problems))
