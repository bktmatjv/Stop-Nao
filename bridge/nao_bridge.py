# -*- coding: utf-8 -*-
r"""Persistent NAO bridge (Python 2.7 compatible).

Reads one JSON object per line from stdin and writes exactly one JSON line per
response to stdout (flushed). Everything else (logs, naoqi noise) goes to stderr.
Every response has "ok"; every --mock response also has "simulated": true
(real mode sends "simulated": false).

Protocol:
    {"cmd":"ping"}
        -> {"ok":true,"mock":bool}
    {"cmd":"sense"}
        -> {"ok":true,"sonar_left":f,"sonar_right":f,"moving":bool,"posture":str}
           posture comes from ALRobotPosture.getPosture() ("Stand", "StandInit",
           "Sit", "Crouch", "Unknown", ...)
    {"cmd":"move","x":f,"theta":f}
        -> {"ok":true,"x":f,"theta":f,"clamped":bool}   NON-blocking moveTo(x, 0, theta)
           x is clamped to [0, 0.30] m and theta to [-1.6, 1.6] rad. Refused
           (ok:false) unless the posture is Stand, StandInit or StandZero.
    {"cmd":"stop"}
        -> {"ok":true}                                  stopMove()
    {"cmd":"wave"}
        -> {"ok":true,"duration":f}   right-arm wave, arms only, blocking (<= 4 s).
           Refused while moving or when not standing. The arm is restored.
    {"cmd":"posture","name":"Sit"|"StandInit"}
        -> {"ok":true}                goToPosture(name, 0.6); refused while moving
    {"cmd":"say","text":"..."}                 -> {"ok":true}   (utf-8, Spanish)
    {"cmd":"eyes","color":"green|yellow|red|blue|white"}  -> {"ok":true}
    {"cmd":"set_mock","sonar_left"?,"sonar_right"?,"posture"?,"moving"?}
        -> {"ok":true}                (mock mode only)
    {"cmd":"quit"}                             -> {"ok":true}, then the process exits

Usage:
    C:\Python27\python.exe bridge\nao_bridge.py --robot nyx|zyra|sim
    C:\Python27\python.exe bridge\nao_bridge.py --ip 1.2.3.4 --port 9559
    C:\Python27\python.exe bridge\nao_bridge.py --mock      (no naoqi, no robot)

Safety: this bridge never calls rest(), never disables the fall manager or any
other protection, and never releases stiffness. Walking is only accepted from a
standing posture, with clamped distance and angle. The wave only touches arm
joints (whitelist), every angle is checked against the joint limits (from the
robot in real mode) before anything moves.

The process-bridge idea (Python 3 relaunching C:\Python27\python.exe with the
pynaoqi folder on PATH) is adapted from the professor's naoooo/py27-py3.py and
naoooo/nao_show.py (joint names, wave angles, limit validation before moving,
StandInit start-up). The joint whitelist and the "simulated" flag are adapted
from the archived dance bridge.
"""
from __future__ import print_function

import argparse
import json
import math
import numbers
import os
import re
import sys
import time

SDK_DIR = r"C:\Python27\Lib\site-packages"

ROBOTS = {
    "nyx": ("127.0.0.1", 59242),
    "zyra": ("192.168.108.90", 9559),
    "sim": ("10.190.103.36", 51940),
}

SONAR_LEFT_KEY = "Device/SubDeviceList/US/Left/Sensor/Value"
SONAR_RIGHT_KEY = "Device/SubDeviceList/US/Right/Sensor/Value"

DEFAULT_SONAR = 1.5
DEFAULT_POSTURE = "StandInit"

# Movement contract.
MAX_X = 0.40        # m, forward only (NAO has no rear sensor)
MAX_THETA = 1.6     # rad
STANDING_POSTURES = ("Stand", "StandInit", "StandZero")
ALLOWED_POSTURES = ("Sit", "StandInit")
POSTURE_SPEED = 0.6
# Mock walking speed: duration = x / MOCK_LINEAR + |theta| / MOCK_ANGULAR.
MOCK_LINEAR = 0.1   # m/s
MOCK_ANGULAR = 0.5  # rad/s

# Fixed eye colour table (0xRRGGBB).
EYE_COLORS = {
    "green": 0x00FF00,
    "yellow": 0xFFFF00,
    "red": 0xFF0000,
    "blue": 0x0000FF,
    "white": 0xFFFFFF,
    "magenta": 0xFF00FF,
    "cyan": 0x00FFFF,
    "purple": 0x800080,
    "orange": 0xFFA500,
    "pink": 0xFF69B4,
}

# Joints allowed for wave and dance (arms and head, balance-safe).
ALLOWED_JOINT = re.compile(
    r"^(Head(Yaw|Pitch)|[LR](ShoulderPitch|ShoulderRoll|ElbowYaw|ElbowRoll|WristYaw|Hand))$"
)

# APPROXIMATE NAO H25 (v4/v5) limits in radians, used ONLY in --mock mode.
# Real limits come from the robot (ALMotion.getLimits). Hands are ratios [0, 1].
APPROX_LIMITS = {
    "HeadYaw": (-2.0857, 2.0857),
    "HeadPitch": (-0.6720, 0.5149),
    "LShoulderPitch": (-2.0857, 2.0857),
    "LShoulderRoll": (-0.3142, 1.3265),
    "LElbowYaw": (-2.0857, 2.0857),
    "LElbowRoll": (-1.5446, -0.0349),
    "LWristYaw": (-1.8238, 1.8238),
    "LHand": (0.0, 1.0),
    "RShoulderPitch": (-2.0857, 2.0857),
    "RShoulderRoll": (-1.3265, 0.3142),
    "RElbowYaw": (-2.0857, 2.0857),
    "RElbowRoll": (0.0349, 1.5446),
    "RWristYaw": (-1.8238, 1.8238),
    "RHand": (0.0, 1.0),
}

# Mock "current" arm and head angles (arms down, head straight), from nao_show.py.
MOCK_ARM_REST = {
    "HeadYaw": 0.0, "HeadPitch": 0.0,
    "LShoulderPitch": 1.40, "LShoulderRoll": 0.15, "LElbowYaw": -1.20,
    "LElbowRoll": -0.45, "LWristYaw": 0.0, "LHand": 0.30,
    "RShoulderPitch": 1.40, "RShoulderRoll": -0.15, "RElbowYaw": 1.20,
    "RElbowRoll": 0.45, "RWristYaw": 0.0, "RHand": 0.30,
}

# Right-arm wave (angles from the professor's nao_show.py "Saludar" steps).
# Joint -> (raised pose, [(time, elbow-roll)] waving keyframes); all times are
# absolute seconds, the arm returns to its starting angles at WAVE_END.
WAVE_RAISE_T = 1.0
WAVE_HOLD_T = 3.0
WAVE_END = 3.8
WAVE_RAISED = {
    "RShoulderPitch": 0.25, "RShoulderRoll": -0.30, "RElbowYaw": 1.20,
    "RElbowRoll": 1.15, "RHand": 1.0,
}
WAVE_ELBOW = [(1.5, 0.65), (2.0, 1.30), (2.5, 0.65)]
WAVE_MAX_VELOCITY = 3.0  # rad/s sanity cap, defence in depth

# Dance choreography: arm + head keyframes (rhythmic, safe, no leg movement).
DANCE_JOINTS = (
    "HeadYaw", "HeadPitch",
    "LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll",
    "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll",
)
DANCE_STEPS = [
    (1.0, {
        "HeadYaw": -0.20, "HeadPitch": -0.10,
        "RShoulderPitch": 0.30, "RShoulderRoll": -0.50, "RElbowYaw": 1.20, "RElbowRoll": 1.05,
        "LShoulderPitch": 1.10, "LShoulderRoll": 0.30, "LElbowYaw": -1.20, "LElbowRoll": -0.70,
    }),
    (2.0, {
        "HeadYaw": 0.20, "HeadPitch": -0.10,
        "RShoulderPitch": 1.10, "RShoulderRoll": -0.30, "RElbowYaw": 1.20, "RElbowRoll": 0.70,
        "LShoulderPitch": 0.30, "LShoulderRoll": 0.50, "LElbowYaw": -1.20, "LElbowRoll": -1.05,
    }),
    (3.0, {
        "HeadYaw": 0.0, "HeadPitch": -0.20,
        "RShoulderPitch": 0.25, "RShoulderRoll": -0.60, "RElbowYaw": 1.20, "RElbowRoll": 1.15,
        "LShoulderPitch": 0.25, "LShoulderRoll": 0.60, "LElbowYaw": -1.20, "LElbowRoll": -1.15,
    }),
    (3.8, {
        "HeadYaw": 0.0, "HeadPitch": 0.0,
        "RShoulderPitch": 0.45, "RShoulderRoll": -0.40, "RElbowYaw": 1.20, "RElbowRoll": 0.60,
        "LShoulderPitch": 0.45, "LShoulderRoll": 0.40, "LElbowYaw": -1.20, "LElbowRoll": -0.60,
    }),
    (4.6, {
        "HeadYaw": 0.0, "HeadPitch": -0.15,
        "RShoulderPitch": 0.25, "RShoulderRoll": -0.60, "RElbowYaw": 1.20, "RElbowRoll": 1.15,
        "LShoulderPitch": 0.25, "LShoulderRoll": 0.60, "LElbowYaw": -1.20, "LElbowRoll": -1.15,
    }),
    (5.6, None),  # returns to starting pose
]

# Greeting choreography: right-arm salute tip (~2 s), standing or sitting.
GREET_JOINTS = (
    "HeadYaw", "HeadPitch",
    "RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RHand",
)
GREET_STEPS = [
    (0.6, {
        "HeadYaw": 0.0, "HeadPitch": -0.15,
        "RShoulderPitch": -0.20, "RShoulderRoll": -0.20,
        "RElbowYaw": 1.20, "RElbowRoll": 0.40, "RHand": 1.0,
    }),
    (1.0, {
        "HeadYaw": 0.0, "HeadPitch": -0.20,
        "RShoulderPitch": -0.20, "RShoulderRoll": -0.20,
        "RElbowYaw": 1.20, "RElbowRoll": 0.80, "RHand": 1.0,
    }),
    (1.5, {
        "HeadYaw": 0.0, "HeadPitch": -0.10,
        "RShoulderPitch": -0.20, "RShoulderRoll": -0.20,
        "RElbowYaw": 1.20, "RElbowRoll": 0.40, "RHand": 1.0,
    }),
    (2.2, None),  # returns to starting pose
]

_STRING_TYPES = (type(u""), type(""))


def log(msg):
    try:
        sys.stderr.write("[bridge] %s\n" % msg)
        sys.stderr.flush()
    except Exception:
        pass


class RequestError(Exception):
    """The request is invalid or unsafe; reported as {"ok": false}."""


def _is_number(value):
    return (isinstance(value, numbers.Real) and not isinstance(value, bool)
            and not math.isnan(value) and not math.isinf(value))


def build_wave(start):
    """Return (names, angles, times) for the wave, ending at the `start` angles."""
    names = sorted(WAVE_RAISED)
    angles, times = [], []
    for n in names:
        a = [WAVE_RAISED[n]]
        t = [WAVE_RAISE_T]
        if n == "RElbowRoll":
            for tt, v in WAVE_ELBOW:
                t.append(tt)
                a.append(v)
        t.append(WAVE_HOLD_T)
        a.append(WAVE_RAISED[n])
        t.append(WAVE_END)
        a.append(start[n])
        angles.append(a)
        times.append(t)
    return names, angles, times


def build_dance(start):
    """Return (names, angles, times) for the dance, ending at the `start` angles."""
    names = sorted(DANCE_JOINTS)
    angles, times = [], []
    for n in names:
        a_row, t_row = [], []
        for t, pose_dict in DANCE_STEPS:
            t_row.append(t)
            if pose_dict is None:
                a_row.append(start.get(n, MOCK_ARM_REST.get(n, 0.0)))
            else:
                a_row.append(pose_dict.get(n, start.get(n, 0.0)))
        angles.append(a_row)
        times.append(t_row)
    return names, angles, times


def build_greet(start):
    """Return (names, angles, times) for the salute greeting, ending at the `start` angles."""
    names = sorted(GREET_JOINTS)
    angles, times = [], []
    for n in names:
        a_row, t_row = [], []
        for t, pose_dict in GREET_STEPS:
            t_row.append(t)
            if pose_dict is None:
                a_row.append(start.get(n, MOCK_ARM_REST.get(n, 0.0)))
            else:
                a_row.append(pose_dict.get(n, start.get(n, 0.0)))
        angles.append(a_row)
        times.append(t_row)
    return names, angles, times


def validate_wave(robot, names, angles, times):
    """Whitelist, limits and velocity checks; raises RequestError."""
    forbidden = [n for n in names if not ALLOWED_JOINT.match(n)]
    if forbidden:
        raise RequestError("joint(s) not allowed: %s"
                           % ", ".join(sorted(forbidden)))
    limits = robot.limits(names)
    bad = []
    for n, a_row in zip(names, angles):
        lo, hi = limits[n]
        for v in a_row:
            if not lo <= v <= hi:
                bad.append("%s=%s outside [%s, %s]" % (n, v, lo, hi))
    if bad:
        raise RequestError("angle(s) out of limits: %s" % "; ".join(bad))
    start = robot.current_angles(names)
    for n, a_row, t_row in zip(names, angles, times):
        prev_a, prev_t = start.get(n), 0.0
        for a, t in zip(a_row, t_row):
            if prev_a is not None and abs(a - prev_a) / (t - prev_t) > WAVE_MAX_VELOCITY:
                raise RequestError("wave too fast for %s" % n)
            prev_a, prev_t = a, t


class MockRobot(object):
    """Stand-in robot: no naoqi import, logs every action to stderr.

    Keeps a small mutable state (sonars, posture, moving) updated by
    move/posture/stop/set_mock. `moving` is true for a simulated duration after
    a move and then turns false by itself.
    """

    is_mock = True

    def __init__(self):
        self.sonar_left = DEFAULT_SONAR
        self.sonar_right = DEFAULT_SONAR
        self.posture_name = DEFAULT_POSTURE
        self._move_until = 0.0
        self.eyes = EYE_COLORS["blue"]
        self.arm = dict(MOCK_ARM_REST)

    @property
    def moving(self):
        return time.time() < self._move_until

    def get_posture(self):
        return self.posture_name

    def is_moving(self):
        return self.moving

    def move_to(self, x, theta):
        duration = x / MOCK_LINEAR + abs(theta) / MOCK_ANGULAR
        self._move_until = time.time() + duration
        log("MOCK moveTo(%.3f, 0, %.3f) simulated for %.2fs" % (x, theta, duration))

    def stop(self):
        self._move_until = 0.0
        log("MOCK stopMove()")

    def go_posture(self, name, speed):
        self._move_until = 0.0
        self.posture_name = name
        log("MOCK goToPosture(%s, %.2f)" % (name, speed))
        return True

    def say(self, text):
        if isinstance(text, type(u"")):
            text = text.encode("utf-8")
        log("MOCK say: %s" % text)

    def set_eyes(self, rgb):
        self.eyes = rgb
        log("MOCK eyes 0x%06X" % rgb)

    def sense(self):
        return self.sonar_left, self.sonar_right, self.moving, self.posture_name

    def limits(self, names):
        missing = [n for n in names if n not in APPROX_LIMITS]
        if missing:
            raise RequestError("unknown joint(s): %s" % ", ".join(sorted(missing)))
        return dict((n, APPROX_LIMITS[n]) for n in names)

    def current_angles(self, names):
        return dict((n, self.arm[n]) for n in names if n in self.arm)

    def play(self, names, angles, times):
        duration = max(t[-1] for t in times)
        for n, a in zip(names, angles):
            self.arm[n] = a[-1]
        log("MOCK wave %.2fs (simulated, no robot moved)" % duration)
        return duration

    def set_mock(self, sonar_left=None, sonar_right=None, posture=None, moving=None):
        if sonar_left is not None:
            self.sonar_left = float(sonar_left)
        if sonar_right is not None:
            self.sonar_right = float(sonar_right)
        if posture is not None:
            self.posture_name = posture
        if moving is not None:
            self._move_until = float("inf") if moving else 0.0

    def close(self):
        pass


class NaoRobot(object):
    """Real robot through naoqi.ALProxy (imported lazily, never in mock mode)."""

    is_mock = False

    def __init__(self, ip, port):
        os.environ["PATH"] = SDK_DIR + os.pathsep + os.environ.get("PATH", "")
        if SDK_DIR not in sys.path:
            sys.path.append(SDK_DIR)
        from naoqi import ALProxy

        log("connecting to %s:%d" % (ip, port))
        self.motion = ALProxy("ALMotion", ip, port)
        self.memory = ALProxy("ALMemory", ip, port)
        self.posture = ALProxy("ALRobotPosture", ip, port)
        self.tts = None
        self.leds = None
        self.sonar = None
        try:
            self.tts = ALProxy("ALTextToSpeech", ip, port)
            try:
                self.tts.setLanguage("Spanish")
            except Exception as exc:
                log("setLanguage failed, using default: %s" % exc)
        except Exception as exc:
            log("no ALTextToSpeech: %s" % exc)
        try:
            self.leds = ALProxy("ALLeds", ip, port)
        except Exception as exc:
            log("no ALLeds: %s" % exc)
        try:
            self.sonar = ALProxy("ALSonar", ip, port)
            self.sonar.subscribe("nao_bridge")
        except Exception as exc:
            log("no ALSonar subscription: %s" % exc)
            self.sonar = None
        self._limits_cache = {}

        # Same start-up as the professor's nao_show.py. Nothing is disabled.
        self.motion.stiffnessInterpolation("Body", 1.0, 1.0)
        if not self.posture.goToPosture("StandInit", 0.5):
            log("goToPosture(StandInit) returned False")
        self.motion.moveInit()
        self.set_eyes(EYE_COLORS["blue"])

    def get_posture(self):
        return str(self.posture.getPosture())

    def is_moving(self):
        return bool(self.motion.moveIsActive())

    def move_to(self, x, theta):
        self.motion.post.moveTo(x, 0.0, theta)  # non-blocking

    def stop(self):
        self.motion.stopMove()

    def go_posture(self, name, speed):
        return bool(self.posture.goToPosture(str(name), speed))

    def say(self, text):
        if self.tts is None:
            log("say skipped (no TTS)")
            return
        if isinstance(text, type(u"")):
            text = text.encode("utf-8")
        self.tts.say(text)

    def set_eyes(self, rgb):
        if self.leds is None:
            return
        self.leds.fadeRGB("FaceLeds", rgb, 0.2)

    def sense(self):
        left = float(self.memory.getData(SONAR_LEFT_KEY))
        right = float(self.memory.getData(SONAR_RIGHT_KEY))
        return left, right, self.is_moving(), self.get_posture()

    def limits(self, names):
        missing = [n for n in names if n not in self._limits_cache]
        if missing:
            raw = self.motion.getLimits(missing)
            if len(raw) != len(missing):
                raise RequestError("robot did not return limits for all joints")
            for n, lim in zip(missing, raw):
                self._limits_cache[n] = (float(lim[0]), float(lim[1]))
        return dict((n, self._limits_cache[n]) for n in names)

    def current_angles(self, names):
        values = self.motion.getAngles(list(names), True)
        return dict(zip(names, [float(v) for v in values]))

    def play(self, names, angles, times):
        names = [str(n) for n in names]
        self.motion.angleInterpolation(names, angles, times, True)  # blocking
        return max(t[-1] for t in times)

    def set_mock(self, **kwargs):
        raise RequestError("set_mock is only available in mock mode")

    def close(self):
        # Never release stiffness or rest the robot automatically.
        if self.sonar is not None:
            try:
                self.sonar.unsubscribe("nao_bridge")
            except Exception:
                pass


def _require_number(req, key):
    value = req.get(key)
    if not _is_number(value):
        raise RequestError("%s must be a finite number" % key)
    return float(value)


def _require_standing(robot, what):
    posture = robot.get_posture()
    if posture not in STANDING_POSTURES:
        raise RequestError("cannot %s: posture is %s, not standing (%s)"
                           % (what, posture, "/".join(STANDING_POSTURES)))


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def cmd_move(robot, req):
    x = _require_number(req, "x")
    theta = _require_number(req, "theta")
    _require_standing(robot, "move")
    cx = _clamp(x, 0.0, MAX_X)
    ct = _clamp(theta, -MAX_THETA, MAX_THETA)
    robot.move_to(cx, ct)
    return {"ok": True, "x": cx, "theta": ct, "clamped": (cx != x or ct != theta)}


def cmd_wave(robot, req):
    if robot.is_moving():
        raise RequestError("cannot wave while moving; stop first")
    _require_standing(robot, "wave")
    names = sorted(WAVE_RAISED)
    start = robot.current_angles(names)
    limits = robot.limits(names)
    for n in names:
        if n not in start:
            raise RequestError("cannot read current angle of %s" % n)
        lo, hi = limits[n]
        start[n] = _clamp(start[n], lo, hi)  # restore target must be valid too
    names, angles, times = build_wave(start)
    validate_wave(robot, names, angles, times)
    duration = robot.play(names, angles, times)
    return {"ok": True, "duration": duration}


def cmd_dance(robot, req):
    if robot.is_moving():
        raise RequestError("cannot dance while moving; stop first")
    _require_standing(robot, "dance")
    names = sorted(DANCE_JOINTS)
    start = robot.current_angles(names)
    limits = robot.limits(names)
    for n in names:
        if n not in start:
            raise RequestError("cannot read current angle of %s" % n)
        lo, hi = limits[n]
        start[n] = _clamp(start[n], lo, hi)
    names, angles, times = build_dance(start)
    validate_wave(robot, names, angles, times)
    duration = robot.play(names, angles, times)
    return {"ok": True, "duration": duration}


def cmd_greet(robot, req):
    if robot.is_moving():
        raise RequestError("cannot greet while moving; stop first")
    names = sorted(GREET_JOINTS)
    start = robot.current_angles(names)
    limits = robot.limits(names)
    for n in names:
        if n not in start:
            raise RequestError("cannot read current angle of %s" % n)
        lo, hi = limits[n]
        start[n] = _clamp(start[n], lo, hi)
    names, angles, times = build_greet(start)
    validate_wave(robot, names, angles, times)
    duration = robot.play(names, angles, times)
    return {"ok": True, "duration": duration}


def cmd_posture(robot, req):
    name = req.get("name")
    if not isinstance(name, _STRING_TYPES) or name not in ALLOWED_POSTURES:
        raise RequestError("posture must be one of %s" % ", ".join(ALLOWED_POSTURES))
    if robot.is_moving():
        raise RequestError("cannot change posture while moving; stop first")
    if not robot.go_posture(str(name), POSTURE_SPEED):
        raise RequestError("goToPosture(%s) returned False" % name)
    return {"ok": True}


def cmd_eyes(robot, req):
    color = req.get("color")
    if not isinstance(color, _STRING_TYPES) or color not in EYE_COLORS:
        raise RequestError("color must be one of %s" % ", ".join(sorted(EYE_COLORS)))
    robot.set_eyes(EYE_COLORS[color])
    return {"ok": True}


def cmd_set_mock(robot, req):
    if not robot.is_mock:
        raise RequestError("set_mock is only available in mock mode")
    kwargs = {}
    for key in ("sonar_left", "sonar_right"):
        if key in req and req[key] is not None:
            value = req[key]
            if not isinstance(value, numbers.Real) or isinstance(value, bool):
                raise RequestError("%s must be a number" % key)
            kwargs[key] = value
    if req.get("posture") is not None:
        if not isinstance(req["posture"], _STRING_TYPES) or not req["posture"]:
            raise RequestError("posture must be a non-empty string")
        kwargs["posture"] = str(req["posture"])
    if req.get("moving") is not None:
        if not isinstance(req["moving"], bool):
            raise RequestError("moving must be a boolean")
        kwargs["moving"] = req["moving"]
    robot.set_mock(**kwargs)  # validated first: all-or-nothing
    return {"ok": True}


def handle(robot, req):
    """Process one decoded request. Returns (response_dict, should_quit)."""
    if not isinstance(req, dict):
        return {"ok": False, "error": "request must be a JSON object"}, False
    cmd = req.get("cmd")
    if cmd == "ping":
        return {"ok": True, "mock": bool(robot.is_mock)}, False
    if cmd == "sense":
        left, right, moving, posture = robot.sense()
        return {"ok": True, "sonar_left": left, "sonar_right": right,
                "moving": moving, "posture": posture}, False
    if cmd == "move":
        return cmd_move(robot, req), False
    if cmd == "stop":
        robot.stop()
        return {"ok": True}, False
    if cmd == "wave":
        return cmd_wave(robot, req), False
    if cmd == "dance":
        return cmd_dance(robot, req), False
    if cmd == "greet":
        return cmd_greet(robot, req), False
    if cmd == "posture":
        return cmd_posture(robot, req), False
    if cmd == "say":
        text = req.get("text")
        if not isinstance(text, _STRING_TYPES):
            raise RequestError("say requires a text string")
        robot.say(text)
        return {"ok": True}, False
    if cmd == "eyes":
        return cmd_eyes(robot, req), False
    if cmd == "set_mock":
        return cmd_set_mock(robot, req), False
    if cmd == "quit":
        return {"ok": True}, True
    return {"ok": False, "error": "unknown cmd: %r" % (cmd,)}, False


def process_line(robot, line):
    """Decode one input line, return (response_dict, should_quit). Never raises."""
    resp, quit_now = None, False
    req = None
    try:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        req = json.loads(line)
    except Exception as exc:
        resp = {"ok": False, "error": "bad JSON: %s" % exc}
    if resp is None:
        try:
            resp, quit_now = handle(robot, req)
        except RequestError as exc:
            resp = {"ok": False, "error": str(exc)}
        except Exception as exc:
            resp = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
    resp["simulated"] = bool(robot.is_mock)
    return resp, quit_now


def serve(robot, stdin, out):
    while True:
        line = stdin.readline()
        if not line:  # EOF: the parent went away
            break
        line = line.strip()
        if not line:
            continue
        resp, quit_now = process_line(robot, line)
        if not resp.get("ok"):
            log("error response: %s" % resp.get("error"))
        out.write(json.dumps(resp) + "\n")  # ensure_ascii keeps it 7-bit safe
        out.flush()
        if quit_now:
            break


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="NAO JSON-lines bridge (Python 2.7).")
    p.add_argument("--robot", choices=sorted(ROBOTS), default=None)
    p.add_argument("--ip", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--mock", action="store_true", help="no naoqi, no robot")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # Keep the protocol channel clean. naoqi's C++ logger ("[I] qimessaging...")
    # writes straight to file descriptor 1, which a Python-level sys.stdout swap
    # cannot catch. So: keep a private duplicate of descriptor 1 for the JSON
    # protocol, then point descriptor 1 at stderr BEFORE naoqi is imported.
    sys.stdout.flush()
    proto_fd = os.dup(1)
    os.dup2(2, 1)
    if os.name == "nt":
        try:
            import msvcrt
            msvcrt.setmode(proto_fd, os.O_BINARY)
            msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        except Exception:
            pass
    if sys.version_info[0] == 2:
        real_out = os.fdopen(proto_fd, "wb")
    else:  # only for tooling; the bridge itself runs under Python 2.7
        real_out = os.fdopen(proto_fd, "w", encoding="ascii", newline="\n")
    sys.stdout = sys.stderr

    if args.mock:
        robot = MockRobot()
    else:
        ip, port = ROBOTS[args.robot or "nyx"]
        if args.ip:
            ip = args.ip
        if args.port:
            port = args.port
        try:
            robot = NaoRobot(ip, port)
        except Exception as exc:
            log("cannot start real robot: %s: %s" % (type(exc).__name__, exc))
            return 1
    if os.environ.get("NAO_BRIDGE_TEST_NOISE"):
        # Test hook: imitate naoqi's C++ logger writing straight to descriptor 1.
        os.write(1, b"[I] 0 qimessaging.session: simulated native log line\r\n")
    log("ready (mock=%s)" % robot.is_mock)
    try:
        serve(robot, sys.stdin, real_out)
    finally:
        robot.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
