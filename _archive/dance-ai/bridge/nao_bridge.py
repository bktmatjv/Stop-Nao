# -*- coding: utf-8 -*-
r"""Persistent NAO dance bridge (Python 2.7 compatible).

Reads one JSON object per line from stdin and writes exactly one JSON line per
response to stdout (flushed). Everything else (logs, naoqi noise) goes to stderr.
Every response has "ok"; in --mock mode every response also has "simulated": true
(real mode sends "simulated": false).

Protocol (arms + head only, no legs, no walking):
    {"cmd":"ping"}
        -> {"ok":true,"mock":bool}
    {"cmd":"get_limits","names":[...]}
        -> {"ok":true,"limits":{name:[min,max]}}
    {"cmd":"play_keyframes","names":[...],"angles":[[..],..],"times":[[..],..]}
        -> {"ok":true,"task_id":int,"duration":float}   NON-blocking
           (real: motion.post.angleInterpolation(names, angles, times, True))
    {"cmd":"is_running","task_id":int}
        -> {"ok":true,"running":bool}
    {"cmd":"stop_motion"}
        -> {"ok":true,"stopped":[ids],"neutral":bool}  kills running keyframe
           tasks, then returns the arms/head used to a neutral pose
    {"cmd":"posture","name":"StandInit","speed":0.5}
        -> {"ok":true}            (whitelist: StandInit, Stand, StandZero)
    {"cmd":"say","text":"..."}                 -> {"ok":true}   (utf-8)
    {"cmd":"set_eyes","rgb":int | [r,g,b]}     -> {"ok":true}
    {"cmd":"quit"}                             -> {"ok":true}, then the process exits

Usage:
    C:\Python27\python.exe bridge\nao_bridge.py --robot nyx|zyra|sim
    C:\Python27\python.exe bridge\nao_bridge.py --ip 1.2.3.4 --port 9559
    C:\Python27\python.exe bridge\nao_bridge.py --mock      (no naoqi, no robot)

Safety: this bridge never calls rest(), never disables the fall manager or any
other protection, and never releases stiffness. Requests are checked before
anything moves: only arm/head joints, every angle inside the joint limits
(from the robot in real mode), strictly increasing times, and a hard cap on
joint angular velocity.

The process-bridge idea (Python 3 relaunching C:\Python27\python.exe with the
pynaoqi folder on PATH) is adapted from the professor's naoooo/py27-py3.py and
naoooo/nao_show.py (joint names, limit validation before moving, StandInit,
"never release stiffness automatically").
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

# Only these joints may be commanded: arms and head (hands included).
ALLOWED_JOINT = re.compile(
    r"^(HeadYaw|HeadPitch|[LR](ShoulderPitch|ShoulderRoll|ElbowYaw|ElbowRoll|WristYaw|Hand))$"
)

ALLOWED_POSTURES = ("StandInit", "Stand", "StandZero")

# APPROXIMATE NAO H25 (v4/v5) limits in radians, used ONLY in --mock mode.
# Real limits come from the robot (ALMotion.getLimits); these are rounded from
# the public NAOqi documentation and may differ slightly between hardware
# versions. Hands are opening ratios in [0, 1].
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

# Neutral arms-down pose (values taken from the professor's nao_show.py
# "Bajar el brazo" step, mirrored for the left side).
NEUTRAL = {
    "HeadYaw": 0.0, "HeadPitch": 0.0,
    "LShoulderPitch": 1.40, "RShoulderPitch": 1.40,
    "LShoulderRoll": 0.15, "RShoulderRoll": -0.15,
    "LElbowYaw": -1.20, "RElbowYaw": 1.20,
    "LElbowRoll": -0.45, "RElbowRoll": 0.45,
}
NEUTRAL_SECONDS = 2.0

HARD_MAX_VELOCITY = 3.0   # rad/s, defence in depth (choreographies stay <= 2.0)
MAX_KEYFRAME_SECONDS = 120.0
MAX_TASKS_KEPT = 64

_STRING_TYPES = (type(u""), type(""))


def log(msg):
    try:
        sys.stderr.write("[bridge %s] %s\n" % (time.strftime("%H:%M:%S"), msg))
        sys.stderr.flush()
    except Exception:
        pass


def _is_number(value):
    return (isinstance(value, numbers.Real) and not isinstance(value, bool)
            and not math.isnan(value) and not math.isinf(value))


class RequestError(Exception):
    """The request is invalid or unsafe; reported as {"ok": false}."""


# ---------------------------------------------------------------------------
# Robot back-ends
# ---------------------------------------------------------------------------
class MockRobot(object):
    """Stand-in robot: no naoqi import; play_keyframes only simulates time."""

    is_mock = True

    def __init__(self):
        self._tasks = {}      # id -> [end_time, stopped]
        self._next_id = 1
        self.pose = {}        # last commanded angle per joint
        self.used = set()
        self.eyes = 0xFFFFFF

    def limits(self, names):
        missing = [n for n in names if n not in APPROX_LIMITS]
        if missing:
            raise RequestError("unknown joint(s): %s" % ", ".join(sorted(missing)))
        return dict((n, APPROX_LIMITS[n]) for n in names)

    def current_angles(self, names):
        return dict((n, self.pose[n]) for n in names if n in self.pose)

    def play(self, names, angles, times):
        duration = max(t[-1] for t in times)
        tid = self._next_id
        self._next_id += 1
        self._tasks[tid] = [time.time() + duration, False]
        for n, a in zip(names, angles):
            self.pose[n] = a[-1]
            self.used.add(n)
        log("MOCK play_keyframes id=%d joints=%d duration=%.2fs (simulated, no robot moved)"
            % (tid, len(names), duration))
        return tid, duration

    def is_running(self, tid):
        task = self._tasks.get(tid)
        return bool(task) and (not task[1]) and time.time() < task[0]

    def stop_motion(self):
        stopped = []
        for tid, task in self._tasks.items():
            if not task[1] and time.time() < task[0]:
                task[1] = True
                stopped.append(tid)
        neutral = False
        for n in self.used:
            if n in NEUTRAL:
                self.pose[n] = NEUTRAL[n]
                neutral = True
        log("MOCK stop_motion stopped=%s neutral=%s" % (sorted(stopped), neutral))
        return sorted(stopped), neutral

    def go_posture(self, name, speed):
        for n in NEUTRAL:
            self.pose[n] = NEUTRAL[n]
        log("MOCK goToPosture(%s, %.2f)" % (name, speed))
        return True

    def say(self, text):
        if isinstance(text, type(u"")):
            text = text.encode("utf-8")
        log("MOCK say: %s" % text)

    def set_eyes(self, rgb):
        self.eyes = rgb
        log("MOCK eyes 0x%06X" % rgb)

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
        self.posture = ALProxy("ALRobotPosture", ip, port)
        self.tts = None
        self.leds = None
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
        self._limits_cache = {}
        self._tasks = []
        self.used = set()

        # Same start-up as the professor's nao_show.py. Nothing is disabled.
        self.motion.stiffnessInterpolation("Body", 1.0, 1.0)
        if not self.posture.goToPosture("StandInit", 0.5):
            log("goToPosture(StandInit) returned False")

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
        tid = self.motion.post.angleInterpolation(names, angles, times, True)
        self._tasks.append(tid)
        del self._tasks[:-MAX_TASKS_KEPT]
        self.used.update(names)
        return tid, max(t[-1] for t in times)

    def is_running(self, tid):
        return bool(self.motion.isRunning(tid))

    def stop_motion(self):
        stopped, errors = [], []
        for tid in list(self._tasks):
            try:
                if self.motion.isRunning(tid):
                    self.motion.stop(tid)
                    stopped.append(tid)
            except Exception as exc:
                errors.append("stop(%s): %s" % (tid, exc))
        if self.used:
            try:
                self.motion.killTasksUsingResources(sorted(self.used))
            except Exception as exc:
                errors.append("killTasksUsingResources: %s" % exc)
        self._tasks = []
        if errors:
            log("stop_motion warnings: %s" % "; ".join(errors))
        neutral = False
        joints = sorted(n for n in self.used if n in NEUTRAL)
        if joints:
            try:
                self.motion.angleInterpolation(
                    joints, [[NEUTRAL[n]] for n in joints],
                    [[NEUTRAL_SECONDS] for n in joints], True)
                neutral = True
            except Exception as exc:
                log("neutral pose failed: %s" % exc)
        if errors and not stopped and not neutral:
            raise RequestError("could not stop motion: %s" % "; ".join(errors))
        return stopped, neutral

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
        if self.leds is not None:
            self.leds.fadeRGB("FaceLeds", rgb, 0.2)

    def close(self):
        # Never release stiffness or rest the robot automatically.
        pass


# ---------------------------------------------------------------------------
# Request validation and command handlers
# ---------------------------------------------------------------------------
def _validate_keyframes(robot, req):
    names, angles, times = req.get("names"), req.get("angles"), req.get("times")
    if not isinstance(names, list) or not names:
        raise RequestError("names must be a non-empty list")
    if not all(isinstance(n, _STRING_TYPES) for n in names):
        raise RequestError("names must be strings")
    if len(set(names)) != len(names):
        raise RequestError("names must be unique")
    forbidden = [n for n in names if not ALLOWED_JOINT.match(n)]
    if forbidden:
        raise RequestError("joint(s) not allowed (arms and head only): %s"
                           % ", ".join(sorted(forbidden)))
    for label, seq in (("angles", angles), ("times", times)):
        if not isinstance(seq, list) or len(seq) != len(names):
            raise RequestError("%s must be a list with one list per joint" % label)
        for row in seq:
            if not isinstance(row, list) or not row or not all(_is_number(v) for v in row):
                raise RequestError("%s rows must be non-empty lists of finite numbers" % label)
    for n, a_row, t_row in zip(names, angles, times):
        if len(a_row) != len(t_row):
            raise RequestError("%s: angles and times differ in length" % n)
        if t_row[0] <= 0 or any(b <= a for a, b in zip(t_row, t_row[1:])):
            raise RequestError("%s: times must be > 0 and strictly increasing" % n)
        if t_row[-1] > MAX_KEYFRAME_SECONDS:
            raise RequestError("%s: motion longer than %.0f s" % (n, MAX_KEYFRAME_SECONDS))

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
    fast = []
    for n, a_row, t_row in zip(names, angles, times):
        prev_a, prev_t = start.get(n), 0.0
        for a, t in zip(a_row, t_row):
            if prev_a is not None and abs(a - prev_a) / (t - prev_t) > HARD_MAX_VELOCITY:
                fast.append("%s %.2f rad/s" % (n, abs(a - prev_a) / (t - prev_t)))
                break
            prev_a, prev_t = a, t
    if fast:
        raise RequestError("too fast (> %.1f rad/s): %s" % (HARD_MAX_VELOCITY, "; ".join(fast)))
    return names, angles, times


def _parse_rgb(value):
    if isinstance(value, list) and len(value) == 3 and all(
            _is_number(v) and 0 <= v <= 255 for v in value):
        r, g, b = [int(v) for v in value]
        return (r << 16) | (g << 8) | b
    if _is_number(value) and 0 <= value <= 0xFFFFFF:
        return int(value)
    raise RequestError("rgb must be an int 0..0xFFFFFF or [r,g,b] with 0..255")


def handle(robot, req):
    """Process one decoded request. Returns (response_dict, should_quit)."""
    if not isinstance(req, dict):
        return {"ok": False, "error": "request must be a JSON object"}, False
    cmd = req.get("cmd")
    if cmd == "ping":
        return {"ok": True, "mock": bool(robot.is_mock)}, False
    if cmd == "get_limits":
        names = req.get("names")
        if not isinstance(names, list) or not all(isinstance(n, _STRING_TYPES) for n in names):
            raise RequestError("get_limits requires names: list of strings")
        lims = robot.limits(names)
        return {"ok": True, "limits": dict((n, [lims[n][0], lims[n][1]]) for n in names)}, False
    if cmd == "play_keyframes":
        names, angles, times = _validate_keyframes(robot, req)
        tid, duration = robot.play(names, angles, times)
        return {"ok": True, "task_id": tid, "duration": duration}, False
    if cmd == "is_running":
        tid = req.get("task_id")
        if not isinstance(tid, numbers.Integral) or isinstance(tid, bool):
            raise RequestError("is_running requires an integer task_id")
        return {"ok": True, "running": bool(robot.is_running(tid))}, False
    if cmd == "stop_motion":
        stopped, neutral = robot.stop_motion()
        return {"ok": True, "stopped": stopped, "neutral": neutral}, False
    if cmd == "posture":
        name = req.get("name")
        speed = req.get("speed", 0.5)
        if name not in ALLOWED_POSTURES:
            raise RequestError("posture must be one of %s" % ", ".join(ALLOWED_POSTURES))
        if not _is_number(speed) or not 0 < speed <= 1.0:
            raise RequestError("speed must be in (0, 1]")
        if not robot.go_posture(name, float(speed)):
            return {"ok": False, "error": "goToPosture(%s) returned False" % name}, False
        return {"ok": True}, False
    if cmd == "say":
        text = req.get("text")
        if not isinstance(text, _STRING_TYPES):
            raise RequestError("say requires a text string")
        robot.say(text)
        return {"ok": True}, False
    if cmd == "set_eyes":
        robot.set_eyes(_parse_rgb(req.get("rgb")))
        return {"ok": True}, False
    if cmd == "quit":
        return {"ok": True}, True
    return {"ok": False, "error": "unknown cmd: %r" % (cmd,)}, False


def process_line(robot, line):
    """Decode one input line, return (response_dict, should_quit). Never raises."""
    resp, quit_now = None, False
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
        out.write((json.dumps(resp) + "\n").encode("ascii"))  # ensure_ascii: 7-bit safe
        out.flush()
        if quit_now:
            break


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="NAO dance JSON-lines bridge (Python 2.7).")
    p.add_argument("--robot", choices=sorted(ROBOTS), default=None)
    p.add_argument("--ip", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--mock", action="store_true", help="no naoqi, no robot")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # Keep the protocol channel clean: stray prints go to stderr.
    real_out = sys.stdout
    sys.stdout = sys.stderr
    if os.name == "nt":
        try:
            import msvcrt
            msvcrt.setmode(real_out.fileno(), os.O_BINARY)
            msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        except Exception:
            pass

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
    log("ready (mock=%s)" % robot.is_mock)
    try:
        serve(robot, sys.stdin, getattr(real_out, "buffer", real_out))
    finally:
        robot.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
