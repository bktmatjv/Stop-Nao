# -*- coding: utf-8 -*-
r"""Persistent NAO bridge (Python 2.7 compatible).

Reads one JSON object per line from stdin and writes exactly one JSON line per
response to stdout (flushed). Everything else (logs, naoqi noise) goes to stderr.

Protocol:
    {"cmd":"ping"}                         -> {"ok":true,"mock":bool}
    {"cmd":"sense"}                        -> {"ok":true,"sonar_left":f,"sonar_right":f,"moving":bool}
    {"cmd":"act","action":"<ACTION>"}      -> {"ok":true,"action":"<ACTION>"} | {"ok":false,"error":...}
    {"cmd":"say","text":"..."}             -> {"ok":true}
    {"cmd":"set_mock","sonar_left":x,"sonar_right":y}  (mock mode only) -> {"ok":true}
    {"cmd":"quit"}                         -> {"ok":true}, then the process exits

Usage:
    C:\Python27\python.exe bridge\nao_bridge.py --robot nyx|zyra|sim
    C:\Python27\python.exe bridge\nao_bridge.py --ip 1.2.3.4 --port 9559
    C:\Python27\python.exe bridge\nao_bridge.py --mock      (no naoqi, no robot)

The process-bridge idea (Python 3 relaunching C:\Python27\python.exe with the
pynaoqi folder on PATH) is adapted from the professor's naoooo/py27-py3.py and
naoooo/nao_show.py.
"""
from __future__ import print_function

import argparse
import json
import os
import sys

SDK_DIR = r"C:\Python27\Lib\site-packages"

ROBOTS = {
    "nyx": ("127.0.0.1", 59242),
    "zyra": ("192.168.108.90", 9559),
    "sim": ("10.190.103.36", 51940),
}

SONAR_LEFT_KEY = "Device/SubDeviceList/US/Left/Sensor/Value"
SONAR_RIGHT_KEY = "Device/SubDeviceList/US/Right/Sensor/Value"

DEFAULT_SONAR = 1.5
STEP_FORWARD_M = 0.2
TURN_RAD = 0.785
OBSTACLE_TEXT = u"Hay un obst\u00e1culo, no puedo avanzar"
ASK_CLOSER_TEXT = u"Ac\u00e9rcate, no te veo bien"

RED = 0xFF0000
WHITE = 0xFFFFFF

_STRING_TYPES = (type(u""), type(""))


def log(msg):
    try:
        sys.stderr.write("[bridge] %s\n" % msg)
        sys.stderr.flush()
    except Exception:
        pass


class MockRobot(object):
    """Stand-in robot: no naoqi import, logs every action to stderr."""

    is_mock = True

    def __init__(self):
        self.sonar_left = DEFAULT_SONAR
        self.sonar_right = DEFAULT_SONAR
        self.moving = False
        self.eyes = WHITE

    def move_to(self, x, y, theta):
        log("MOCK moveTo(%s, %s, %s)" % (x, y, theta))

    def stop(self):
        log("MOCK stopMove()")

    def say(self, text):
        if isinstance(text, type(u"")):
            text = text.encode("utf-8")
        log("MOCK say: %s" % text)

    def set_eyes(self, rgb):
        self.eyes = rgb
        log("MOCK eyes 0x%06X" % rgb)

    def sense(self):
        return self.sonar_left, self.sonar_right, self.moving

    def set_mock(self, left=None, right=None):
        if left is not None:
            self.sonar_left = float(left)
        if right is not None:
            self.sonar_right = float(right)

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
        self.eyes = WHITE

        self.motion.stiffnessInterpolation("Body", 1.0, 1.0)
        if not self.posture.goToPosture("StandInit", 0.5):
            log("goToPosture(StandInit) returned False")
        self.motion.moveInit()

    def move_to(self, x, y, theta):
        self.motion.post.moveTo(x, y, theta)  # non-blocking

    def stop(self):
        self.motion.stopMove()

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
        self.eyes = rgb
        self.leds.fadeRGB("FaceLeds", rgb, 0.2)

    def sense(self):
        left = float(self.memory.getData(SONAR_LEFT_KEY))
        right = float(self.memory.getData(SONAR_RIGHT_KEY))
        return left, right, bool(self.motion.moveIsActive())

    def set_mock(self, left=None, right=None):
        raise ValueError("set_mock is only available in mock mode")

    def close(self):
        if self.sonar is not None:
            try:
                self.sonar.unsubscribe("nao_bridge")
            except Exception:
                pass


def _restore_eyes(robot):
    if robot.eyes != WHITE:
        robot.set_eyes(WHITE)


def _move(x, y, theta):
    def run(robot):
        _restore_eyes(robot)
        robot.move_to(x, y, theta)
    return run


def _stop(robot):
    _restore_eyes(robot)
    robot.stop()


def _warn(robot):
    robot.stop()
    robot.set_eyes(RED)
    robot.say(OBSTACLE_TEXT)


def _ask_closer(robot):
    robot.say(ASK_CLOSER_TEXT)


def _wait(robot):
    pass


# Single action table: action name -> callable(robot).
ACTIONS = {
    "ADVANCE": _move(STEP_FORWARD_M, 0.0, 0.0),
    "TURN_LEFT": _move(0.0, 0.0, TURN_RAD),
    "TURN_RIGHT": _move(0.0, 0.0, -TURN_RAD),
    "STOP": _stop,
    "WARN_OBSTACLE": _warn,
    "ASK_CLOSER": _ask_closer,
    "WAIT": _wait,
}


def handle(robot, req):
    """Process one decoded request. Returns (response_dict, should_quit)."""
    if not isinstance(req, dict):
        return {"ok": False, "error": "request must be a JSON object"}, False
    cmd = req.get("cmd")
    if cmd == "ping":
        return {"ok": True, "mock": bool(robot.is_mock)}, False
    if cmd == "sense":
        left, right, moving = robot.sense()
        return {"ok": True, "sonar_left": left, "sonar_right": right,
                "moving": moving}, False
    if cmd == "act":
        action = req.get("action")
        fn = ACTIONS.get(action) if isinstance(action, _STRING_TYPES) else None
        if fn is None:
            return {"ok": False, "error": "unknown action: %s" % (action,)}, False
        fn(robot)
        return {"ok": True, "action": action}, False
    if cmd == "say":
        text = req.get("text")
        if not isinstance(text, _STRING_TYPES):
            return {"ok": False, "error": "say requires a text string"}, False
        robot.say(text)
        return {"ok": True}, False
    if cmd == "set_mock":
        if not robot.is_mock:
            return {"ok": False,
                    "error": "set_mock is only available in mock mode"}, False
        robot.set_mock(req.get("sonar_left"), req.get("sonar_right"))
        return {"ok": True}, False
    if cmd == "quit":
        return {"ok": True}, True
    return {"ok": False, "error": "unknown cmd: %r" % (cmd,)}, False


def process_line(robot, line):
    """Decode one input line, return (response_dict, should_quit). Never raises."""
    try:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        req = json.loads(line)
    except Exception as exc:
        return {"ok": False, "error": "bad JSON: %s" % exc}, False
    try:
        return handle(robot, req)
    except Exception as exc:
        return {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}, False


def serve(robot, stdin, out):
    while True:
        line = stdin.readline()
        if not line:  # EOF: the parent went away
            break
        line = line.strip()
        if not line:
            continue
        resp, quit_now = process_line(robot, line)
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
        serve(robot, sys.stdin, real_out)
    finally:
        robot.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
