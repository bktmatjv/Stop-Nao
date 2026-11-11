"""Python 3 client for the persistent Python 2.7 NAO bridge.

The idea of driving naoqi (Python 2.7 only) from a modern Python through a
subprocess is adapted from the professor's ``naoooo/py27-py3.py`` and
``naoooo/nao_show.py`` (which relaunches itself with ``C:\\Python27\\python.exe``
and prepends the pynaoqi folder to PATH so the DLLs load). Unlike the original,
which spawns a new process per call, the bridge is launched ONCE and spoken to
with JSON lines over stdin/stdout.
"""
import json
import os
import queue
import subprocess
import sys
import threading
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIDGE_SCRIPT = os.path.join(ROOT, "bridge", "nao_bridge.py")
SDK_DIR = r"C:\Python27\Lib\site-packages"


class NaoBridgeError(RuntimeError):
    """The bridge process died, timed out or returned something unusable."""


class NaoClient:
    """Talks to ``bridge/nao_bridge.py`` running under Python 2.7.

    Usage::

        with NaoClient(robot="nyx") as nao:
            state = nao.sense()
            nao.move(0.2, 0.0)

    Calls are serialized with a lock, so one client can be shared by threads.
    Every method returns the bridge's decoded JSON reply (a dict with ``ok``);
    refusals come back as ``{"ok": False, "error": ...}``, not as exceptions.
    ``NaoBridgeError`` is raised only when the bridge itself is unusable.
    """

    def __init__(self, robot=None, mock=False, ip=None, port=None, timeout=30.0,
                 python27=r"C:\Python27\python.exe"):
        self.timeout = timeout
        self._lock = threading.Lock()
        cmd = [python27, BRIDGE_SCRIPT]
        if mock:
            cmd.append("--mock")
        else:
            if robot:
                cmd += ["--robot", robot]
            if ip:
                cmd += ["--ip", str(ip)]
            if port:
                cmd += ["--port", str(port)]
        env = os.environ.copy()
        env["PATH"] = SDK_DIR + os.pathsep + env.get("PATH", "")
        try:
            self._proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=env, text=True,
                encoding="utf-8", bufsize=1,
            )
        except OSError as exc:
            raise NaoBridgeError("cannot launch bridge with %r: %s" % (python27, exc))
        self._lines = queue.Queue()
        self._stderr_tail = deque(maxlen=30)
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        threading.Thread(target=self._pump_stderr, daemon=True).start()

    # -- plumbing ---------------------------------------------------------
    def _pump_stdout(self):
        try:
            for line in self._proc.stdout:
                self._lines.put(line)
        except (OSError, ValueError):
            pass
        self._lines.put(None)  # EOF sentinel

    def _pump_stderr(self):
        try:
            for line in self._proc.stderr:
                self._stderr_tail.append(line.rstrip())
        except (OSError, ValueError):
            pass

    def _dead(self, why):
        tail = "\n".join(self._stderr_tail)
        code = self._proc.poll()
        return NaoBridgeError("NAO bridge %s (exit code %s). stderr tail:\n%s"
                              % (why, code, tail))

    def _write_line(self, line):
        try:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except (OSError, ValueError):
            raise self._dead("is not running")

    def _read_line(self):
        try:
            line = self._lines.get(timeout=self.timeout)
        except queue.Empty:
            raise self._dead("did not answer within %.1fs" % self.timeout)
        if line is None:
            self._lines.put(None)  # keep the sentinel for later calls
            raise self._dead("died")
        return line.rstrip("\r\n")  # the Windows pipe may deliver CRLF

    MAX_NOISE_LINES = 50  # non-JSON lines tolerated before one reply

    def _call(self, **request):
        with self._lock:  # one request/response pair at a time
            self._write_line(json.dumps(request))
            for _ in range(self.MAX_NOISE_LINES):
                line = self._read_line()
                if not line.lstrip().startswith("{"):
                    # Native library log that slipped into stdout: keep it as a log.
                    self._stderr_tail.append("[stdout noise] " + line)
                    continue
                try:
                    return json.loads(line)
                except json.JSONDecodeError as exc:
                    raise NaoBridgeError("bridge returned invalid JSON: %s (line %r)" % (exc, line[:120]))
            raise self._dead("sent only non-JSON output")

    # -- public API -------------------------------------------------------
    def ping(self):
        return self._call(cmd="ping")

    def sense(self):
        """Sonars, ``moving`` and ``posture``."""
        return self._call(cmd="sense")

    def move(self, x, theta):
        """Non-blocking walk request; the bridge clamps and may refuse."""
        return self._call(cmd="move", x=x, theta=theta)

    def stop(self):
        return self._call(cmd="stop")

    def wave(self):
        """Right-arm wave (blocks up to ~4 s on the real robot)."""
        return self._call(cmd="wave")

    def dance(self):
        """Celebratory dance with arms and head (blocks up to ~5.6 s on the robot)."""
        return self._call(cmd="dance")

    def greet(self):
        """Quick salute greeting: right-arm raise + tip (blocks up to ~2 s)."""
        return self._call(cmd="greet")

    def posture(self, name):
        """Go to "Sit" or "StandInit" (anything else is refused)."""
        return self._call(cmd="posture", name=name)

    def say(self, text):
        return self._call(cmd="say", text=text)

    def eyes(self, color):
        """Face LEDs: green, yellow, red, blue or white."""
        return self._call(cmd="eyes", color=color)

    def set_mock(self, **kwargs):
        """Override simulated state (mock only): sonar_left, sonar_right,
        posture, moving."""
        return self._call(cmd="set_mock", **kwargs)

    def close(self):
        proc = self._proc
        if proc.poll() is None:
            try:
                with self._lock:
                    self._write_line(json.dumps({"cmd": "quit"}))
                proc.wait(timeout=3)
            except (NaoBridgeError, subprocess.TimeoutExpired):
                proc.kill()
                proc.wait()
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                stream.close()
            except (OSError, ValueError):
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False
