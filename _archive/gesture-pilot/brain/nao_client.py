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
            nao.act("ADVANCE")
    """

    def __init__(self, robot="nyx", mock=False, python27=r"C:\Python27\python.exe",
                 ip=None, port=None, timeout=30.0):
        self.timeout = timeout
        cmd = [python27, BRIDGE_SCRIPT]
        if mock:
            cmd.append("--mock")
        else:
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
        return line

    def _call(self, **request):
        self._write_line(json.dumps(request))
        try:
            return json.loads(self._read_line())
        except json.JSONDecodeError as exc:
            raise NaoBridgeError("bridge returned invalid JSON: %s" % exc)

    # -- public API -------------------------------------------------------
    def ping(self):
        return self._call(cmd="ping")

    def sense(self):
        return self._call(cmd="sense")

    def act(self, action):
        return self._call(cmd="act", action=action)

    def say(self, text):
        return self._call(cmd="say", text=text)

    def set_mock(self, sonar_left, sonar_right):
        """Override simulated sonar values (mock mode only)."""
        return self._call(cmd="set_mock", sonar_left=sonar_left,
                          sonar_right=sonar_right)

    def close(self):
        proc = self._proc
        if proc.poll() is None:
            try:
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
