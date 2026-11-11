"""Python 3 client for the persistent Python 2.7 NAO dance bridge.

The idea of driving naoqi (Python 2.7 only) from a modern Python through a
subprocess is adapted from the professor's ``naoooo/py27-py3.py`` and
``naoooo/nao_show.py`` (which relaunch themselves with ``C:\\Python27\\python.exe``
and prepend the pynaoqi folder to PATH so the DLLs load). Unlike the originals,
which spawn a new process per call, the bridge is launched ONCE and spoken to
with JSON lines over stdin/stdout.

Calls are serialized with a lock, so ``stop_motion`` may safely be called from
another thread while a controller thread is polling ``is_running``.
"""
import json
import os
import queue
import subprocess
import threading
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BRIDGE_SCRIPT = os.path.join(ROOT, "bridge", "nao_bridge.py")
PYTHON27 = r"C:\Python27\python.exe"
SDK_DIR = r"C:\Python27\Lib\site-packages"


class NaoBridgeError(RuntimeError):
    """The bridge process died, timed out, or answered ``ok: false``."""


class NaoClient:
    """Talks to ``bridge/nao_bridge.py`` running under Python 2.7.

    Usage::

        with NaoClient(mock=True) as nao:
            tid = nao.play_keyframes(names, angles, times)
            while nao.is_running(tid):
                ...

    ``robot`` is ``nyx``, ``zyra`` or ``sim`` (default ``nyx`` inside the
    bridge); ``ip``/``port`` override it. ``mock=True`` never touches naoqi or
    any robot.
    """

    def __init__(self, robot=None, mock=False, ip=None, port=None, timeout=30.0,
                 python27=PYTHON27):
        self.timeout = timeout
        self.mock = mock
        self.simulated = None  # "simulated" flag of the latest bridge response
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
        self._lock = threading.Lock()
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

    @property
    def stderr_tail(self):
        return "\n".join(self._stderr_tail)

    def _dead(self, why):
        code = self._proc.poll()
        return NaoBridgeError("NAO bridge %s (exit code %s). stderr tail:\n%s"
                              % (why, code, self.stderr_tail))

    def _write_line(self, line):
        try:
            # text mode turns "\n" into CRLF on Windows; the bridge strips it.
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except (OSError, ValueError):
            raise self._dead("is not running")

    def _read_line(self, timeout):
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            raise self._dead("did not answer within %.1fs" % timeout)
        if line is None:
            self._lines.put(None)  # keep the sentinel for later calls
            raise self._dead("died")
        return line

    def _call(self, _timeout=None, **request):
        """Send one request, return the decoded response; raise on ``ok: false``."""
        with self._lock:
            self._write_line(json.dumps(request))
            raw = self._read_line(self.timeout if _timeout is None else _timeout)
        try:
            resp = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise NaoBridgeError("bridge returned invalid JSON: %s" % exc)
        self.simulated = resp.get("simulated")
        if not resp.get("ok"):
            raise NaoBridgeError("bridge refused %r: %s" % (request.get("cmd"), resp.get("error")))
        return resp

    # -- public API -------------------------------------------------------
    def ping(self):
        return self._call(cmd="ping")

    def get_limits(self, names):
        """Return ``{joint: (min, max)}`` (approximate table in mock mode)."""
        resp = self._call(cmd="get_limits", names=list(names))
        return {n: (float(lo), float(hi)) for n, (lo, hi) in resp["limits"].items()}

    def play_keyframes(self, names, angles, times):
        """Start a NON-blocking keyframe motion; return its integer task id."""
        resp = self._call(cmd="play_keyframes", names=list(names),
                          angles=[list(a) for a in angles],
                          times=[list(t) for t in times])
        return int(resp["task_id"])

    def is_running(self, task_id):
        return bool(self._call(cmd="is_running", task_id=task_id)["running"])

    def stop_motion(self):
        """Kill the running keyframe task(s) and return the arms to neutral."""
        return self._call(cmd="stop_motion")

    def posture(self, name="StandInit", speed=0.5):
        return self._call(cmd="posture", name=name, speed=speed)

    def say(self, text):
        return self._call(cmd="say", text=text)

    def set_eyes(self, rgb):
        """``rgb`` is an int 0xRRGGBB or an ``(r, g, b)`` tuple of 0..255."""
        value = list(rgb) if isinstance(rgb, (tuple, list)) else int(rgb)
        return self._call(cmd="set_eyes", rgb=value)

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
