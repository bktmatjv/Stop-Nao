"""Non-blocking front for ``NaoClient``: every robot call runs on ONE worker thread.

The camera/main loop must never wait for the bridge (``posture`` blocks for
seconds, every command is a JSON round trip). ``AsyncClient`` mirrors the
wrapped client's methods but they return a ``Ticket`` at once; the worker runs
the commands strictly in FIFO order. ``stop()`` is the exception: it drops every
queued-but-unstarted command and goes to the front of the queue. (A command the
worker is already running cannot be interrupted: the bridge is a single
request/response pipe, so the stop is sent right after it.)

The worker also polls ``sense`` every ``sense_interval`` s and caches the latest
reading with its timestamp, so decisions read ``latest_sense()`` without I/O.
"""
from __future__ import annotations

import sys
import threading
import time
from collections import deque

DEFAULT_SENSE_INTERVAL = 0.5  # s between background sonar/posture polls
STOP = "stop"


class Ticket(dict):
    """Immediate acknowledgement of a queued command (a dict so ``ack.get("ok")`` works).

    ``{"ok": True, "queued": True}`` until the worker finishes; afterwards the real
    outcome is in ``response`` / ``error`` / ``latency`` / ``cancelled``.
    """

    def __init__(self, cmd: str, enqueued_at: float = 0.0):
        super().__init__(ok=True, queued=True)
        self.cmd = cmd
        self.enqueued_at = enqueued_at
        self.finished_at: float | None = None
        self.response = None
        self.error = ""          # "<cmd>: <text>" when the command failed (not when cancelled)
        self.cancelled = False   # dropped by a stop() or by close() before it started
        self.latency = 0.0       # seconds the worker spent on it
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: list = []

    @property
    def done(self) -> bool:
        return self._event.is_set()

    @property
    def succeeded(self) -> bool:
        return self.done and not self.error and not self.cancelled

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def add_done_callback(self, fn) -> None:
        """Call ``fn(ticket)`` when finished (at once if it already is); never raises."""
        with self._lock:
            if not self._event.is_set():
                self._callbacks.append(fn)
                return
        _safe(fn, self)

    def _finish(self, response, error: str, latency: float, now: float,
                cancelled: bool = False) -> None:
        with self._lock:
            if self._event.is_set():
                return
            self.response, self.error, self.latency = response, error, latency
            self.cancelled, self.finished_at = cancelled, now
            callbacks, self._callbacks = self._callbacks, []
            self._event.set()
        for fn in callbacks:
            _safe(fn, self)


def _safe(fn, *args) -> None:
    try:
        fn(*args)
    except Exception as exc:  # a callback must never kill the worker or the caller
        print(f"WARNING: async callback failed: {exc}", file=sys.stderr)


class AsyncClient:
    """Wraps a ``NaoClient`` (or any object with the same methods) behind one worker thread."""

    is_async = True

    def __init__(self, client, sense_interval: float | None = DEFAULT_SENSE_INTERVAL,
                 on_result=None, clock=time.monotonic, join_timeout: float = 5.0):
        self._client = client
        self.sense_interval = sense_interval
        self.on_result = on_result  # on_result(cmd, response, error, latency)
        self.clock = clock
        self.join_timeout = join_timeout
        self.last_error: str | None = None
        self._cv = threading.Condition()
        self._queue: deque = deque()
        self._closing = False
        self._closed = False
        self._sense = None            # (reading, started_at)
        self._last_sense_at: float | None = None
        self._sense_wanted = sense_interval is not None
        self._thread = threading.Thread(target=self._run, name="nao-async", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ public API
    def call(self, name: str, *args, **kwargs) -> Ticket:
        """Enqueue ``client.<name>(*args, **kwargs)``; returns immediately."""
        ticket = Ticket(name, self.clock())
        dropped = []
        with self._cv:
            if self._closing:
                dropped.append(ticket)
            elif name == STOP:
                existing = next((item for item in self._queue if item[1] == STOP), None)
                if existing is not None:
                    # A stop is already waiting: share its ticket, but still drop everything
                    # queued behind it so a repeated e-stop never lets later commands run.
                    ticket = existing[0]
                    dropped = [item[0] for item in self._queue if item is not existing]
                    self._queue.clear()
                    self._queue.appendleft(existing)
                else:
                    dropped = [item[0] for item in self._queue]
                    self._queue.clear()
                    self._queue.appendleft((ticket, name, args, kwargs))
            else:
                self._queue.append((ticket, name, args, kwargs))
            self._cv.notify_all()
        for t in dropped:
            t._finish({"ok": False, "error": "cancelled"}, "", 0.0, self.clock(), cancelled=True)
        return ticket

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        attr = getattr(self._client, name)  # AttributeError for unknown methods
        if not callable(attr):
            return attr

        def method(*args, **kwargs) -> Ticket:
            return self.call(name, *args, **kwargs)
        method.__name__ = name
        return method

    def stop(self) -> Ticket:
        return self.call(STOP)

    def latest_sense(self):
        """``(reading, age_s)`` of the newest background poll, or ``(None, None)``."""
        with self._cv:
            if self._sense is None:
                return None, None
            reading, started = self._sense
        return dict(reading), self.clock() - started

    def pending(self) -> int:
        with self._cv:
            return len(self._queue)

    def close(self) -> None:
        """Drain the queue, stop and join the worker (bounded), then close the wrapped client."""
        with self._cv:
            if self._closed:
                return
            self._closed = self._closing = True
            self._cv.notify_all()
        self._thread.join(self.join_timeout)
        with self._cv:
            leftover = [item[0] for item in self._queue]
            self._queue.clear()
        for t in leftover:
            t._finish({"ok": False, "error": "cancelled"}, "", 0.0, self.clock(), cancelled=True)
        closer = getattr(self._client, "close", None)
        if closer is None:
            return
        if self._thread.is_alive():  # worker stuck in a call: do not block the caller on it
            threading.Thread(target=lambda: _safe(closer), daemon=True).start()
        else:
            try:
                closer()
            except Exception as exc:
                print(f"WARNING: error closing the bridge: {exc}", file=sys.stderr)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False

    # --------------------------------------------------------------- worker
    def _sense_state(self):
        """-> (due, overdue, wait): due = poll when idle; overdue = poll even before queued work."""
        if self.sense_interval is None or self._closing:
            return False, False, None
        now = self.clock()
        if self._last_sense_at is None:
            return True, True, 0.0
        elapsed = now - self._last_sense_at
        remaining = self.sense_interval - elapsed
        return (self._sense_wanted or remaining <= 0), remaining <= 0, max(0.0, remaining)

    def _run(self) -> None:
        while True:
            item = None
            poll = False
            with self._cv:
                while True:
                    due, overdue, wait = self._sense_state()
                    head_is_stop = bool(self._queue) and self._queue[0][1] == STOP
                    if self._queue and not (overdue and not head_is_stop):
                        item = self._queue.popleft()
                        break
                    if due:
                        poll = True
                        break
                    if self._closing:
                        return
                    self._cv.wait(timeout=wait)
            if poll:
                self._poll_sense()
            else:
                self._execute(*item)

    def _poll_sense(self) -> None:
        started = self.clock()
        self._last_sense_at = started
        self._sense_wanted = False
        try:
            reading = self._client.sense()
        except Exception as exc:
            reading = {"ok": False, "error": "sense: %s" % exc}
        if not isinstance(reading, dict):
            reading = {"ok": False, "error": "sense: unusable reply"}
        with self._cv:
            self._sense = (reading, started)

    def _execute(self, ticket: Ticket, name: str, args, kwargs) -> None:
        started = self.clock()
        response, error = None, ""
        try:
            response = getattr(self._client, name)(*args, **kwargs)
            if isinstance(response, dict) and not response.get("ok", True):
                error = "%s: %s" % (name, response.get("error") or "unspecified bridge error")
        except Exception as exc:
            error = "%s: %s" % (name, exc)
        now = self.clock()
        if name == "sense" and isinstance(response, dict):
            with self._cv:
                self._sense = (response, started)
                self._last_sense_at = started
        self.last_error = error or None
        if name != "sense" and self.sense_interval is not None:
            self._sense_wanted = True  # look again as soon as the worker is idle
        ticket._finish(response, error, now - started, now)
        if self.on_result is not None:
            _safe(self.on_result, name, response, error, now - started)
