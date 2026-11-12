"""AsyncClient: the caller never blocks, commands stay ordered, stop has priority."""
import os
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from brain.async_client import AsyncClient, Ticket  # noqa: E402


class SlowClient:
    """Fake NaoClient whose commands sleep (real round trips take 0.3 s .. seconds)."""

    def __init__(self, delays=None, fail=()):
        self.delays = delays or {}
        self.fail = set(fail)
        self.calls = []
        self.closed = False
        self.started = threading.Event()
        self.release = threading.Event()
        self.hold = None  # command name that blocks until ``release`` is set

    def _run(self, name, *args):
        self.calls.append((name,) + args)
        self.started.set()
        if self.hold == name:
            self.release.wait(5)
        else:
            time.sleep(self.delays.get(name, 0.0))
        if name in self.fail:
            raise RuntimeError("bridge died")
        return {"ok": True}

    def move(self, x, theta):
        return self._run("move", x, theta)

    def posture(self, name):
        return self._run("posture", name)

    def say(self, text):
        return self._run("say", text)

    def eyes(self, color):
        return self._run("eyes", color)

    def stop(self):
        return self._run("stop")

    def sense(self):
        self.calls.append(("sense",))
        return {"ok": True, "sonar_left": 1.0, "sonar_right": 1.0, "moving": False,
                "posture": "Stand"}

    def refuse(self):
        self.calls.append(("refuse",))
        return {"ok": False, "error": "no way"}

    def close(self):
        self.closed = True


def make(client=None, **kw):
    client = client or SlowClient()
    kw.setdefault("sense_interval", None)
    return client, AsyncClient(client, **kw)


def test_caller_never_blocks_on_slow_commands():
    client = SlowClient({"posture": 0.5, "say": 0.3, "eyes": 0.3})
    client, ac = make(client)
    t0 = time.perf_counter()
    tickets = []
    for i in range(30):
        tickets.append(ac.posture("Sit") if i % 3 == 0 else ac.say("hola") if i % 3 == 1
                       else ac.eyes("red"))
    elapsed = time.perf_counter() - t0
    assert elapsed < 30 / 20.0  # budget: > 20 calls per second
    assert all(isinstance(t, Ticket) and t.get("ok") and t.get("queued") for t in tickets)
    assert not tickets[-1].done
    ac.stop()
    ac.close()


def test_fifo_order_and_ticket_results():
    client, ac = make()
    tickets = [ac.move(0.1, 0.0), ac.say("a"), ac.eyes("red"), ac.posture("Sit")]
    assert tickets[-1].wait(2)
    assert [c[0] for c in client.calls] == ["move", "say", "eyes", "posture"]
    assert all(t.succeeded for t in tickets) and tickets[0].response == {"ok": True}
    ac.close()


def test_stop_drops_queued_commands_and_goes_first_even_when_the_worker_is_busy():
    client = SlowClient()
    client.hold = "posture"
    client, ac = make(client)
    busy = ac.posture("Sit")
    assert client.started.wait(2)
    queued = [ac.say("x"), ac.eyes("red"), ac.move(0.1, 0.0)]
    stopper = ac.stop()
    again = ac.stop()
    late = ac.say("after")
    assert again is stopper  # no pile of stops
    assert all(t.done and t.cancelled for t in queued)
    assert ac.pending() == 2  # stop + the late command
    client.release.set()
    assert late.wait(2) and busy.wait(2)
    assert [c[0] for c in client.calls] == ["posture", "stop", "say"]
    assert not busy.cancelled  # already started: it runs to completion
    ac.close()


def test_stop_returns_instantly_while_the_worker_is_stuck():
    client = SlowClient({"posture": 1.0})
    client, ac = make(client)
    ac.posture("Sit")
    assert client.started.wait(2)
    t0 = time.perf_counter()
    ac.stop()
    assert time.perf_counter() - t0 < 0.05
    ac.close()


def test_worker_survives_exceptions_and_records_last_error():
    seen = []
    client, ac = make(SlowClient(fail={"say"}),
                      on_result=lambda *a: seen.append(a))
    bad = ac.say("boom")
    refused = ac.refuse()
    good = ac.eyes("blue")
    assert good.wait(2)
    assert bad.error == "say: bridge died" and not bad.succeeded
    assert refused.error == "refuse: no way"
    assert ac.last_error is None  # the latest command succeeded
    assert [(s[0], bool(s[2])) for s in seen] == [("say", True), ("refuse", True), ("eyes", False)]
    assert all(isinstance(s[3], float) for s in seen)
    bad2 = ac.say("again")
    assert bad2.wait(2) and ac.last_error == "say: bridge died"
    ac.close()


def test_callback_errors_do_not_kill_the_worker():
    def boom(*a):
        raise ValueError("bad callback")
    client, ac = make(on_result=boom)
    first = ac.eyes("red")
    first.add_done_callback(boom)
    second = ac.eyes("blue")
    assert second.wait(2) and second.succeeded
    ac.close()


def test_unknown_methods_raise_attribute_error():
    _, ac = make()
    assert not hasattr(ac, "fly")
    assert hasattr(ac, "move")
    ac.close()


def test_sense_is_polled_in_the_background_and_cached_with_its_age():
    client, ac = make(sense_interval=0.02)
    deadline = time.time() + 2
    while ac.latest_sense()[0] is None and time.time() < deadline:
        time.sleep(0.01)
    reading, age = ac.latest_sense()
    assert reading["sonar_left"] == 1.0 and 0.0 <= age < 1.0
    time.sleep(0.15)
    assert sum(1 for c in client.calls if c[0] == "sense") >= 3  # keeps polling
    ac.close()


def test_latest_sense_age_comes_from_the_injected_clock():
    now = [100.0]
    client, ac = make(clock=lambda: now[0])
    assert ac.latest_sense() == (None, None)
    assert ac.sense().wait(2)
    now[0] = 100.4
    reading, age = ac.latest_sense()
    assert reading["ok"] and abs(age - 0.4) < 1e-9
    now[0] = 102.0
    assert abs(ac.latest_sense()[1] - 2.0) < 1e-9  # a silent worker means an ever older reading
    ac.close()


def test_failed_poll_is_cached_as_not_ok_and_never_clears_last_error():
    client = SlowClient()
    client.sense = lambda: (_ for _ in ()).throw(RuntimeError("down"))
    client, ac = make(client, sense_interval=0.01)
    bad = ac.say("x")  # ok
    assert bad.wait(2)
    deadline = time.time() + 2
    while ac.latest_sense()[0] is None and time.time() < deadline:
        time.sleep(0.01)
    assert ac.latest_sense()[0]["ok"] is False
    assert ac.last_error is None
    ac.close()


def test_close_drains_then_joins_and_closes_the_wrapped_client():
    client = SlowClient({"eyes": 0.05})
    client, ac = make(client)
    last = ac.eyes("blue")
    ac.close()
    assert last.done and last.succeeded  # queued work was drained
    assert not ac._thread.is_alive() and client.closed
    after = ac.say("late")
    assert after.cancelled  # nothing runs after close
    ac.close()  # idempotent


def test_close_with_a_stuck_worker_returns_within_the_timeout():
    client = SlowClient()
    client.hold = "posture"
    client, ac = make(client, join_timeout=0.2)
    ac.posture("Sit")
    assert client.started.wait(2)
    t0 = time.perf_counter()
    ac.close()
    assert time.perf_counter() - t0 < 1.0
    client.release.set()


def test_second_stop_also_drops_commands_queued_after_the_first_stop():
    # Regression: a shared (already queued) stop used to return early and leave later commands alive.
    client = SlowClient()
    client.hold = "posture"
    client, ac = make(client)
    busy = ac.posture("Sit")
    assert client.started.wait(2)
    first = ac.stop()
    eyes, move = ac.eyes("red"), ac.move(0.2, 0.0)
    second = ac.stop()
    assert second is first
    assert eyes.done and eyes.cancelled and move.done and move.cancelled
    assert ac.pending() == 1  # only the stop
    client.release.set()
    assert busy.wait(2) and first.wait(2)
    assert [c[0] for c in client.calls] == ["posture", "stop"]
    ac.close()
