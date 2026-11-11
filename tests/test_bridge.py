"""Tests for the Python 2.7 NAO bridge, driven through NaoClient in --mock mode.

They run against the REAL ``C:\\Python27\\python.exe`` (skipped if missing). The
bridge is never started in real mode here: no robot connection is made.
"""
import json
import os
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from brain.nao_client import NaoBridgeError, NaoClient  # noqa: E402

PYTHON27 = r"C:\Python27\python.exe"

pytestmark = pytest.mark.skipif(
    not os.path.isfile(PYTHON27), reason="Python 2.7 not installed"
)


@pytest.fixture
def client():
    with NaoClient(mock=True, python27=PYTHON27, timeout=15) as c:
        yield c


def _wait_until_stopped(client, limit=4.0):
    deadline = time.time() + limit
    while time.time() < deadline:
        if not client.sense()["moving"]:
            return True
        time.sleep(0.05)
    return False


# -- ping / sense / simulated -------------------------------------------------
def test_ping_reports_mock(client):
    r = client.ping()
    assert r["ok"] is True and r["mock"] is True


def test_sense_defaults(client):
    r = client.sense()
    assert r["ok"] is True
    assert r["sonar_left"] == pytest.approx(1.5)
    assert r["sonar_right"] == pytest.approx(1.5)
    assert r["moving"] is False
    assert r["posture"] == "StandInit"


def test_every_mock_response_is_simulated(client):
    replies = [
        client.ping(),
        client.sense(),
        client.move(0.1, 0.0),
        client.stop(),
        client.wave(),
        client.posture("Sit"),
        client.say("Hola"),
        client.eyes("green"),
        client.set_mock(sonar_left=0.5),
        client.posture("Bogus"),  # error responses too
    ]
    assert all(r["simulated"] is True for r in replies)
    assert all("ok" in r for r in replies)


# -- move ---------------------------------------------------------------------
def test_move_ok_returns_values(client):
    r = client.move(0.2, 0.5)
    assert r["ok"] is True
    assert r["x"] == pytest.approx(0.2)
    assert r["theta"] == pytest.approx(0.5)
    assert r["clamped"] is False


@pytest.mark.parametrize(
    "x, theta, ex, et",
    [
        (5.0, 0.0, 0.40, 0.0),
        (-0.2, 0.0, 0.0, 0.0),
        (0.1, 9.0, 0.1, 1.6),
        (0.1, -9.0, 0.1, -1.6),
    ],
)
def test_move_is_clamped(client, x, theta, ex, et):
    r = client.move(x, theta)
    assert r["ok"] is True
    assert r["x"] == pytest.approx(ex)
    assert r["theta"] == pytest.approx(et)
    assert r["clamped"] is True


def test_move_rejects_non_numbers(client):
    for raw in ['{"cmd":"move"}', '{"cmd":"move","x":"a","theta":0}',
                '{"cmd":"move","x":true,"theta":0}']:
        client._write_line(raw)
        reply = json.loads(client._read_line())
        assert reply["ok"] is False and "error" in reply


@pytest.mark.parametrize("posture", ["Sit", "Crouch", "Unknown", "LyingBack"])
def test_move_refused_when_not_standing(client, posture):
    client.set_mock(posture=posture)
    r = client.move(0.1, 0.0)
    assert r["ok"] is False
    assert "stand" in r["error"].lower()
    assert client.sense()["moving"] is False


@pytest.mark.parametrize("posture", ["Stand", "StandInit", "StandZero"])
def test_move_allowed_when_standing(client, posture):
    client.set_mock(posture=posture)
    assert client.move(0.05, 0.0)["ok"] is True


def test_move_refused_after_sitting_down(client):
    assert client.posture("Sit")["ok"] is True
    assert client.sense()["posture"] == "Sit"
    assert client.move(0.1, 0.0)["ok"] is False
    assert client.posture("StandInit")["ok"] is True
    assert client.move(0.1, 0.0)["ok"] is True


def test_moving_flips_true_then_false(client):
    assert client.move(0.05, 0.0)["ok"] is True  # simulated ~0.5 s
    assert client.sense()["moving"] is True
    assert _wait_until_stopped(client)


def test_stop_ends_motion(client):
    client.move(0.3, 0.0)  # simulated ~3 s
    assert client.sense()["moving"] is True
    assert client.stop()["ok"] is True
    assert client.sense()["moving"] is False


# -- wave ---------------------------------------------------------------------
def test_wave_ok_when_idle(client):
    r = client.wave()
    assert r["ok"] is True
    assert 0 < r["duration"] <= 4.0
    assert client.sense()["moving"] is False


def test_wave_refused_while_moving(client):
    client.move(0.3, 0.0)
    r = client.wave()
    assert r["ok"] is False and "moving" in r["error"]
    client.stop()
    assert client.wave()["ok"] is True


def test_wave_refused_when_forced_moving(client):
    client.set_mock(moving=True)
    assert client.wave()["ok"] is False
    client.set_mock(moving=False)
    assert client.wave()["ok"] is True


def test_wave_refused_when_sitting(client):
    client.posture("Sit")
    assert client.wave()["ok"] is False


# -- posture ------------------------------------------------------------------
def test_posture_whitelist(client):
    assert client.posture("Sit")["ok"] is True
    assert client.posture("StandInit")["ok"] is True
    for bad in ["Stand", "Crouch", "LyingBack", "", "sit", None, 3]:
        r = client.posture(bad)
        assert r["ok"] is False, bad
    assert client.sense()["posture"] == "StandInit"


def test_posture_refused_while_moving(client):
    client.move(0.3, 0.0)
    assert client.posture("Sit")["ok"] is False
    assert client.sense()["posture"] == "StandInit"


# -- say / eyes ---------------------------------------------------------------
def test_say_ok(client):
    assert client.say("Hola")["ok"] is True
    assert client.say("Hay un obst\u00e1culo, no puedo avanzar")["ok"] is True


def test_say_requires_text(client):
    client._write_line('{"cmd":"say"}')
    assert json.loads(client._read_line())["ok"] is False


@pytest.mark.parametrize("color", ["green", "yellow", "red", "blue", "white"])
def test_eyes_known_colors(client, color):
    assert client.eyes(color)["ok"] is True


@pytest.mark.parametrize("color", ["bogus_color", "", "GREEN", None, 5])
def test_eyes_rejects_unknown_colors(client, color):
    r = client.eyes(color)
    assert r["ok"] is False and "color" in r["error"]


# -- set_mock -----------------------------------------------------------------
def test_set_mock_then_sense(client):
    assert client.set_mock(sonar_left=0.2, sonar_right=0.9, posture="Sit")["ok"] is True
    r = client.sense()
    assert r["sonar_left"] == pytest.approx(0.2)
    assert r["sonar_right"] == pytest.approx(0.9)
    assert r["posture"] == "Sit"


def test_set_mock_partial_update_keeps_the_rest(client):
    client.set_mock(sonar_left=0.4)
    r = client.sense()
    assert r["sonar_left"] == pytest.approx(0.4)
    assert r["sonar_right"] == pytest.approx(1.5)


def test_set_mock_moving_flag(client):
    client.set_mock(moving=True)
    assert client.sense()["moving"] is True
    client.set_mock(moving=False)
    assert client.sense()["moving"] is False


def test_set_mock_invalid_is_all_or_nothing(client):
    r = client.set_mock(sonar_left=0.1, moving="yes")
    assert r["ok"] is False
    assert client.sense()["sonar_left"] == pytest.approx(1.5)


# -- robustness ---------------------------------------------------------------
def test_old_act_command_is_gone(client):
    r = client._call(cmd="act", action="ADVANCE")
    assert r["ok"] is False and "unknown cmd" in r["error"]


def test_unknown_command(client):
    r = client._call(cmd="bogus_cmd")
    assert r["ok"] is False and "unknown cmd" in r["error"]
    assert client.ping()["ok"] is True


def test_dance_command(client):
    r = client.dance()
    assert r["ok"] is True
    assert "duration" in r and r["duration"] > 0.0


def test_bad_input_does_not_kill_bridge(client):
    for raw in [
        "this is not json",
        "[1, 2]",
        '{"cmd": "nope"}',
        '{"no_cmd": 1}',
        '{"cmd": "move"}',
        '{"cmd": "say"}',
        '{"cmd": "eyes"}',
        '{"cmd": "posture"}',
    ]:
        client._write_line(raw)
        reply = json.loads(client._read_line())
        assert reply["ok"] is False and "error" in reply
        assert reply["simulated"] is True
    assert client.ping()["ok"] is True


# -- lifecycle ----------------------------------------------------------------
def test_close_terminates_process():
    c = NaoClient(mock=True, python27=PYTHON27, timeout=15)
    c.ping()
    proc = c._proc
    c.close()
    assert proc.poll() is not None


def test_dead_bridge_raises_clear_error():
    c = NaoClient(mock=True, python27=PYTHON27, timeout=15)
    c.ping()
    c._proc.kill()
    c._proc.wait()
    with pytest.raises(NaoBridgeError):
        c.ping()
    c.close()


def test_client_is_thread_safe(client):
    import threading

    errors = []

    def worker():
        try:
            for _ in range(15):
                assert client.sense()["ok"] is True
                assert client.ping()["ok"] is True
        except Exception as exc:  # noqa: BLE001 - collected and asserted below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


# -- protocol channel stays clean of native (C-level) log output ---------------
def test_native_stdout_noise_does_not_reach_the_protocol(monkeypatch):
    # naoqi's C++ logger writes "[I] qimessaging..." lines straight to file
    # descriptor 1. NAO_BRIDGE_TEST_NOISE makes the mock bridge do the same.
    monkeypatch.setenv("NAO_BRIDGE_TEST_NOISE", "1")
    with NaoClient(mock=True, python27=PYTHON27, timeout=15) as c:
        assert c.ping()["ok"] is True
        assert c.sense()["ok"] is True


def test_client_skips_non_json_lines(tmp_path, monkeypatch):
    import brain.nao_client as nc

    fake = tmp_path / "fake_bridge.py"
    fake.write_text(
        "import sys, json\n"
        "for line in sys.stdin:\n"
        "    print('[I] 1234 qimessaging.session: noise before the reply', flush=True)\n"
        "    print(json.dumps({'ok': True, 'echo': json.loads(line)['cmd']}), flush=True)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(nc, "BRIDGE_SCRIPT", str(fake))
    c = nc.NaoClient(mock=True, python27=sys.executable, timeout=10)
    try:
        assert c.ping() == {"ok": True, "echo": "ping"}
        assert c.sense() == {"ok": True, "echo": "sense"}
    finally:
        c.close()
