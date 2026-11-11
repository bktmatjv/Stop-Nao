"""Tests for the Python 2.7 NAO bridge, driven through NaoClient in mock mode.

The bridge is never started in real mode here: no robot connection is made.
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from brain.nao_client import NaoBridgeError, NaoClient  # noqa: E402

PYTHON27 = r"C:\Python27\python.exe"

pytestmark = pytest.mark.skipif(
    not os.path.isfile(PYTHON27), reason="Python 2.7 not installed"
)

ACTIONS = [
    "ADVANCE",
    "TURN_LEFT",
    "TURN_RIGHT",
    "STOP",
    "WARN_OBSTACLE",
    "ASK_CLOSER",
    "WAIT",
]


@pytest.fixture
def client():
    with NaoClient(mock=True, python27=PYTHON27, timeout=15) as c:
        yield c


def test_ping_reports_mock(client):
    assert client.ping() == {"ok": True, "mock": True}


def test_sense_defaults(client):
    r = client.sense()
    assert r["ok"] is True
    assert r["sonar_left"] == pytest.approx(1.5)
    assert r["sonar_right"] == pytest.approx(1.5)
    assert r["moving"] is False


def test_set_mock_then_sense(client):
    assert client.set_mock(0.2, 0.9)["ok"] is True
    r = client.sense()
    assert r["sonar_left"] == pytest.approx(0.2)
    assert r["sonar_right"] == pytest.approx(0.9)


@pytest.mark.parametrize("action", ACTIONS)
def test_every_action_ok(client, action):
    assert client.act(action) == {"ok": True, "action": action}


def test_unknown_action_is_error(client):
    r = client.act("DANCE")
    assert r["ok"] is False
    assert "error" in r


def test_say_ok(client):
    assert client.say("Hola") == {"ok": True}
    assert client.say("Hay un obst\u00e1culo")["ok"] is True


def test_bad_input_does_not_kill_bridge(client):
    for raw in [
        "this is not json",
        "[1, 2]",
        '{"cmd": "nope"}',
        '{"no_cmd": 1}',
        '{"cmd": "act"}',
        '{"cmd": "say"}',
    ]:
        client._write_line(raw)
        reply = json.loads(client._read_line())
        assert reply["ok"] is False and "error" in reply
    assert client.ping()["ok"] is True


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
