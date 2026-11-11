"""Tests for the NAO bridge (mock mode), client, choreographies and controller.

SAFETY: the bridge is only ever started with ``--mock``; no robot is contacted.
"""
import csv
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dance import STYLES  # noqa: E402
from dance.config import BPM_MAX, BPM_MIN  # noqa: E402
from dance.robot import choreographies as ch  # noqa: E402
from dance.robot.controller import DanceController  # noqa: E402
from dance.robot.nao_client import NaoBridgeError, NaoClient  # noqa: E402

PYTHON27 = r"C:\Python27\python.exe"
BRIDGE = ROOT / "bridge" / "nao_bridge.py"

pytestmark = pytest.mark.skipif(
    not os.path.isfile(PYTHON27), reason="Python 2.7 not installed at C:\\Python27"
)


class RawBridge:
    """Speaks raw JSON lines to the mock bridge (independent of NaoClient)."""

    def __init__(self):
        self.proc = subprocess.Popen(
            [PYTHON27, str(BRIDGE), "--mock"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.lines = queue.Queue()
        self.stderr = []
        threading.Thread(target=self._pump, args=(self.proc.stdout, self.lines), daemon=True).start()
        threading.Thread(target=self._pump_err, daemon=True).start()

    @staticmethod
    def _pump(stream, q):
        for line in stream:
            q.put(line)
        q.put(None)

    def _pump_err(self):
        for line in self.proc.stderr:
            self.stderr.append(line.decode("utf-8", "replace"))

    def send_raw(self, text):
        self.proc.stdin.write(text.encode("utf-8") + b"\r\n")  # CRLF on purpose
        self.proc.stdin.flush()
        line = self.lines.get(timeout=10)
        assert line is not None, "bridge died; stderr=%r" % self.stderr
        return json.loads(line.decode("utf-8"))

    def call(self, **req):
        return self.send_raw(json.dumps(req))

    def close(self):
        try:
            self.proc.stdin.write(b'{"cmd":"quit"}\n')
            self.proc.stdin.flush()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
        for s in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try:
                s.close()
            except OSError:
                pass  # the process already exited (e.g. after an explicit quit)


@pytest.fixture
def bridge():
    b = RawBridge()
    yield b
    b.close()


@pytest.fixture
def client():
    c = NaoClient(mock=True, timeout=10.0)
    yield c
    c.close()


SHORT = {
    "names": ["RShoulderPitch", "HeadYaw"],
    "angles": [[1.0, 0.8], [0.1, -0.1]],
    "times": [[0.4, 0.8], [0.4, 0.8]],
}


# --------------------------------------------------------------------------
# Bridge protocol
# --------------------------------------------------------------------------
def test_ping_is_mock_and_simulated(bridge):
    r = bridge.call(cmd="ping")
    assert r["ok"] is True and r["mock"] is True and r["simulated"] is True


def test_every_response_has_ok_and_simulated(bridge):
    reqs = [
        {"cmd": "ping"},
        {"cmd": "get_limits", "names": ["HeadYaw"]},
        {"cmd": "play_keyframes", **SHORT},
        {"cmd": "is_running", "task_id": 1},
        {"cmd": "stop_motion"},
        {"cmd": "posture", "name": "StandInit", "speed": 0.5},
        {"cmd": "say", "text": "hola"},
        {"cmd": "set_eyes", "rgb": [255, 0, 0]},
        {"cmd": "nonsense"},
    ]
    for req in reqs:
        r = bridge.call(**req)
        assert "ok" in r and r["simulated"] is True, (req, r)


def test_get_limits_returns_pairs_and_rejects_unknown(bridge):
    r = bridge.call(cmd="get_limits", names=["HeadYaw", "RShoulderRoll"])
    assert r["ok"] is True
    lo, hi = r["limits"]["HeadYaw"]
    assert lo < 0 < hi
    assert r["limits"]["RShoulderRoll"][1] < 0.4  # right roll mostly negative
    bad = bridge.call(cmd="get_limits", names=["NotAJoint"])
    assert bad["ok"] is False and "NotAJoint" in bad["error"]


def test_play_keyframes_is_nonblocking_and_is_running_flips(bridge):
    t0 = time.time()
    r = bridge.call(cmd="play_keyframes", **SHORT)  # lasts 0.8 s
    assert r["ok"] is True and "task_id" in r
    assert time.time() - t0 < 0.6  # returned before the motion ended
    tid = r["task_id"]
    assert bridge.call(cmd="is_running", task_id=tid)["running"] is True
    time.sleep(1.1)
    assert bridge.call(cmd="is_running", task_id=tid)["running"] is False


def test_stop_motion_mid_run(bridge):
    long_req = {
        "names": ["RShoulderPitch"],
        "angles": [[1.0, 0.8, 1.0]],
        "times": [[1.0, 2.0, 3.0]],
    }
    tid = bridge.call(cmd="play_keyframes", **long_req)["task_id"]
    assert bridge.call(cmd="is_running", task_id=tid)["running"] is True
    r = bridge.call(cmd="stop_motion")
    assert r["ok"] is True
    assert bridge.call(cmd="is_running", task_id=tid)["running"] is False


def test_play_keyframes_rejects_unsafe_requests(bridge):
    legs = dict(SHORT, names=["LHipPitch", "HeadYaw"])
    assert bridge.call(cmd="play_keyframes", **legs)["ok"] is False
    out_of_limits = dict(SHORT, angles=[[1.0, 0.8], [5.0, 0.0]])
    r = bridge.call(cmd="play_keyframes", **out_of_limits)
    assert r["ok"] is False and "HeadYaw" in r["error"]
    not_increasing = dict(SHORT, times=[[0.8, 0.4], [0.4, 0.8]])
    assert bridge.call(cmd="play_keyframes", **not_increasing)["ok"] is False
    too_fast = {"names": ["RShoulderPitch"], "angles": [[1.5, -1.0]], "times": [[1.0, 1.2]]}
    assert bridge.call(cmd="play_keyframes", **too_fast)["ok"] is False
    ragged = dict(SHORT, times=[[0.4], [0.4, 0.8]])
    assert bridge.call(cmd="play_keyframes", **ragged)["ok"] is False


def test_posture_whitelist_say_and_eyes(bridge):
    assert bridge.call(cmd="posture", name="StandInit", speed=0.5)["ok"] is True
    assert bridge.call(cmd="posture", name="Rest", speed=0.5)["ok"] is False
    assert bridge.call(cmd="say", text=u"canci\u00f3n \u00f1and\u00fa")["ok"] is True
    assert bridge.call(cmd="say")["ok"] is False
    assert bridge.call(cmd="set_eyes", rgb=0x00FF00)["ok"] is True
    assert bridge.call(cmd="set_eyes", rgb="red")["ok"] is False


def test_unknown_command_and_bad_json_survive(bridge):
    r = bridge.call(cmd="launch_missiles")
    assert r["ok"] is False and "unknown" in r["error"]
    r = bridge.send_raw("{this is not json")
    assert r["ok"] is False and r["simulated"] is True
    r = bridge.send_raw("[1, 2, 3]")
    assert r["ok"] is False
    assert bridge.call(cmd="ping")["ok"] is True  # still alive


def test_mock_logs_timestamped_events_to_stderr(bridge):
    bridge.call(cmd="ping")
    bridge.call(cmd="play_keyframes", **SHORT)
    time.sleep(0.3)
    text = "".join(bridge.stderr)
    assert "MOCK" in text and "play_keyframes" in text


def test_quit_exits_process(bridge):
    assert bridge.call(cmd="quit")["ok"] is True
    assert bridge.proc.wait(timeout=5) == 0


# --------------------------------------------------------------------------
# NaoClient
# --------------------------------------------------------------------------
def test_client_roundtrip(client):
    assert client.ping()["mock"] is True
    assert client.simulated is True
    limits = client.get_limits(["HeadYaw", "HeadPitch"])
    assert set(limits) == {"HeadYaw", "HeadPitch"}
    tid = client.play_keyframes(**SHORT)
    assert isinstance(tid, int)
    assert client.is_running(tid) is True
    client.stop_motion()
    assert client.is_running(tid) is False
    client.posture("StandInit", 0.5)
    client.say("hola")
    client.set_eyes((0, 0, 255))


def test_client_raises_on_bridge_error(client):
    with pytest.raises(NaoBridgeError):
        client.get_limits(["NotAJoint"])


def test_client_context_manager_and_dead_bridge():
    with NaoClient(mock=True, timeout=5.0) as c:
        assert c.ping()["ok"]
    with pytest.raises(NaoBridgeError):
        c.ping()


def test_client_missing_python_raises():
    with pytest.raises(NaoBridgeError):
        NaoClient(mock=True, python27=r"C:\definitely\missing\python.exe")


# --------------------------------------------------------------------------
# Choreographies
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def limits():
    with NaoClient(mock=True, timeout=10.0) as c:
        return c.get_limits(list(ch.JOINTS))


def _step_count(style, bpm, duration=20.0):
    names, angles, times = ch.build_keyframes(style, bpm, duration)
    return len(times[0]) / float(times[0][-1])


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("bpm", [BPM_MIN, 90.0, 120.0, 150.0, BPM_MAX])
def test_build_keyframes_format_and_limits(style, bpm, limits):
    names, angles, times = ch.build_keyframes(style, bpm, 8.0)
    assert names == list(ch.JOINTS)
    assert len(angles) == len(times) == len(names)
    for ang, tms in zip(angles, times):
        assert len(ang) == len(tms) >= 2
        assert all(b > a for a, b in zip(tms, tms[1:]))
        assert tms[0] > 0
    assert times[0][-1] >= 8.0
    ch.validate_keyframes(names, angles, limits)  # does not raise
    for name, ang in zip(names, angles):
        lo, hi = limits[name]
        assert all(lo <= a <= hi for a in ang)


def test_only_arms_and_head():
    for name in ch.JOINTS:
        assert name.startswith(("Head", "LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist"))


@pytest.mark.parametrize("style", STYLES)
@pytest.mark.parametrize("bpm", [60.0, 120.0, 180.0])
def test_velocity_cap_and_min_step_time(style, bpm):
    names, angles, times = ch.build_keyframes(style, bpm, 12.0)
    for name, ang, tms in zip(names, angles, times):
        prev_a, prev_t = ch.NEUTRAL_POSE[name], 0.0
        for i, (a, t) in enumerate(zip(ang, tms)):
            dt = t - prev_t
            if i > 0:  # keyframe 0 is the lead-in transition (>= 0.8 s)
                assert dt >= ch.MIN_STEP_SECONDS - 1e-9
            assert abs(a - prev_a) / dt <= ch.MAX_JOINT_VELOCITY + 1e-9
            prev_a, prev_t = a, t


def test_first_transition_takes_at_least_800ms():
    for style in STYLES:
        names, angles, times = ch.build_keyframes(style, 180.0, 4.0)
        assert times[0][0] >= 0.8 - 1e-9


def test_start_pose_sets_first_transition():
    far = {n: ch.NEUTRAL_POSE[n] for n in ch.JOINTS}
    far["RShoulderPitch"] = -1.0  # 2.4 rad from neutral: needs > 0.8 s
    names, angles, times = ch.build_keyframes("SOFT", 120.0, 4.0, start_pose=far)
    first_target = angles[names.index("RShoulderPitch")][0]
    assert abs(first_target - (-1.0)) / times[0][0] <= ch.MAX_JOINT_VELOCITY + 1e-9


def test_energetic_has_more_steps_per_second_than_soft():
    for bpm in (80.0, 100.0, 120.0, 150.0):
        assert _step_count("ENERGETIC", bpm) > _step_count("SOFT", bpm), bpm
        assert _step_count("ENERGETIC", bpm) >= _step_count("MODERATE", bpm), bpm
        assert _step_count("MODERATE", bpm) > _step_count("SOFT", bpm), bpm


def test_tempo_scales_timing():
    slow = ch.build_keyframes("MODERATE", 80.0, 12.0)[2][0]
    fast = ch.build_keyframes("MODERATE", 140.0, 12.0)[2][0]
    assert len(fast) > len(slow)


def test_bpm_folding():
    assert ch.fold_bpm(240.0) == pytest.approx(120.0)
    assert ch.fold_bpm(40.0) == pytest.approx(80.0)
    assert ch.fold_bpm(100.0) == 100.0
    for bad in (0.0, -5.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            ch.fold_bpm(bad)
    a = ch.build_keyframes("MODERATE", 240.0, 6.0)
    b = ch.build_keyframes("MODERATE", 120.0, 6.0)
    assert a == b


def test_unknown_style_and_bad_duration():
    with pytest.raises(ValueError):
        ch.build_keyframes("WALTZ", 100.0, 4.0)
    with pytest.raises(ValueError):
        ch.build_keyframes("SOFT", 100.0, 0.0)


def test_validate_keyframes_lists_all_offenders(limits):
    names = ["HeadYaw", "RShoulderPitch"]
    angles = [[0.0, 9.0], [-9.0, 0.0]]
    with pytest.raises(ValueError) as exc:
        ch.validate_keyframes(names, angles, limits)
    msg = str(exc.value)
    assert "HeadYaw" in msg and "RShoulderPitch" in msg and "9.0" in msg
    with pytest.raises(ValueError):
        ch.validate_keyframes(["Mystery"], [[0.0]], limits)


def test_final_pose_roundtrip():
    names, angles, times = ch.build_keyframes("ENERGETIC", 120.0, 3.0)
    pose = ch.final_pose(names, angles)
    assert set(pose) == set(names)
    assert pose["HeadYaw"] == angles[names.index("HeadYaw")][-1]


# --------------------------------------------------------------------------
# DanceController
# --------------------------------------------------------------------------
def test_controller_runs_segment_and_logs(client, tmp_path):
    log = tmp_path / "run.csv"
    ctl = DanceController(client, log_path=str(log))
    t0 = time.time()
    done = ctl.run_segment("MODERATE", 120.0, 1.5)
    elapsed = time.time() - t0
    assert done is True
    assert 1.5 <= elapsed < 6.0  # mock simulates the real duration
    kinds = [e["event"] for e in ctl.events]
    assert kinds[0] == "segment_start" and kinds[-1] == "segment_end"
    start = ctl.events[0]
    assert start["style"] == "MODERATE" and start["bpm"] == 120.0
    assert start["simulated"] is True and start["task_id"] is not None
    rows = list(csv.DictReader(open(log, newline="", encoding="utf-8")))
    assert [r["event"] for r in rows] == kinds
    assert set(rows[0]) >= {"timestamp", "event", "style", "bpm", "duration", "task_id", "simulated", "error"}


def test_controller_chains_segments_from_last_pose(client):
    ctl = DanceController(client)
    assert ctl.run_segment("SOFT", 100.0, 1.0) is True
    pose = ctl.last_pose
    assert pose is not None and set(pose) == set(ch.JOINTS)
    assert ctl.run_segment("ENERGETIC", 100.0, 1.0) is True


def test_controller_stop_during_run(client):
    ctl = DanceController(client)
    result = {}

    def worker():
        result["done"] = ctl.run_segment("SOFT", 100.0, 8.0)

    th = threading.Thread(target=worker)
    th.start()
    time.sleep(0.8)
    t0 = time.time()
    ctl.stop()
    th.join(timeout=5)
    assert not th.is_alive()
    assert time.time() - t0 < 3.0
    assert result["done"] is False
    kinds = [e["event"] for e in ctl.events]
    assert "stop" in kinds and "segment_end" not in kinds
    assert ctl.stopped is True
    # Further segments are refused until the stop flag is cleared.
    assert ctl.run_segment("SOFT", 100.0, 1.0) is False
    ctl.clear_stop()
    assert ctl.run_segment("SOFT", 100.0, 1.0) is True


def test_controller_logs_errors_and_reraises(client):
    ctl = DanceController(client)
    with pytest.raises(ValueError):
        ctl.run_segment("WALTZ", 100.0, 2.0)
    assert ctl.events[-1]["event"] == "error" and "WALTZ" in ctl.events[-1]["error"]


def test_controller_does_not_import_ml_modules():
    src = (ROOT / "dance" / "robot" / "controller.py").read_text(encoding="utf-8")
    assert "dance.model" not in src and "dance.audio" not in src
