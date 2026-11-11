"""Tests for the end-to-end pipeline (``dance.main``).

SAFETY: no robot, no bridge and no audio device is used here. The mock path is
exercised with a stub predictor, and ``NaoClient`` is patched to fail if it is
ever constructed.
"""
import csv
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dance import main as dance_main  # noqa: E402
from dance.model.infer import SegmentPrediction  # noqa: E402
from dance.robot.choreographies import build_keyframes  # noqa: E402


# ----------------------------------------------------------------- helpers


def _seg(style, conf, i=0, bpm=110.0, features=None):
    return SegmentPrediction(
        start_s=i * 8.0,
        end_s=(i + 1) * 8.0,
        style=style,
        confidence=conf,
        bpm=bpm,
        features=features if features is not None else {"bpm": bpm, "rms_mean": 0.1, "onset_rate": 2.0},
    )


class StubPredictor:
    """Duck-typed ``DancePredictor``: returns fixed predictions, no audio."""

    def __init__(self, preds, model_name="stub_model", segment_seconds=8.0):
        self.preds = list(preds)
        self.model_name = model_name
        self.segment_seconds = segment_seconds
        self.paths = []

    def analyze_file(self, path):
        self.paths.append(path)
        return list(self.preds)


class FakeController:
    def __init__(self, results=None, error=None, stopped=False):
        self.results = list(results) if results is not None else None
        self.error = error
        self.stopped = stopped
        self.calls = []
        self.stop_calls = 0

    def run_segment(self, style, bpm, duration_s):
        self.calls.append((style, bpm, duration_s))
        if self.error is not None:
            raise self.error
        if self.results is not None:
            return self.results.pop(0)
        return True

    def stop(self):
        self.stop_calls += 1
        self.stopped = True


class FakePlayer:
    def __init__(self):
        self.played = []
        self.stops = 0

    def play(self, start_s, end_s):
        self.played.append((start_s, end_s))

    def stop(self):
        self.stops += 1


# --------------------------------------------------------------------- CLI


def test_cli_help_lists_the_two_modes(capsys):
    with pytest.raises(SystemExit) as excinfo:
        dance_main.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "--mock" in out and "--robot" in out and "--song" in out


def test_song_is_required():
    with pytest.raises(SystemExit) as excinfo:
        dance_main.main(["--mock"])
    assert excinfo.value.code == 2


def test_mode_must_be_explicit(capsys):
    assert dance_main.main(["--song", "song.wav"]) == 2
    err = capsys.readouterr().err
    assert "--mock" in err and "--robot" in err


def test_mock_and_robot_are_exclusive(capsys):
    assert dance_main.main(["--song", "song.wav", "--mock", "--robot", "zyra"]) == 2
    assert "mutually exclusive" in capsys.readouterr().err


def test_missing_model_is_reported_without_traceback(tmp_path, capsys):
    code = dance_main.main(["--song", "song.wav", "--mock", "--model", str(tmp_path / "nope.joblib")])
    assert code == 2
    assert "ERROR" in capsys.readouterr().err


# -------------------------------------------------------------------- plan


def test_build_plan_uses_smoothed_styles_and_keeps_features():
    predictor = StubPredictor([
        _seg("SOFT", 0.9, 0),
        _seg("ENERGETIC", 0.2, 1),   # low confidence -> keeps SOFT
        _seg("ENERGETIC", 0.9, 2),
    ])
    plan = dance_main.build_plan(predictor, "song.wav", min_conf=0.5)
    assert [s.style for s in plan.segments] == ["SOFT", "SOFT", "ENERGETIC"]
    assert [s.index for s in plan.segments] == [1, 2, 3]
    assert plan.segments[0].features["rms_mean"] == pytest.approx(0.1)
    assert plan.duration_s == pytest.approx(24.0)
    assert predictor.paths == ["song.wav"]


def test_build_plan_motion_covers_the_segment():
    predictor = StubPredictor([_seg("MODERATE", 0.8, i, bpm=120.0) for i in range(3)])
    plan = dance_main.build_plan(predictor, "song.wav")
    for segment in plan.segments:
        assert segment.steps >= 2
        assert segment.motion_s >= segment.duration_s
        assert segment.duration_s == pytest.approx(8.0)


def test_build_plan_chains_segments_from_the_previous_pose(monkeypatch):
    seen = []

    def fake_build_keyframes(style, bpm, duration_s, start_pose=None):
        seen.append(start_pose)
        return ["HeadYaw"], [[0.1 * len(seen)]], [[1.0]]

    monkeypatch.setattr(dance_main, "build_keyframes", fake_build_keyframes)
    predictor = StubPredictor([_seg("SOFT", 0.9, i) for i in range(3)])
    dance_main.build_plan(predictor, "song.wav")
    assert seen[0] is None
    assert seen[1] == {"HeadYaw": 0.1}
    assert seen[2] == {"HeadYaw": 0.2}


def test_build_plan_without_segments_is_a_clear_error():
    with pytest.raises(ValueError, match="No usable segments"):
        dance_main.build_plan(StubPredictor([]), "tiny.wav")


def test_build_plan_rejects_unknown_style():
    with pytest.raises(ValueError, match="unknown style"):
        dance_main.build_plan(StubPredictor([_seg("TANGO", 0.9)]), "song.wav")


# ------------------------------------------------------------------ report


def test_format_plan_shows_everything_the_simulation_must_show():
    predictor = StubPredictor([
        _seg("SOFT", 0.91, 0, bpm=90.0),
        _seg("ENERGETIC", 0.77, 1, bpm=150.0),
    ])
    plan = dance_main.build_plan(predictor, "cancion.wav")
    text = dance_main.format_plan(plan)
    assert "cancion.wav" in text
    assert "stub_model" in text
    assert "120.0 BPM" in text          # median of 90 and 150
    assert "Features" in text and "rms_mean" in text
    assert "Segments" in text and "Timeline" in text
    assert "SOFT" in text and "ENERGETIC" in text
    assert "0:08.0-0:16.0" in text      # timeline in minutes:seconds


def test_format_plan_all_features_is_wider():
    features = {"bpm": 100.0, "mfcc7_std": 0.5, "zcr_mean": 0.1}
    predictor = StubPredictor([_seg("SOFT", 0.9, features=features)])
    plan = dance_main.build_plan(predictor, "song.wav")
    short = dance_main.format_plan(plan)
    full = dance_main.format_plan(plan, all_features=True)
    assert "mfcc7_std" in full and "zcr_mean" in full
    assert "mfcc7_std" not in short
    assert "all extracted features" in full


def test_format_plan_without_features_does_not_crash():
    predictor = StubPredictor([_seg("SOFT", 0.9, features={})])
    plan = dance_main.build_plan(predictor, "song.wav")
    text = dance_main.format_plan(plan)
    assert "Segments" in text
    assert "Features" not in text


def test_timeline_csv_has_one_row_per_segment(tmp_path):
    predictor = StubPredictor([_seg("SOFT", 0.9, 0), _seg("MODERATE", 0.8, 1)])
    plan = dance_main.build_plan(predictor, "song.wav")
    path = dance_main.write_timeline_csv(plan, tmp_path / "log.csv", simulated=True)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 2
    assert list(rows[0]) == dance_main.TIMELINE_FIELDS
    assert rows[0]["style"] == "SOFT"
    assert rows[0]["simulated"] == "True"
    assert float(rows[1]["start_s"]) == pytest.approx(8.0)


def test_log_path_is_timestamped_and_tagged(tmp_path):
    path = dance_main.log_path_for(tmp_path, "zyra")
    assert path.parent == tmp_path
    assert path.name.endswith("_zyra.csv")
    assert path.name[:8].isdigit()


# -------------------------------------------------------------- mock mode


def test_mock_run_never_constructs_a_robot_client(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("mock mode must never start the NAO bridge")

    monkeypatch.setattr(dance_main, "NaoClient", forbidden)
    monkeypatch.setattr(dance_main, "DancePredictor", lambda path: StubPredictor([_seg("SOFT", 0.9)]))
    code = dance_main.main(["--song", "song.wav", "--mock", "--log-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert dance_main.SIMULATION_NOTICE in out
    assert "no physical movements were executed" in out
    logs = list(tmp_path.glob("*_mock.csv"))
    assert len(logs) == 1


def test_mock_run_reports_unusable_song(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dance_main, "DancePredictor", lambda path: StubPredictor([]))
    code = dance_main.main(["--song", "tiny.wav", "--mock", "--log-dir", str(tmp_path)])
    assert code == 2
    assert "No usable segments" in capsys.readouterr().err


# ------------------------------------------------------------- robot mode


def test_run_robot_dances_every_segment_in_order():
    predictor = StubPredictor([_seg("SOFT", 0.9, 0, bpm=90.0), _seg("ENERGETIC", 0.9, 1, bpm=140.0)])
    plan = dance_main.build_plan(predictor, "song.wav")
    controller, player = FakeController(), FakePlayer()
    completed = dance_main.run_robot(plan, controller, player, print_fn=lambda *a, **k: None)
    assert completed == 2
    assert [c[0] for c in controller.calls] == ["SOFT", "ENERGETIC"]
    assert [c[1] for c in controller.calls] == [90.0, 140.0]
    assert player.played == [(0.0, 8.0), (8.0, 16.0)]
    assert player.stops >= 2


def test_run_robot_stops_after_a_refused_segment():
    predictor = StubPredictor([_seg("SOFT", 0.9, i) for i in range(3)])
    plan = dance_main.build_plan(predictor, "song.wav")
    controller, player = FakeController(results=[True, False]), FakePlayer()
    completed = dance_main.run_robot(plan, controller, player, print_fn=lambda *a, **k: None)
    assert completed == 1
    assert len(controller.calls) == 2          # the third segment is never started
    assert player.stops >= 2


def test_run_robot_does_nothing_when_already_stopped():
    predictor = StubPredictor([_seg("SOFT", 0.9, i) for i in range(2)])
    plan = dance_main.build_plan(predictor, "song.wav")
    controller = FakeController(stopped=True)
    assert dance_main.run_robot(plan, controller, FakePlayer(), print_fn=lambda *a, **k: None) == 0
    assert controller.calls == []


def test_run_robot_stops_the_robot_when_a_segment_raises():
    predictor = StubPredictor([_seg("SOFT", 0.9, i) for i in range(2)])
    plan = dance_main.build_plan(predictor, "song.wav")
    controller, player = FakeController(error=ValueError("joint limit")), FakePlayer()
    with pytest.raises(ValueError, match="joint limit"):
        dance_main.run_robot(plan, controller, player, print_fn=lambda *a, **k: None)
    assert controller.stop_calls == 1          # never leave the robot moving
    assert player.stops >= 1                   # and never leave the audio playing


def test_audio_player_disabled_never_touches_the_sound_device(monkeypatch):
    class FakeSD:
        calls = 0

        def play(self, *a, **k):
            FakeSD.calls += 1

        def stop(self):
            FakeSD.calls += 1

    monkeypatch.setitem(sys.modules, "sounddevice", FakeSD())
    player = dance_main.AudioPlayer([0.0] * 100, 22050, enabled=False)
    player.play(0.0, 1.0)
    player.stop()
    assert FakeSD.calls == 0


# ------------------------------------------------------------------- stop


def test_stop_listener_calls_on_stop_on_q():
    pressed = threading.Event()
    listener = dance_main.StopListener(pressed.set, read_key=lambda timeout: "q", poll_interval=0.01)
    listener.start()
    assert pressed.wait(2.0)
    listener.stop()


def test_stop_listener_ignores_other_keys():
    called = threading.Event()
    listener = dance_main.StopListener(called.set, read_key=lambda timeout: "x", poll_interval=0.01)
    listener.start()
    time.sleep(0.1)
    listener.stop()
    assert not called.is_set()


# ------------------------------------------- real mode wiring (fake client)


class FakeClient:
    def __init__(self, simulated=False):
        self.simulated = simulated
        self.calls = []

    def ping(self):
        self.calls.append("ping")
        return {"robot": "zyra"}

    def posture(self, name="StandInit", speed=0.5):
        self.calls.append(("posture", name, speed))

    def close(self):
        self.calls.append("close")


class StoppingController(FakeController):
    """Stops the run (like the q key) on the first segment."""

    def run_segment(self, style, bpm, duration_s):
        self.calls.append((style, bpm, duration_s))
        self.stop()
        return False


def _wire(monkeypatch, client=None, controller=None):
    """Point the real mode at fakes: no bridge process, no robot, no audio device."""
    client = client or FakeClient()
    controller = controller or FakeController()
    seen = {}
    monkeypatch.setattr(dance_main, "NaoClient", lambda **kwargs: client)
    monkeypatch.setattr(
        dance_main, "DanceController",
        lambda client_, log_path=None: (seen.update(log_path=log_path), controller)[1],
    )
    monkeypatch.setattr(
        dance_main, "DancePredictor",
        lambda path: StubPredictor([_seg("SOFT", 0.9, 0), _seg("MODERATE", 0.9, 1)]),
    )
    monkeypatch.setattr("dance.audio.features.load_audio", lambda path, sr=22050: ([0.0] * 100, 22050))
    return client, controller, seen


def test_real_run_wires_client_controller_and_log(tmp_path, monkeypatch, capsys):
    client, controller, seen = _wire(monkeypatch)
    code = dance_main.main(
        ["--song", "song.wav", "--robot", "zyra", "--no-audio", "--log-dir", str(tmp_path)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert client.calls[0] == "ping"
    assert ("posture", "StandInit", 0.5) in client.calls
    assert client.calls[-1] == "close"
    assert len(controller.calls) == 2
    assert seen["log_path"].name.endswith("_zyra_motion.csv")
    assert seen["log_path"].parent == tmp_path
    timeline = list(tmp_path.glob("*_zyra.csv"))
    assert len(timeline) == 1 and timeline[0] != seen["log_path"]
    assert "Segments completed: 2/2" in out
    assert dance_main.SIMULATION_NOTICE not in out


def test_real_run_returns_130_when_the_user_stops(tmp_path, monkeypatch, capsys):
    controller = StoppingController()
    client, _, _ = _wire(monkeypatch, controller=controller)
    code = dance_main.main(
        ["--song", "song.wav", "--robot", "zyra", "--no-audio", "--log-dir", str(tmp_path)]
    )
    assert code == 130
    assert "stopped before the end" in capsys.readouterr().err
    assert client.calls[-1] == "close"


def test_real_run_reports_bridge_errors_without_traceback(tmp_path, monkeypatch, capsys):
    def broken(**kwargs):
        raise dance_main.NaoBridgeError("bridge died")

    monkeypatch.setattr(dance_main, "NaoClient", broken)
    monkeypatch.setattr(dance_main, "DancePredictor", lambda path: StubPredictor([_seg("SOFT", 0.9)]))
    code = dance_main.main(
        ["--song", "song.wav", "--robot", "zyra", "--no-audio", "--log-dir", str(tmp_path)]
    )
    assert code == 2
    assert "ERROR: bridge died" in capsys.readouterr().err


def test_real_run_stops_the_robot_when_a_segment_fails(tmp_path, monkeypatch, capsys):
    controller = FakeController(error=ValueError("joint limit"))
    client, _, _ = _wire(monkeypatch, controller=controller)
    code = dance_main.main(
        ["--song", "song.wav", "--robot", "zyra", "--no-audio", "--log-dir", str(tmp_path)]
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "ERROR: joint limit" in err
    assert "The robot was stopped" in err
    assert controller.stop_calls == 1
    assert client.calls[-1] == "close"


def test_real_run_warns_when_the_bridge_is_simulated(tmp_path, monkeypatch, capsys):
    client = FakeClient(simulated=True)
    _wire(monkeypatch, client=client)
    dance_main.main(["--song", "song.wav", "--robot", "sim", "--no-audio", "--log-dir", str(tmp_path)])
    assert "SIMULATED robot" in capsys.readouterr().out


def test_real_run_passes_ip_and_port_to_the_bridge(tmp_path, monkeypatch):
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return FakeClient()

    monkeypatch.setattr(dance_main, "NaoClient", factory)
    _wire(monkeypatch)
    monkeypatch.setattr(dance_main, "NaoClient", factory)
    dance_main.main(["--song", "song.wav", "--robot", "zyra", "--ip", "10.0.0.5", "--port", "9559",
                     "--no-audio", "--log-dir", str(tmp_path)])
    assert captured == {"robot": "zyra", "ip": "10.0.0.5", "port": 9559}


# ---------------------------------------------------------------- contracts

def test_main_does_not_import_robot_modules_before_it_needs_them():
    """The ML pipeline must stay usable on a machine without naoqi/2.7."""
    source = Path(dance_main.__file__).read_text(encoding="utf-8")
    assert "import naoqi" not in source
    # The bridge is only launched by NaoClient, never by the analysis code.
    assert "subprocess" not in source


def test_choreography_styles_match_the_model_styles():
    from dance import STYLES

    predictor = StubPredictor([_seg(style, 0.9, i) for i, style in enumerate(STYLES)])
    plan = dance_main.build_plan(predictor, "song.wav")
    for segment in plan.segments:
        names, angles, times = build_keyframes(segment.style, segment.bpm, segment.duration_s)
        assert len(names) == len(angles) == len(times)
