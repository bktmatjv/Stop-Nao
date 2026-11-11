"""End-to-end pipeline: song -> features -> trained model -> choreography -> NAO.

Two modes, chosen explicitly (there is no silent default, so nobody can mistake
a simulation for a real robot run):

* ``--mock`` (simulation): analyzes the song and prints what the system decided
  (song, BPM, extracted features, predicted style per segment, selected
  choreography and timeline), and writes the timeline to ``runs/``. It never
  starts the bridge and never talks to a robot, and it says so.
* ``--robot zyra|nyx|sim``: same analysis, then plays the song on the laptop and
  runs each segment's choreography through the persistent Python 2.7 bridge
  (``bridge/nao_bridge.py``). Audio is played on the laptop: NAO's microphone is
  not used in the MVP.

Stop mechanism: press ``q`` or Ctrl+C. Both halt the motion, return the robot to
StandInit and end the run. A person must always be next to the robot.

CLI::

    python -m dance.main --song path/to/song.mp3 --mock
    python -m dance.main --song path/to/song.mp3 --robot zyra
"""
from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from dance import STYLES, config
from dance.model.infer import DEFAULT_MODEL_PATH, DancePredictor, smooth_predictions
from dance.robot.choreographies import build_keyframes, final_pose
from dance.robot.controller import DanceController
from dance.robot.nao_client import NaoBridgeError, NaoClient

DEFAULT_LOG_DIR = Path("runs")

#: Features shown by default: one representative per family (tempo, rhythm,
#: loudness, brightness, timbre). ``--all-features`` prints all of them.
DISPLAY_FEATURES = (
    "bpm",
    "beat_strength",
    "onset_rate",
    "rms_mean",
    "spectral_centroid_mean",
    "percussive_ratio",
    "mfcc1_mean",
)

TIMELINE_FIELDS = [
    "index", "start_s", "end_s", "duration_s", "style", "confidence",
    "bpm", "steps", "motion_s", "simulated",
]

SIMULATION_NOTICE = (
    "SIMULATION MODE: no physical movements were executed. This run only shows "
    "what the system would do; use --robot <name> for a real dance."
)


# ---------------------------------------------------------------------- plan


@dataclass(frozen=True)
class SegmentPlan:
    """One segment of the song: what the model predicted and what NAO would do."""

    index: int
    start_s: float
    end_s: float
    style: str
    confidence: float | None
    bpm: float
    steps: int            # keyframes in the choreography
    motion_s: float       # seconds of motion (>= segment duration)
    features: dict = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class RunPlan:
    song: Path
    model_name: str
    segment_seconds: float
    segments: list
    min_conf: float = 0.5

    @property
    def duration_s(self) -> float:
        return self.segments[-1].end_s if self.segments else 0.0

    @property
    def bpm_median(self) -> float:
        values = sorted(s.bpm for s in self.segments)
        if not values:
            return 0.0
        middle = len(values) // 2
        return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2.0

    @property
    def bpm_range(self) -> tuple:
        if not self.segments:
            return (0.0, 0.0)
        bpms = [s.bpm for s in self.segments]
        return (min(bpms), max(bpms))

    def mean_features(self, names=None) -> dict:
        """Mean of each feature over the segments that reported it."""
        names = list(names) if names is not None else sorted(
            {n for s in self.segments for n in s.features}
        )
        out = {}
        for name in names:
            values = [s.features[name] for s in self.segments if name in s.features]
            if values:
                out[name] = sum(values) / len(values)
        return out


def build_plan(predictor, song_path, min_conf: float = 0.5) -> RunPlan:
    """Analyze the song and turn the smoothed predictions into a run plan.

    Raises ``ValueError`` when the song has no usable segments. The choreography
    is built here (chained from segment to segment) so an invalid style or tempo
    fails before anything moves.
    """
    raw = predictor.analyze_file(str(song_path))
    if not raw:
        raise ValueError(
            f"No usable segments in {song_path} (the file may be shorter than "
            f"{config.MIN_SEGMENT_FRACTION * 100:.0f}% of a {predictor.segment_seconds:g}s segment)."
        )
    smoothed = smooth_predictions(raw, min_conf=min_conf)
    segments = []
    pose = None
    for i, (r, s) in enumerate(zip(raw, smoothed), start=1):
        duration = s.end_s - s.start_s
        names, angles, times = build_keyframes(s.style, s.bpm, duration, pose)
        pose = final_pose(names, angles)
        segments.append(
            SegmentPlan(
                index=i,
                start_s=float(s.start_s),
                end_s=float(s.end_s),
                style=s.style,
                confidence=s.confidence,
                bpm=float(s.bpm),
                steps=len(times[0]),
                motion_s=float(times[0][-1]),
                features=dict(r.features or {}),
            )
        )
    return RunPlan(
        song=Path(song_path),
        model_name=str(predictor.model_name),
        segment_seconds=float(predictor.segment_seconds),
        segments=segments,
        min_conf=min_conf,
    )


# ------------------------------------------------------------------- report


def _clock(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    return "%d:%04.1f" % (int(seconds // 60), seconds % 60)


def _confidence(value) -> str:
    return "  n/a" if value is None else "%5.2f" % value


def format_plan(plan: RunPlan, all_features: bool = False) -> str:
    """Human-readable plan: song, model, tempo, features, segments and timeline."""
    low, high = plan.bpm_range
    lines = [
        "Song:      %s" % plan.song,
        "Model:     %s | segment: %gs | confidence threshold: %.2f"
        % (plan.model_name, plan.segment_seconds, plan.min_conf),
        "Duration:  %s (%d segments)" % (_clock(plan.duration_s), len(plan.segments)),
        "Tempo:     %.1f BPM (segments %.1f-%.1f)" % (plan.bpm_median, low, high),
        "Dance:     %s (arms + head only, tempo-scaled)"
        % ", ".join(sorted({s.style for s in plan.segments}, key=STYLES.index)),
    ]

    features = plan.mean_features(None if all_features else DISPLAY_FEATURES)
    if features:
        title = "all extracted features" if all_features else "mean of the key features"
        lines += ["", "Features (%s)" % title]
        width = max(len(n) for n in features)
        for name, value in features.items():
            lines.append("  %-*s %12.4f" % (width, name, value))

    lines += ["", "Segments"]
    header = "  %3s %8s %8s %7s  %-10s %5s %5s %7s" % (
        "#", "start", "end", "bpm", "style", "conf", "steps", "motion",
    )
    lines += [header, "  " + "-" * (len(header) - 2)]
    for s in plan.segments:
        lines.append(
            "  %3d %8.1f %8.1f %7.1f  %-10s %5s %5d %6.1fs"
            % (s.index, s.start_s, s.end_s, s.bpm, s.style, _confidence(s.confidence),
               s.steps, s.motion_s)
        )

    lines += ["", "Timeline"]
    for s in plan.segments:
        lines.append(
            "  %s-%s  %-10s  %3.0f BPM  %d keyframes"
            % (_clock(s.start_s), _clock(s.end_s), s.style, s.bpm, s.steps)
        )
    return "\n".join(lines)


def write_timeline_csv(plan: RunPlan, path, simulated: bool) -> Path:
    """Write the temporal log of the run (one row per segment). Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TIMELINE_FIELDS)
        writer.writeheader()
        for s in plan.segments:
            writer.writerow({
                "index": s.index,
                "start_s": "%.3f" % s.start_s,
                "end_s": "%.3f" % s.end_s,
                "duration_s": "%.3f" % s.duration_s,
                "style": s.style,
                "confidence": "" if s.confidence is None else "%.4f" % s.confidence,
                "bpm": "%.2f" % s.bpm,
                "steps": s.steps,
                "motion_s": "%.3f" % s.motion_s,
                "simulated": bool(simulated),
            })
    return path


def log_path_for(log_dir, tag: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(log_dir) / ("%s_%s.csv" % (stamp, tag))


# -------------------------------------------------------------------- stop


def _default_read_key(timeout: float):
    """Return one key press without waiting for Enter (Windows) or None.

    On systems without ``msvcrt`` this falls back to a line of input, which
    needs Enter; the caller treats any non-empty answer as a stop request.
    """
    try:
        import msvcrt
    except ImportError:  # pragma: no cover - Windows is the supported platform
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            return None
        return (line or " ")[:1]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if msvcrt.kbhit():
            return msvcrt.getwch()
        time.sleep(0.01)
    return None


class StopListener(threading.Thread):
    """Background thread: calls ``on_stop`` when q (or Enter on POSIX) is pressed."""

    def __init__(self, on_stop, read_key=None, poll_interval: float = 0.1):
        super().__init__(daemon=True, name="stop-listener")
        self._on_stop = on_stop
        self._read_key = read_key or _default_read_key
        self._poll_interval = poll_interval
        self._finished = threading.Event()

    def run(self):
        while not self._finished.is_set():
            key = self._read_key(self._poll_interval)
            if key is None:
                continue
            if key.lower() in ("q", "\x03"):  # q or Ctrl+C
                self._on_stop()
                return

    def stop(self):
        """Stop listening (does NOT stop the robot)."""
        self._finished.set()
        self.join(timeout=1.0)


# ------------------------------------------------------------------- audio


class AudioPlayer:
    """Plays consecutive slices of the song on the laptop (non-blocking).

    ``enabled=False`` turns it into a no-op, for ``--no-audio`` runs (and tests,
    which never touch an audio device).
    """

    def __init__(self, y, sr, enabled: bool = True):
        self._y = y
        self._sr = sr
        self.enabled = enabled

    def play(self, start_s: float, end_s: float) -> None:
        if not self.enabled:
            return
        import sounddevice as sd

        chunk = self._y[int(round(start_s * self._sr)): int(round(end_s * self._sr))]
        sd.stop()
        sd.play(chunk, self._sr)

    def stop(self) -> None:
        if not self.enabled:
            return
        import sounddevice as sd

        sd.stop()


# ------------------------------------------------------------------- modes


def run_robot(plan: RunPlan, controller, player=None, print_fn=print) -> int:
    """Dance the plan on the robot; returns the number of completed segments."""
    completed = 0
    try:
        for segment in plan.segments:
            if controller.stopped:
                break
            print_fn(
                "[%d/%d] %s @ %.0f BPM (%.1fs)"
                % (segment.index, len(plan.segments), segment.style, segment.bpm, segment.duration_s),
                flush=True,
            )
            if player is not None:
                player.play(segment.start_s, segment.end_s)
            try:
                ok = controller.run_segment(segment.style, segment.bpm, segment.duration_s)
            except Exception:
                # An error mid-dance must never leave the robot moving.
                controller.stop()
                raise
            if player is not None:
                player.stop()
            if not ok:
                break
            completed += 1
    finally:
        if player is not None:
            player.stop()
    return completed


def _run_real(plan: RunPlan, args) -> int:
    try:
        from dance.audio.features import load_audio
    except ImportError as exc:  # pragma: no cover - librosa is a hard requirement
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2

    log_path = log_path_for(args.log_dir, "%s_motion" % args.robot)
    client = None
    controller = None
    listener = None
    dancing = False
    try:
        client = NaoClient(robot=args.robot, ip=args.ip, port=args.port)
        controller = DanceController(client, log_path=log_path)
        info = client.ping()
        if client.simulated:
            print("WARNING: the bridge reports a SIMULATED robot (%r): nothing physical will move."
                  % args.robot)
        else:
            print("Connected to %r (bridge %s)." % (args.robot, info.get("robot", "?")))
        client.posture("StandInit", 0.5)
        y, sr = load_audio(str(plan.song))
        player = AudioPlayer(y, sr, enabled=not args.no_audio)
        listener = StopListener(controller.stop)
        listener.start()
        print("Dancing. Press q (or Ctrl+C) to stop.", flush=True)
        dancing = True
        completed = run_robot(plan, controller, player)
    except NaoBridgeError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        if dancing:
            print("The robot was stopped and returned to StandInit.", file=sys.stderr)
        return 2
    except ValueError as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        if dancing:
            print("The robot was stopped and returned to StandInit.", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        if controller is not None:
            controller.stop()
        print("\nStopped by the user.", file=sys.stderr)
        return 130
    finally:
        if listener is not None:
            listener.stop()
        if client is not None:
            client.close()

    print("Segments completed: %d/%d" % (completed, len(plan.segments)))
    print("Run log: %s" % log_path)
    if controller is not None and controller.stopped:
        print("The run was stopped before the end.", file=sys.stderr)
        return 130
    return 0


# --------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m dance.main",
        description="Analyze a song, predict a dance style per segment and dance it on NAO.",
    )
    p.add_argument("--song", required=True, help="audio file to analyze and dance")
    p.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="trained model artifact (joblib)")
    p.add_argument("--min-conf", type=float, default=0.5,
                   help="below this confidence the previous style is kept (smoothing)")
    p.add_argument("--mock", action="store_true",
                   help="simulation: analyze and print the plan, never contact a robot")
    p.add_argument("--robot", choices=["zyra", "nyx", "sim"],
                   help="dance on this robot (uses the Python 2.7 bridge)")
    p.add_argument("--ip", help="override the robot IP (bridge --ip)")
    p.add_argument("--port", type=int, help="override the robot port (bridge --port)")
    p.add_argument("--no-audio", action="store_true", help="do not play the song on the laptop")
    p.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="directory for run logs")
    p.add_argument("--all-features", action="store_true", help="print every extracted feature")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.mock and args.robot:
        print("ERROR: --mock and --robot are mutually exclusive.", file=sys.stderr)
        return 2
    if not args.mock and not args.robot:
        print("ERROR: choose --mock (simulation, no robot) or --robot zyra|nyx|sim.", file=sys.stderr)
        return 2

    try:
        predictor = DancePredictor(args.model)
        plan = build_plan(predictor, args.song, min_conf=args.min_conf)
    except (FileNotFoundError, ValueError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2

    print(format_plan(plan, all_features=args.all_features))
    tag = "mock" if args.mock else args.robot
    log_path = write_timeline_csv(plan, log_path_for(args.log_dir, tag), simulated=args.mock)
    print("\nTimeline log: %s" % log_path)

    if args.mock:
        print("\n" + SIMULATION_NOTICE)
        return 0
    return _run_real(plan, args)


if __name__ == "__main__":
    sys.exit(main())
