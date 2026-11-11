"""Dance controller: runs choreography segments on a NaoClient.

Decoupled from the ML code: it only needs a style name, a tempo and a duration.
The segments are chained smoothly because every segment starts from the pose
where the previous one ended. ``stop()`` is the accessible stop mechanism: it is
thread-safe, kills the running motion, returns the robot to StandInit and keeps
refusing new segments until ``clear_stop()`` is called.

Events are kept in ``self.events`` and, if ``log_path`` is given, appended to a
CSV file with the columns timestamp, event, style, bpm, duration, task_id,
simulated, error.
"""
import csv
import os
import threading
import time
from datetime import datetime

from dance.robot.choreographies import (
    JOINTS, build_keyframes, final_pose, validate_keyframes,
)

LOG_FIELDS = ["timestamp", "event", "style", "bpm", "duration", "task_id", "simulated", "error"]


class DanceController:
    def __init__(self, client, log_path=None, poll_interval=0.05, wait_margin=5.0):
        self.client = client
        self.log_path = log_path
        self.poll_interval = poll_interval
        self.wait_margin = wait_margin  # extra seconds before declaring a timeout
        self.events = []
        self.last_pose = None           # last commanded pose (joint -> radians)
        self._limits = None
        self._stop_event = threading.Event()
        self._log_lock = threading.Lock()

    # -- stop mechanism -----------------------------------------------------
    @property
    def stopped(self):
        return self._stop_event.is_set()

    def clear_stop(self):
        """Allow new segments after a stop()."""
        self._stop_event.clear()

    def stop(self):
        """Thread-safe emergency stop: halt the motion and go to StandInit."""
        self._stop_event.set()
        self._record("stop")
        try:
            self.client.stop_motion()
        except Exception as exc:
            self._record("error", error="stop_motion failed: %s" % exc)
        try:
            self.client.posture("StandInit", 0.5)
        except Exception as exc:
            self._record("error", error="StandInit failed: %s" % exc)
        self.last_pose = None  # StandInit is close to the neutral pose

    # -- running ------------------------------------------------------------
    def run_segment(self, style, bpm, duration_s):
        """Dance one segment; return True if it ran to the end, False if stopped.

        Errors (invalid style, limit violation, bridge failure) are logged as an
        ``error`` event and re-raised.
        """
        if self.stopped:
            self._record("skipped", style, bpm, duration_s, error="stop requested")
            return False
        task_id = None
        try:
            if self._limits is None:
                self._limits = self.client.get_limits(JOINTS)
            names, angles, times = build_keyframes(style, bpm, duration_s, self.last_pose)
            validate_keyframes(names, angles, self._limits)
            total = times[0][-1]
            task_id = self.client.play_keyframes(names, angles, times)
            self._record("segment_start", style, bpm, duration_s, task_id)
            if self.stopped:  # stop() raced with the start: make sure it is killed
                self.client.stop_motion()
                return False
            self.last_pose = final_pose(names, angles)
            return self._wait(task_id, total, style, bpm, duration_s)
        except Exception as exc:
            self._record("error", style, bpm, duration_s, task_id, error="%s: %s" % (type(exc).__name__, exc))
            raise

    def _wait(self, task_id, total, style, bpm, duration_s):
        deadline = time.monotonic() + total + self.wait_margin
        while True:
            if self._stop_event.wait(self.poll_interval):
                return False  # stop() already halted the motion and logged it
            if not self.client.is_running(task_id):
                self._record("segment_end", style, bpm, duration_s, task_id)
                return True
            if time.monotonic() > deadline:
                self.client.stop_motion()
                raise TimeoutError("segment still running %.1fs after its expected end" % self.wait_margin)

    # -- logging ------------------------------------------------------------
    def _record(self, event, style=None, bpm=None, duration=None, task_id=None, error=None):
        row = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "event": event,
            "style": style,
            "bpm": bpm,
            "duration": duration,
            "task_id": task_id,
            "simulated": getattr(self.client, "simulated", None),
            "error": error,
        }
        with self._log_lock:
            self.events.append(row)
            if self.log_path:
                try:
                    new = not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0
                    with open(self.log_path, "a", newline="", encoding="utf-8") as fh:
                        writer = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
                        if new:
                            writer.writeheader()
                        writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
                except OSError:
                    pass  # logging must never break the dance
