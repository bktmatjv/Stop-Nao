"""Record labeled pose landmarks from the laptop webcam.

Usage:
    python -m brain.gestures.capture --person ana --label ADVANCE --seconds 20 --camera 0

Output: data/gestures/<person>_<label>_<timestamp>.csv with columns
``person,label,x0,y0,z0,v0,...,x32,y32,z32,v32``.

Note on left/right: the preview is mirrored (like a mirror) but the saved
landmarks are the raw MediaPipe ones, so LEFT always means the PERSON's left.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import LABELS
from .features import CSV_COLUMNS, NUM_LANDMARKS, missing_key_landmarks

INSTRUCTIONS = {
    "ADVANCE": "ADVANCE    = both arms raised above the head.",
    "STOP": "STOP       = both forearms crossed in front of the chest.",
    "TURN_LEFT": "TURN_LEFT  = YOUR left arm extended horizontally to the side.",
    "TURN_RIGHT": "TURN_RIGHT = YOUR right arm extended horizontally to the side.",
    "NONE": "NONE       = arms relaxed / random natural movements / walking around.",
}


# ASCII only: OpenCV's built-in fonts cannot render accents.
_JOINT_ES = {"l_shoulder": "hombro izq", "r_shoulder": "hombro der", "l_elbow": "codo izq",
             "r_elbow": "codo der", "l_wrist": "muneca izq", "r_wrist": "muneca der"}
LOW_FULL_RATIO = 0.5


def framing_warning(landmarks: np.ndarray | None) -> str | None:
    """Overlay warning when the person or a key joint is out of frame, else None."""
    if landmarks is None:
        return "SIN PERSONA: colocate frente a la camara"
    missing = missing_key_landmarks(landmarks)
    if not missing:
        return None
    return "FUERA DE CUADRO: " + ", ".join(_JOINT_ES[m] for m in missing) + " -> alejate ~2 m"


def landmarks_to_row(person: str, label: str, landmarks: np.ndarray) -> list:
    """Flatten a (33, 4) landmark array into one CSV row matching CSV_COLUMNS."""
    arr = np.asarray(landmarks, dtype=float)
    if arr.shape != (NUM_LANDMARKS, 4):
        raise ValueError(f"landmarks must have shape (33, 4), got {arr.shape}")
    return [person, label] + [float(v) for v in arr.reshape(-1)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m brain.gestures.capture",
        description="Record labeled pose landmarks from the webcam into a CSV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Gestures:\n  " + "\n  ".join(INSTRUCTIONS[k] for k in LABELS),
    )
    parser.add_argument("--person", required=True, help="Name of the person being recorded (no spaces/underscores).")
    parser.add_argument("--label", required=True, choices=LABELS, help="Gesture being recorded.")
    parser.add_argument("--seconds", type=float, default=20.0, help="Recording duration (default 20).")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default 0).")
    parser.add_argument("--out", default="data/gestures", help="Output directory (default data/gestures).")
    parser.add_argument("--countdown", type=int, default=3, help="Countdown seconds before recording (default 3).")
    return parser


def run(args: argparse.Namespace) -> int:
    # Heavy imports are deferred so that --help and imports stay light.
    import cv2
    import mediapipe as mp

    person = args.person.strip().replace(" ", "-").replace("_", "-")
    label = args.label
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Person: {person} | Gesture: {label} | Duration: {args.seconds:.0f}s")
    print(INSTRUCTIONS[label])
    print("Keep your upper body (head to hips) visible. Vary your position/distance a little.")
    print("Press 'q' in the preview window to quit (partial data is saved).")
    print("=" * 60)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {args.camera}", file=sys.stderr)
        return 1

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    rows: list[list] = []
    skipped = 0
    full = 0  # frames with all shoulders/elbows/wrists visible
    warning: str | None = None
    window = "NAO gesture capture"

    with mp_pose.Pose(model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        phase = "countdown"
        phase_start = time.time()
        rec_start = 0.0
        quit_requested = False
        while not quit_requested:
            ok, frame = cap.read()
            if not ok:
                print("ERROR: failed to read from camera", file=sys.stderr)
                break
            now = time.time()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            lm = None
            if result.pose_landmarks is not None:
                mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                lm = np.array(
                    [[p.x, p.y, p.z, p.visibility] for p in result.pose_landmarks.landmark],
                    dtype=float,
                )
            warning = framing_warning(lm)

            if phase == "countdown":
                remaining = args.countdown - (now - phase_start)
                if remaining <= 0:
                    phase = "recording"
                    rec_start = now
                    print("Recording...")
                    text = "GO!"
                else:
                    text = f"Starting in {int(np.ceil(remaining))}"
            else:
                elapsed = now - rec_start
                if elapsed >= args.seconds:
                    break
                if lm is not None:
                    rows.append(landmarks_to_row(person, label, lm))
                    full += int(not missing_key_landmarks(lm))
                else:
                    skipped += 1
                text = (f"REC {label} {elapsed:4.1f}/{args.seconds:.0f}s "
                        f"frames={len(rows)} completos={full} sin-pose={skipped}")

            display = cv2.flip(frame, 1)  # mirror preview only; saved data is raw
            cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if warning:
                cv2.putText(display, warning, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(display, INSTRUCTIONS[label], (10, display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow(window, display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                quit_requested = True

    cap.release()
    cv2.destroyAllWindows()

    if not rows:
        print("No frames recorded; nothing saved.")
        return 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{person}_{label}_{stamp}.csv"
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)
    print(f"Saved {len(rows)} frames ({skipped} without pose, {full} with all arm joints visible) to {path}")
    if full < LOW_FULL_RATIO * len(rows):
        print(f"WARNING: only {full}/{len(rows)} frames show both arms completely. "
              "Step back (~2 m) so head, hips and both extended arms fit in the frame, then re-record.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
