"""Record labeled hand landmarks from the laptop webcam.

Usage:
    python -m brain.gestures.capture --person ana --label THUMBS_UP --seconds 15 --camera 0

Output: data/gestures/<person>_<label>_<timestamp>.csv with columns
``person,label,handedness,x0,y0,z0,...,x20,y20,z20``.

Tips: record with the hand you will use in the demo and vary the rotation and
the distance (0.5 - 1.5 m) a little while recording. The preview is mirrored
(like a mirror) but the saved landmarks and the handedness label are the raw
MediaPipe ones; see ``features.py`` for what that means for POINT_LEFT /
POINT_RIGHT.
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
from .features import CSV_COLUMNS, MIN_HAND_SIZE, NUM_LANDMARKS, hand_size

# ASCII only: OpenCV's built-in fonts cannot render accents.
INSTRUCTIONS = {
    "THUMBS_UP": "THUMBS_UP   = pulgar arriba, resto de la mano cerrada.",
    "OPEN_PALM": "OPEN_PALM   = palma abierta hacia la camara, dedos separados.",
    "POINT_LEFT": "POINT_LEFT  = senala con el indice hacia TU izquierda.",
    "POINT_RIGHT": "POINT_RIGHT = senala con el indice hacia TU derecha.",
    "VICTORY": "VICTORY     = dos dedos en V (indice y medio), el resto cerrado.",
    "FIST": "FIST        = puno cerrado.",
    "TWO_FINGERS": "TWO_FINGERS = indice y medio juntos y rectos, apuntando al frente (saludo).",
    "LOVE_YOU": "LOVE_YOU    = pulgar, indice y menique extendidos; medio y anular doblados.",
    "NONE": "NONE        = mano relajada / movimientos naturales / sin gesto.",
}
NO_HAND_WARNING = "NO SE VE LA MANO"
FAR_WARNING = "MANO MUY LEJOS -> acercate o acerca la mano"
LOW_VALID_RATIO = 0.5

# LOVE_YOU is easy to confuse with VICTORY: keep middle+ring clearly curled and pinky+thumb clearly out.

# Two-finger VICTORY hint for the capture preview
VICTORY_HINT = "VICTORY: exactamente 2 dedos en V hacia arriba"


def hand_warning(landmarks: np.ndarray | None) -> str | None:
    """Overlay warning when there is no hand or it is too far, else None."""
    if landmarks is None:
        return NO_HAND_WARNING
    if hand_size(landmarks) < MIN_HAND_SIZE:
        return FAR_WARNING
    return None


def landmarks_to_row(person: str, label: str, handedness: str, landmarks: np.ndarray) -> list:
    """Flatten a (21, 3) landmark array into one CSV row matching CSV_COLUMNS."""
    arr = np.asarray(landmarks, dtype=float)
    if arr.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"landmarks must have shape (21, 3), got {arr.shape}")
    if handedness not in ("Left", "Right"):
        raise ValueError(f"handedness must be 'Left' or 'Right', got {handedness!r}")
    return [person, label, handedness] + [float(v) for v in arr.reshape(-1)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m brain.gestures.capture",
        description="Record labeled hand landmarks from the webcam into a CSV file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Gestures:\n  " + "\n  ".join(INSTRUCTIONS[k] for k in LABELS),
    )
    parser.add_argument("--person", required=True, help="Name of the person being recorded (no spaces/underscores).")
    parser.add_argument("--label", required=True, choices=LABELS, help="Gesture being recorded.")
    parser.add_argument("--seconds", type=float, default=15.0, help="Recording duration (default 15).")
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
    print("Use the hand you will use in the demo. Vary rotation and distance (0.5 - 1.5 m) a little.")
    print("Press 'q' in the preview window to quit (partial data is saved).")
    print("=" * 60)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: cannot open camera {args.camera}", file=sys.stderr)
        return 1

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    rows: list[list] = []
    no_hand = 0
    too_far = 0
    window = "NAO gesture capture"

    with mp_hands.Hands(max_num_hands=1, model_complexity=1, min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
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
            result = hands.process(rgb)

            lm = None
            handedness = None
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                lm = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=float)
                handedness = result.multi_handedness[0].classification[0].label
            warning = hand_warning(lm)

            if phase == "countdown":
                remaining = args.countdown - (now - phase_start)
                if remaining <= 0:
                    phase = "recording"
                    rec_start = now
                    print("Recording...")
                    text = "YA!"
                else:
                    text = f"Empieza en {int(np.ceil(remaining))}"
            else:
                elapsed = now - rec_start
                if elapsed >= args.seconds:
                    break
                if lm is None:
                    no_hand += 1
                elif hand_size(lm) < MIN_HAND_SIZE:
                    too_far += 1  # landmarks of a far hand are unreliable: not recorded
                else:
                    rows.append(landmarks_to_row(person, label, handedness, lm))
                text = (f"REC {label} {elapsed:4.1f}/{args.seconds:.0f}s "
                        f"frames={len(rows)} sin-mano={no_hand} lejos={too_far}")

            display = cv2.flip(frame, 1)  # mirror preview only; saved data is raw
            cv2.putText(display, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if warning:
                cv2.putText(display, warning, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
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
    print(f"Saved {len(rows)} frames ({no_hand} without hand, {too_far} too far) to {path}")
    total = len(rows) + no_hand + too_far
    if len(rows) < LOW_VALID_RATIO * total:
        print(f"WARNING: only {len(rows)}/{total} frames had a usable hand. "
              "Keep the hand in view and 0.5 - 1.5 m from the camera, then re-record.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
