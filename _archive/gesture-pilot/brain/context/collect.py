"""Record REAL context scenarios with the robot (human-labeled dataset).

The team puts NAO in a real situation, this tool reads the robot sensors,
asks the operator which gesture is shown, and then asks a human which action
NAO SHOULD take. The suggestion of any model/rule is deliberately NOT shown,
so labels are human judgment (not rule-generated).

Usage:
    python -m brain.context.collect --scenario-id esc01 --out data/context/scenarios.csv [--robot zyra] [--live] [--mock]

``--mock`` simulates the sonars (no robot, for rehearsing the procedure).
``--live`` tries the webcam + MediaPipe for gesture/person_distance and falls
back to keyboard input if it is not available.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from . import ACTIONS, GESTURES, SCENARIO_COLUMNS


def append_row(path, row: dict) -> None:
    """Append one scenario row (writes the header when the file is new)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SCENARIO_COLUMNS))
        if new_file:
            writer.writeheader()
        writer.writerow({k: row[k] for k in SCENARIO_COLUMNS})


def _mock_sense() -> dict:
    return {
        "sonar_left": round(random.uniform(0.15, 2.0), 2),
        "sonar_right": round(random.uniform(0.15, 2.0), 2),
        "moving": random.random() < 0.25,
    }


def _ask_choice(prompt: str, options, input_fn=input) -> str:
    while True:
        print(prompt)
        for i, opt in enumerate(options, 1):
            print(f"  {i}) {opt}")
        raw = input_fn("> ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        if raw.upper() in options:
            return raw.upper()
        print("Opcion no valida, intenta de nuevo.")


def _ask_float(prompt: str, default: float | None, lo: float, hi: float, input_fn=input) -> float:
    while True:
        raw = input_fn(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            value = float(raw.replace(",", "."))
        except ValueError:
            print("Ingresa un numero.")
            continue
        if lo <= value <= hi:
            return value
        print(f"El valor debe estar entre {lo} y {hi}.")


def _read_live(frames: int = 15):
    """Best-effort webcam read: (gesture, confidence, person_distance) or None."""
    try:
        import cv2
        import mediapipe as mp

        from brain.gestures.features import person_distance
        from brain.gestures.predictor import GesturePredictor

        predictor = GesturePredictor("models/gesture_model.joblib")
        cap = cv2.VideoCapture(0)
        pose = mp.solutions.pose.Pose()
        label, conf, dist = None, 0.0, None
        for _ in range(frames):
            ok, frame = cap.read()
            if not ok:
                continue
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks is None:
                continue
            import numpy as np

            lm = np.array([[p.x, p.y, p.z, p.visibility] for p in res.pose_landmarks.landmark])
            label, conf = predictor.update(lm)
            dist = float(person_distance(lm))
        cap.release()
        if label is None or dist is None:
            return None
        return label, float(conf), dist
    except Exception as exc:  # noqa: BLE001 - optional path, fall back to keyboard
        print(f"[live] no disponible ({exc}); se usara el teclado.")
        return None


def record_one(scenario_id: str, out, client, mock: bool, live: bool, input_fn=input) -> dict:
    """Record a single labeled observation and append it to ``out``."""
    sense = _mock_sense() if mock else client.sense()
    sonar_left, sonar_right = float(sense["sonar_left"]), float(sense["sonar_right"])
    moving = int(bool(sense.get("moving", False)))
    print(f"\nSensores: sonar_izq={sonar_left:.2f} m, sonar_der={sonar_right:.2f} m, en_movimiento={moving}")

    live_data = _read_live() if live else None
    if live_data:
        gesture, confidence, person_distance = live_data
        print(f"Camara: gesto={gesture} conf={confidence:.2f} distancia_persona={person_distance:.3f}")
    else:
        gesture = _ask_choice("Que gesto esta mostrando la persona?", list(GESTURES), input_fn)
        confidence = _ask_float("Confianza del gesto 0-1 [Enter=1.0]: ", 1.0, 0.0, 1.0, input_fn)
        person_distance = _ask_float(
            "Distancia de la persona (ancho de hombros / ancho de imagen, 0.03-0.45, ej. 0.16): ",
            None, 0.0, 1.0, input_fn,
        )
    action = _ask_choice("Que DEBERIA hacer NAO en esta situacion? (criterio humano)", list(ACTIONS), input_fn)
    row = {
        "gesture": gesture,
        "gesture_confidence": round(confidence, 3),
        "sonar_left": round(sonar_left, 3),
        "sonar_right": round(sonar_right, 3),
        "person_distance": round(person_distance, 3),
        "moving": moving,
        "action": action,
        "scenario_id": scenario_id,
    }
    append_row(out, row)
    print(f"Guardado: {row}")
    return row


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Record real, human-labeled context scenarios with NAO.")
    p.add_argument("--scenario-id", required=True, help="id of the physical scene (rows of one scene share it)")
    p.add_argument("--out", default="data/context/scenarios.csv")
    p.add_argument("--robot", default="zyra", help="nyx|zyra|sim (ignored with --mock)")
    p.add_argument("--live", action="store_true", help="read gesture/person_distance from the webcam")
    p.add_argument("--mock", action="store_true", help="simulate sonars, do not connect to a robot")
    return p


def main(argv=None, input_fn=input) -> int:
    args = build_parser().parse_args(argv)
    client = None
    if not args.mock:
        from brain.nao_client import NaoClient  # lazy: needs the Python 2.7 bridge

        client = NaoClient(robot=args.robot, mock=False)
    count = 0
    try:
        while True:
            record_one(args.scenario_id, args.out, client, args.mock, args.live, input_fn)
            count += 1
            again = input_fn("Otra muestra en esta escena? [Enter=si, q=salir]: ").strip().lower()
            if again == "q":
                break
    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        if client is not None:
            client.close()
    print(f"{count} muestra(s) guardadas en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
