"""Record REAL context scenarios with the robot (human-labeled dataset).

The team puts NAO in a real situation, this tool reads the robot sensors,
asks the operator which gesture is shown, and then asks a HUMAN which action
NAO SHOULD take. The suggestion of any model/rule is deliberately NOT shown,
so labels are human judgment (not rule-generated).

Usage:
    python -m brain.context.collect --scenario-id esc01 [--robot zyra] [--ip IP --port PORT]
        [--out data/context/scenarios.csv] [--mock]

``--mock`` does not connect to a robot: the operator types the sonar values
(for rehearsing the procedure). Rows of one physical scene share the
``scenario_id`` (the training split is grouped by it). An unreadable sonar is
stored as 0.0 (invalid = unsafe, same convention as the safety layer).
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from . import ACTIONS, GESTURES, SCENARIO_COLUMNS

STANDING_POSTURES = {"Stand", "StandInit", "StandZero"}
_YES = {"s", "si", "sí", "y", "yes", "1", "true"}


def is_standing(posture) -> bool:
    """True when the ALRobotPosture name is one of the standing postures."""
    return posture in STANDING_POSTURES


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
        print("Opción no válida, intenta de nuevo.")


def _ask_float(prompt: str, default: float | None, lo: float, hi: float, input_fn=input) -> float:
    while True:
        raw = input_fn(prompt).strip()
        if raw == "" and default is not None:
            return default
        try:
            value = float(raw.replace(",", "."))
        except ValueError:
            print("Ingresa un número.")
            continue
        if lo <= value <= hi:
            return value
        print(f"El valor debe estar entre {lo} y {hi}.")


def _ask_yes_no(prompt: str, default: bool, input_fn=input) -> bool:
    raw = input_fn(prompt).strip().lower()
    return default if raw == "" else raw in _YES


def _sonar_value(reading) -> float:
    """Valid reading as float, otherwise 0.0 (stored as 'invalid')."""
    try:
        value = float(reading)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value > 0 else 0.0


def _mock_sense(input_fn=input) -> dict:
    """Simulated sensors: values typed by the operator, no robot involved."""
    return {
        "sonar_left": _ask_float("Sonar izquierdo en metros (ej. 0.45): ", None, 0.0, 5.0, input_fn),
        "sonar_right": _ask_float("Sonar derecho en metros (ej. 1.2): ", None, 0.0, 5.0, input_fn),
        "moving": _ask_yes_no("¿El robot se está moviendo? [s/N]: ", False, input_fn),
        "posture": "Stand" if _ask_yes_no("¿El robot está de pie? [S/n]: ", True, input_fn) else "Sit",
    }


def record_one(scenario_id: str, out, client, mock: bool, input_fn=input) -> dict:
    """Record a single labeled observation and append it to ``out``."""
    sense = _mock_sense(input_fn) if mock else client.sense()
    sonar_left, sonar_right = _sonar_value(sense.get("sonar_left")), _sonar_value(sense.get("sonar_right"))
    moving = int(bool(sense.get("moving", False)))
    standing = int(is_standing(sense.get("posture")))
    print(f"\nSensores: sonar_izq={sonar_left:.2f} m, sonar_der={sonar_right:.2f} m, "
          f"en_movimiento={moving}, de_pie={standing} (postura={sense.get('posture')})")

    gesture = _ask_choice("¿Qué gesto está mostrando la persona?", list(GESTURES), input_fn)
    confidence = _ask_float("Confianza del gesto 0-1 [Enter=1.0]: ", 1.0, 0.0, 1.0, input_fn)
    hand_size = _ask_float("Tamaño de la mano (muñeca a base del dedo medio, 0.03-0.45) [Enter=0.15]: ",
                           0.15, 0.0, 1.0, input_fn)
    action = _ask_choice("¿Qué DEBERÍA hacer NAO en esta situación? (criterio humano)", list(ACTIONS), input_fn)
    row = {
        "gesture": gesture,
        "gesture_confidence": round(confidence, 3),
        "sonar_left": round(sonar_left, 3),
        "sonar_right": round(sonar_right, 3),
        "hand_size": round(hand_size, 3),
        "moving": moving,
        "standing": standing,
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
    p.add_argument("--ip", default=None, help="robot IP (overrides --robot)")
    p.add_argument("--port", type=int, default=None, help="robot port")
    p.add_argument("--mock", action="store_true", help="type sonar values by hand, do not connect to a robot")
    return p


def main(argv=None, input_fn=input) -> int:
    args = build_parser().parse_args(argv)
    client = None
    if not args.mock:
        from brain.nao_client import NaoClient  # lazy: needs the Python 2.7 bridge

        client = NaoClient(robot=args.robot, ip=args.ip, port=args.port)
    count = 0
    try:
        while True:
            record_one(args.scenario_id, args.out, client, args.mock, input_fn)
            count += 1
            again = input_fn("¿Otra muestra en esta escena? [Enter=sí, q=salir]: ").strip().lower()
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
