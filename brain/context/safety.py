"""Hard safety / validation layer (spec S3 and S4).

"si recibe la orden gestual de avanzar, pero detecta un obstaculo, decide
permanecer detenido y emitir una advertencia."

``validate(action, ctx)`` takes the REQUESTED action (from the learned model or
from the gesture intent) and the sensed context, and returns a ``Decision``:
the action NAO may really execute, its parameters, a Spanish ``reason`` that is
spoken and shown, the eye colour and whether validation changed the request.
It runs after any model and cannot be bypassed by it. Pure functions, no I/O.

Eye colours: green = accepted, yellow = doubtful/limited, red = refused for
safety, blue = idle/listening.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

SAFE_FRONT = 0.30  # m, minimum free distance in front
SAFE_SIDE = 0.25  # m, minimum free distance on the side we turn towards
ADVANCE_STEP = 0.35  # m, a full advance (extended range: 35 cm)
MIN_STEP = 0.05  # m, smaller steps are not worth taking
TURN_ANGLE = 0.785  # rad (45 degrees)
MIN_CONF = 0.6
TOO_CLOSE_HAND = 0.25  # hand_size above this: person too close to wave

_PASSIVE_REASONS = {
    "WARN_OBSTACLE": ("Hay un obstáculo cerca. No me muevo.", "red"),
    "WARN_TOO_CLOSE": ("Estás demasiado cerca. Aléjate un poco.", "red"),
    "WARN_MUST_STOP": ("Primero debo detenerme.", "red"),
    "WARN_MUST_STAND": ("Primero debo ponerme de pie.", "yellow"),
    "ASK_CONFIRM": ("No estoy seguro del gesto. ¿Puedes repetirlo?", "yellow"),
}


@dataclass
class Decision:
    action: str
    params: dict = field(default_factory=dict)
    reason: str = ""
    eye_color: str = "blue"
    overridden: bool = False


def is_valid_reading(reading) -> bool:
    """A sonar reading is valid when it is a finite number > 0."""
    try:
        value = float(reading)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and value > 0.0


def _cm(meters) -> int:
    return int(round(float(meters) * 100))


def _flag(ctx: dict, key: str, default: bool) -> bool:
    value = ctx.get(key, default)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return bool(value)


def _result(requested, action, params, reason, eye) -> Decision:
    return Decision(action, params, reason, eye, overridden=(action != requested))


def _refuse(requested, action, reason=None) -> Decision:
    default_reason, eye = _PASSIVE_REASONS[action]
    return _result(requested, action, {}, reason or default_reason, eye)


def _obstacle_ahead(requested, front) -> Decision:
    return _refuse(requested, "WARN_OBSTACLE",
                   f"Hay un obstáculo a {_cm(front)} centímetros. No puedo avanzar.")


def _validate_advance(requested: str, ctx: dict) -> Decision:
    left, right = ctx.get("sonar_left"), ctx.get("sonar_right")
    if not (is_valid_reading(left) and is_valid_reading(right)):
        return _refuse(requested, "WARN_OBSTACLE",
                       "No puedo medir la distancia al frente. No avanzo por seguridad.")
    front = min(float(left), float(right))
    if front < SAFE_FRONT:
        return _obstacle_ahead(requested, front)
    graded = front < SAFE_FRONT + ADVANCE_STEP
    step = round(front - SAFE_FRONT, 3) if graded else ADVANCE_STEP
    if graded and step < MIN_STEP - 1e-9:
        return _obstacle_ahead(requested, front)
    if not _flag(ctx, "standing", False):
        return _refuse(requested, "WARN_MUST_STAND")
    if graded:
        return _result(requested, "ADVANCE_SHORT", {"x": step},
                       f"Solo puedo avanzar {_cm(step)} centímetros.", "yellow")
    return _result(requested, "ADVANCE", {"x": ADVANCE_STEP}, "Avanzo, el camino está libre.", "green")


def _validate_turn(action: str, ctx: dict) -> Decision:
    left_side = action == "TURN_LEFT"
    side = "izquierda" if left_side else "derecha"
    reading = ctx.get("sonar_left" if left_side else "sonar_right")
    if not is_valid_reading(reading):
        return _refuse(action, "WARN_OBSTACLE",
                       f"No puedo medir la distancia a mi {side}. No giro por seguridad.")
    if float(reading) < SAFE_SIDE:
        return _refuse(action, "WARN_OBSTACLE",
                       f"Hay un obstáculo a mi {side}, a {_cm(reading)} centímetros. No puedo girar.")
    if not _flag(ctx, "standing", False):
        return _refuse(action, "WARN_MUST_STAND")
    theta = TURN_ANGLE if left_side else -TURN_ANGLE
    return _result(action, action, {"theta": theta}, f"Giro a la {side}.", "cyan")


def _validate_wave(ctx: dict) -> Decision:
    if _flag(ctx, "moving", False):
        return _refuse("WAVE", "WARN_MUST_STOP", "Me estoy moviendo. Primero debo detenerme para saludar.")
    try:
        too_close = float(ctx.get("hand_size")) > TOO_CLOSE_HAND
    except (TypeError, ValueError):
        too_close = False
    if too_close:
        return _refuse("WAVE", "WARN_TOO_CLOSE", "Estás muy cerca. Aléjate un poco para que pueda saludar.")
    if not _flag(ctx, "standing", False):
        return _refuse("WAVE", "WARN_MUST_STAND", "Estoy sentado. Primero debo ponerme de pie para saludar.")
    return _result("WAVE", "WAVE", {}, "¡Hola! ¡Modo fiesta y victoria activado! ¡A bailar!", "magenta")


def _validate_greet(ctx: dict) -> Decision:
    """TWO_FINGERS gesture: a quick salute greeting. Allowed standing or sitting."""
    if _flag(ctx, "moving", False):
        return _refuse("GREET", "WARN_MUST_STOP", "Me estoy moviendo. Espera un momento para saludar.")
    return _result("GREET", "GREET", {}, "¡Hola! ¡Encantado de conocerte!", "cyan")


def _validate_posture(action: str, ctx: dict) -> Decision:
    if _flag(ctx, "moving", False):
        verb = "sentarme" if action == "SIT" else "pararme"
        return _refuse(action, "WARN_MUST_STOP", f"Me estoy moviendo. Primero debo detenerme para {verb}.")
    text = "Me siento a descansar." if action == "SIT" else "Me pongo de pie y estoy listo."
    return _result(action, action, {}, text, "white")


def validate(action: str, ctx: dict) -> Decision:
    """Validate the requested ``action`` against the sensed context ``ctx``.

    ``ctx`` keys: ``gesture_confidence, sonar_left, sonar_right, hand_size,
    moving, standing`` (see ``FEATURE_COLUMNS``). Invalid sonar readings
    (<= 0, NaN, None) are treated as unsafe. ``overridden`` is True whenever
    the returned action differs from the requested one.
    """
    if action == "WAIT":
        return _result(action, "WAIT", {}, "Estoy atento, esperando un gesto.", "blue")
    if action == "STOP":
        return _result(action, "STOP", {}, "Me detengo.", "green")
    if action in _PASSIVE_REASONS:  # warnings / confirmation requests never move the robot
        return _refuse(action, action)
    try:
        confidence = float(ctx.get("gesture_confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if not confidence >= MIN_CONF:  # also catches NaN
        return _refuse(action, "ASK_CONFIRM")
    if action in ("ADVANCE", "ADVANCE_SHORT"):  # a model may propose the short step; re-derive it
        return _validate_advance(action, ctx)
    if action in ("TURN_LEFT", "TURN_RIGHT"):
        return _validate_turn(action, ctx)
    if action == "WAVE":
        return _validate_wave(ctx)
    if action == "GREET":
        return _validate_greet(ctx)
    if action in ("SIT", "STAND"):
        return _validate_posture(action, ctx)
    raise ValueError(f"unknown action: {action!r}")
