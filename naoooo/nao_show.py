# -*- coding: utf-8 -*-
r"""Mini-show para el simulador NAO (sin caminar).

Ejecutar desde Python 3 o Python 2.7:
    .\.venv\Scripts\python.exe .\nao_show.py
    .\.venv\Scripts\python.exe .\nao_show.py --port 59242
    .\.venv\Scripts\python.exe .\nao_show.py --dry-run

Python 3 relanza este mismo archivo con C:\Python27\python.exe.
El modo --dry-run muestra la secuencia sin importar NAOqi ni conectarse.
Usar SOLO con un simulador: localhost tambien puede reenviar a un robot real.
"""
from __future__ import print_function

import argparse
import os
import subprocess
import sys


PYTHON27 = r"C:\Python27\python.exe"
SDK_DIR = r"C:\Python27\Lib\site-packages"

# Valores absolutos en radianes; las manos usan apertura entre 0 y 1.
# Cada paso indica: descripcion, frase opcional, duracion, articulaciones.
STEPS = [
    ("Preparar saludo", "Hola. Soy Nao. Mira lo que puedo hacer.", 1.2,
     {"RShoulderPitch": 0.25, "RShoulderRoll": -0.30,
      "RElbowYaw": 1.20, "RElbowRoll": 1.15, "RHand": 1.0}),
    ("Saludar: uno", None, 0.6, {"RElbowRoll": 0.65}),
    ("Saludar: dos", None, 0.6, {"RElbowRoll": 1.30}),
    ("Saludar: tres", None, 0.6, {"RElbowRoll": 0.65}),
    ("Saludar: cuatro", None, 0.6, {"RElbowRoll": 1.15}),
    ("Bajar el brazo", None, 1.2,
     {"RShoulderPitch": 1.40, "RShoulderRoll": -0.15,
      "RElbowYaw": 1.20, "RElbowRoll": 0.45, "RHand": 0.3}),
    ("Escanear a la izquierda", "Activando modo explorador.", 1.0,
     {"HeadYaw": 0.55, "HeadPitch": -0.10}),
    ("Escanear a la derecha", None, 1.4, {"HeadYaw": -0.55}),
    ("Mirar al frente", "Todo en orden. Ahora, un poco de ritmo.", 1.0,
     {"HeadYaw": 0.0, "HeadPitch": 0.0}),
    ("Preparar baile", None, 1.4,
     {"LShoulderPitch": 0.95, "RShoulderPitch": 0.95,
      "LShoulderRoll": 0.25, "RShoulderRoll": -0.25,
      "LElbowYaw": -1.20, "RElbowYaw": 1.20,
      "LElbowRoll": -0.70, "RElbowRoll": 0.70,
      "LHand": 0.8, "RHand": 0.8}),
]

for beat in range(4):
    STEPS.append(("Baile: compas %d" % (beat + 1), None, 0.8,
                  {"LShoulderPitch": 0.65 if beat % 2 == 0 else 1.15,
                   "RShoulderPitch": 1.15 if beat % 2 == 0 else 0.65}))

STEPS.extend([
    ("Pose final", "Soy pequeno, pero tengo estilo.", 1.2,
     {"LShoulderPitch": 0.75, "RShoulderPitch": 0.75,
      "LShoulderRoll": 0.40, "RShoulderRoll": -0.40,
      "LElbowRoll": -0.90, "RElbowRoll": 0.90,
      "HeadYaw": 0.0, "HeadPitch": -0.08}),
    ("Reverencia con la cabeza", "Gracias por ver mi show.", 1.0,
     {"HeadPitch": 0.25}),
    ("Levantar la mirada", None, 1.0, {"HeadPitch": 0.0}),
])


def say(tts, text):
    if tts is not None and text:
        try:
            tts.say(text)
        except Exception as exc:
            print("[Voz no disponible] %s" % exc)


def run_show(port, mute):
    # Importar solo aqui: --dry-run funciona aunque falte el SDK.
    from naoqi import ALProxy

    host = "127.0.0.1"
    print("Conectando al simulador en %s:%d..." % (host, port))
    motion = ALProxy("ALMotion", host, port)
    posture = ALProxy("ALRobotPosture", host, port)
    tts = None
    if not mute:
        try:
            tts = ALProxy("ALTextToSpeech", host, port)
        except Exception as exc:
            print("[Continuando sin voz] %s" % exc)
        if tts is not None:
            try:
                tts.setLanguage("Spanish")
            except Exception as exc:
                print("[Se usara el idioma disponible] %s" % exc)

    # Comprobar articulaciones y limites ANTES de ordenar movimientos.
    names = sorted(set(name for step in STEPS for name in step[3]))
    available = set(motion.getBodyNames("Body"))
    missing = set(names) - available
    if missing:
        raise RuntimeError("El modelo no tiene estas articulaciones: %s" %
                           ", ".join(sorted(missing)))
    raw_limits = motion.getLimits(names)
    if len(raw_limits) != len(names):
        raise RuntimeError("El simulador no devolvio todos los limites.")
    limits = dict(zip(names, raw_limits))
    for label, phrase, duration, targets in STEPS:
        for name, target in targets.items():
            low, high = limits[name][:2]
            if not low <= target <= high:
                raise RuntimeError("%s: %s fuera de limites (%s, %s)." %
                                   (label, name, low, high))

    motion.stiffnessInterpolation("Body", 1.0, 1.0)
    if not posture.goToPosture("StandInit", 0.5):
        raise RuntimeError("No se pudo adoptar la postura inicial.")

    for index, (label, phrase, duration, targets) in enumerate(STEPS, 1):
        print("[%02d/%02d] %s" % (index, len(STEPS), label))
        say(tts, phrase)
        joints = sorted(targets)
        # Llamada bloqueante: el siguiente paso espera al movimiento actual.
        motion.angleInterpolation(joints,
                                  [[targets[name]] for name in joints],
                                  [[duration] for name in joints], True)

    print("Volviendo a StandInit...")
    if not posture.goToPosture("StandInit", 0.5):
        raise RuntimeError("El show termino, pero fallo la postura final.")
    # No soltar rigidez automaticamente: un robot real podria caerse.
    print("Show terminado.")


def main():
    parser = argparse.ArgumentParser(description="Mini-show para el simulador NAO.")
    parser.add_argument("--port", type=int, default=59242,
                        help="Puerto del simulador (por defecto: 59242).")
    parser.add_argument("--mute", action="store_true", help="Ejecutar sin voz.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Mostrar la secuencia sin conexion ni movimientos.")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("El puerto debe estar entre 1 y 65535.")
    if args.dry_run:
        print("VISTA PREVIA: no se conecta ni mueve nada.")
        for index, (label, phrase, duration, targets) in enumerate(STEPS, 1):
            print("%02d. %s (%s s)" % (index, label, duration))
            if phrase and not args.mute:
                print("    Voz: %s" % phrase)
        print("Al inicio y al final: postura StandInit. No hay caminata.")
        return 0

    print("SOLO SIMULADOR: se moveran cabeza y brazos, sin caminar.")
    if sys.version_info[0] != 2:
        if not os.path.isfile(PYTHON27):
            print("No se encontro Python 2.7 en %s" % PYTHON27)
            return 1
        env = os.environ.copy()
        env["PATH"] = SDK_DIR + os.pathsep + env.get("PATH", "")
        # Ruta absoluta para funcionar desde cualquier directorio.
        return subprocess.call([PYTHON27, os.path.abspath(__file__)] +
                               sys.argv[1:], env=env)

    os.environ["PATH"] = SDK_DIR + os.pathsep + os.environ.get("PATH", "")
    try:
        run_show(args.port, args.mute)
    except KeyboardInterrupt:
        print("\nShow interrumpido; el gesto actual podria seguir hasta terminar.")
        return 130
    except Exception as exc:
        print("Error: %s" % exc)
        print("Revisa que el simulador este abierto y que el puerto sea correcto.")
        print("Si el error menciona DLL, revisa el SDK y Visual C++ 2010 x86.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
