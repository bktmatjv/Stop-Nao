# Archivo: NAO Gesture Pilot (versión con pose corporal)

Primera versión del proyecto del curso Inteligencia Artificial (1ASI0404). El robot reconocía **gestos corporales** con MediaPipe Pose, decidía la acción según el contexto (distancia de sonares, distancia de la persona y movimiento) y aplicaba una capa de seguridad.

## Estado

**Archivado.** El proyecto pasó a reconocer **gestos de la mano** con MediaPipe Hands. Los datos de pose corporal no son compatibles con las características de la mano, por lo que este código se conserva solo como referencia y no lo usa el sistema actual.

## Qué incluye

| Ruta | Contenido |
|------|-----------|
| `brain/` | Captura, características, modelos y decisión de la versión con pose |
| `bridge/` | Puente Python 2.7 con NAOqi |
| `data/`, `models/`, `runs/` | Datos, modelo y registros de esa versión |
| `report/` | Borrador de informe, guion de video, métricas y figuras |
| `tests/` | Pruebas de esa versión |

## Arquitectura de referencia

- **Cerebro (Python 3.11):** visión, características, modelos y lógica de decisión.
- **Puente (Python 2.7):** proceso persistente que importa `naoqi.ALProxy` y responde comandos en JSON por línea.
- **Capa de seguridad:** corrige cualquier `ADVANCE` cuando un sonar está por debajo de 0.30 m (`WARN_OBSTACLE`).

## Uso como referencia

No es necesario instalar nada de esta carpeta para el proyecto actual. Si se quiere revisar, las pruebas pueden ejecutarse desde el entorno `.venv311` con `pytest`, siempre en modo `--mock`.
