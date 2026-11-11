# Archivo: NAO Dance AI

Proyecto del curso Inteligencia Artificial (1ASI0404) en el que el robot NAO escucha una canción, estima su tempo y energía, predice un estilo de movimiento para cada segmento con un modelo de aprendizaje automático y ejecuta una coreografía de brazos y cabeza ajustada al tempo.

Estilos de movimiento: `SOFT`, `MODERATE` y `ENERGETIC`.

## Estado

**Archivado el 10/11/2026.** El equipo cambió el enfoque del proyecto hacia el reconocimiento de gestos de la mano (ver la raíz del repositorio). Este código se conserva como referencia y no lo usa el sistema actual.

## Qué incluye

| Ruta | Contenido |
|------|-----------|
| `dance/audio/` | Extracción de características de audio, anotación y construcción del conjunto de datos |
| `dance/model/` | Entrenamiento, línea base e inferencia de estilos |
| `dance/robot/` | Coreografías y control del robot |
| `bridge/` | Puente Python 2.7 con NAOqi |
| `data/dataset/`, `data/labels/` | Segmentos y etiquetas anotadas |
| `models/` | Modelo preliminar |
| `runs/` | Registros de ejecución |
| `tests/` | Pruebas automatizadas |

## Notas

- **Audio no incluido.** Las canciones de `data/songs/` y `data/demo_songs/` no se versionan por derechos de autor. Para reproducir el proyecto, coloque archivos propios con el mismo formato.
