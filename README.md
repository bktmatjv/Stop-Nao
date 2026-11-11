# NAO: control por gestos de mano con validación del entorno

Un robot NAO que lee **gestos de la mano** desde una webcam y ejecuta el movimiento correspondiente, pero solo después de revisar su entorno. Si el movimiento no es seguro, el robot se niega, explica el motivo por voz y muestra la razón con el color de sus ojos.

**Ejemplo:** pulgar arriba significa "avanza 35 cm". Si hay un obstáculo frente al robot, no avanza, lo explica y sus ojos se ponen rojos.

Proyecto del curso Inteligencia Artificial (1ASI0404).

---

## Inicio rápido

1. Crear el entorno de Python 3.11 e instalar las dependencias (ver [Instalación](#instalación)).
2. Ejecutar el sistema en simulación: `python -m brain.main --mock` (ver [Ejecución del sistema](#ejecución-del-sistema)).
3. Ejecutar las pruebas: `python -m pytest -q`.

---

## Arquitectura

```
webcam
  │
  ▼
MediaPipe Hands                    preentrenado por Google (no entrenado por el equipo)
  │  21 puntos de la mano
  ▼
Modelo de gestos                   Random Forest o SVM, entrenado por el equipo
  │  etiqueta + confianza
  ▼
Modelo de decisión por contexto    Árbol de decisión o Random Forest, entrenado por el equipo
  │  acción propuesta
  ▼
Capa de seguridad                  reglas fijas; tiene la última palabra
  │  acción permitida, color de ojos, motivo hablado
  ▼
Cliente NAO  →  puente NAOqi (Python 2.7, proceso persistente)  →  robot NAO
```

- **MediaPipe Hands** es la única red neuronal del flujo. Se usa preentrenada.
- **Modelo de gestos** (RF o SVM) clasifica los 21 puntos en una de 7 etiquetas. Las predicciones se suavizan en una ventana, y con baja confianza el robot pide repetir el gesto.
- **Modelo de decisión por contexto** (DT o RF) aprende de escenarios etiquetados por personas (sonares, postura, movimiento, tamaño de la mano).
- **Capa de seguridad** aplica umbrales fijos. El modelo puede proponer una acción, pero la capa de seguridad puede corregirla. Siempre se ejecuta al final.
- **Puente.** La API de NAOqi solo funciona en Python 2.7. El programa principal (Python 3.11) inicia el puente como subproceso persistente y se comunica con él mediante líneas JSON. El patrón parte del material del curso, `naoooo/py27-py3.py`.

---

## Gestos, movimientos y comprobaciones

| Gesto | Movimiento del robot | Qué comprueba antes |
|-------|----------------------|---------------------|
| `THUMBS_UP` (pulgar arriba) | Avanzar 35 cm | Que no haya obstáculo al frente. Si hay uno cerca, avanza solo la distancia segura. |
| `OPEN_PALM` (palma abierta) | Detener movimiento y música | Nada: siempre se permite. |
| `POINT_LEFT` (señalar a la izquierda) | Girar hacia ese lado | Que el sonar de ese lado esté libre. |
| `POINT_RIGHT` (señalar a la derecha) | Girar hacia ese lado | Que el sonar de ese lado esté libre. |
| `VICTORY` (dos dedos en V) | Baile festivo y canción aleatoria de `songs/` | Que esté de pie y quieto, y que la persona no esté demasiado cerca. |
| `FIST` (puño cerrado) | Sentarse, o pararse si ya está sentado | Que no se esté moviendo. |
| `NONE` (ninguno) | Ninguno | — |

**Señalar.** La persona mira al robot, así que el robot gira *hacia donde señala la persona*. Su izquierda es la derecha del robot: `POINT_LEFT` hace girar al robot a su derecha. Esto lo controla la constante `POINT_TURNS_TOWARD_POINTED_SIDE = True`.

**Avance reducido.** Si hay un obstáculo cerca pero fuera del margen de seguridad, el robot avanza solo la distancia segura (mínimo 5 cm). Si no hay distancia útil, se niega.

**Gesto poco claro.** Si la confianza es menor a 0.6, el robot pide repetir el gesto en lugar de actuar.

**Sin movimiento hacia atrás.** NAO no tiene sensor trasero, así que nunca retrocede.

---

## Colores de los ojos

| Color | Significado |
|-------|-------------|
| Verde | Avanzar o orden aceptada |
| Amarillo | Dudoso o limitado (avance corto, confirmación) |
| Rojo | Rechazado por seguridad (obstáculo o persona muy cerca) |
| Azul | En espera, escuchando |
| Magenta | Modo VICTORY (baile y música) |
| Cian | Giro a la izquierda o a la derecha |
| Blanco | Cambio de postura (sentarse o ponerse de pie) |

---

## Instalación

**Requisitos:** Windows, una webcam, Python 3.11 para el programa principal y Python 2.7 con `pynaoqi` solo para el puente.

```powershell
# 1. Entorno principal (Python 3.11)
py -3.11 -m venv .venv311
.venv311\Scripts\activate
pip install -r requirements.txt

# 2. Puente (Python 2.7 con pynaoqi)
#    Instalar Python 2.7 en C:\Python27 y el SDK de NAOqi (pynaoqi) para esa versión.
#    El programa principal inicia el puente automáticamente.
```

---

## Estructura del proyecto

| Ruta | Contenido |
|------|-----------|
| `brain/gestures/` | Captura con MediaPipe, características, entrenamiento y predicción del gesto |
| `brain/context/` | Intenciones, características de contexto, capa de seguridad, modelo de decisión y recolección de escenarios |
| `brain/main.py` | Punto de entrada: webcam, decisiones, ejecución y registro |
| `brain/music.py` | Reproducción de canciones para el gesto `VICTORY` |
| `brain/nao_client.py` | Cliente que envía comandos al puente |
| `bridge/` | Puente Python 2.7 con NAOqi (JSON por socket) |
| `data/gestures/` | Muestras de gestos grabadas |
| `data/context/` | Escenarios de contexto con el robot real |
| `models/` | Modelos entrenados (`.joblib`) |
| `report/` | Borrador del informe, guion del video, métricas y figuras |
| `runs/` | Registros de ejecución |
| `tests/` | Pruebas automatizadas (`pytest`) |
| `naoooo/` | Material de referencia del curso (no es código del proyecto) |
| `_archive/` | Proyectos anteriores, conservados como referencia (no los usa el sistema) |

---

## Comandos

Grabar muestras de un gesto (una persona, una etiqueta, 15 segundos):

```powershell
python -m brain.gestures.capture --person <identificador> --label <ETIQUETA> --seconds 15
```

Entrenar el modelo de gestos:

```powershell
python -m brain.gestures.train --data data/gestures --out models/gesture_model.joblib
```

Recolectar escenarios de contexto con el robot real (Zyra):

```powershell
python -m brain.context.collect --scenario-id <id> --robot zyra
```

Entrenar el modelo de decisión por contexto:

```powershell
python -m brain.context.train --data data/context/scenarios.csv --out models/context_model.joblib
```

Ejecutar las pruebas:

```powershell
python -m pytest -q
```

---

## Ejecución del sistema

```powershell
# Simulación: sin robot, con sonares y postura simulados
python -m brain.main --mock

# Robot real (Zyra)
python -m brain.main --robot zyra

# Robot virtual de Choregraphe (el puerto cambia cada vez que se abre Choregraphe)
python -m brain.main --robot nyx --ip 127.0.0.1 --port <puerto>
```

El sistema necesita `models/gesture_model.joblib`. Si no existe, termina con código 2 e indica cómo entrenarlo. Si falta `models/context_model.joblib`, avisa y usa solo las reglas por gesto y la capa de seguridad.

| Opción | Valor por defecto | Descripción |
|--------|-------------------|-------------|
| `--mock` | | Puente sin NAOqi ni robot; sonares y postura simulados |
| `--robot` | | `zyra`, `nyx` o `sim` (obligatorio salvo con `--mock`) |
| `--ip`, `--port` | perfil del robot | Reemplazar la dirección del robot (Choregraphe) |
| `--camera` | `0` | Índice de la webcam |
| `--gesture-model` | `models/gesture_model.joblib` | Modelo de gestos |
| `--context-model` | `models/context_model.joblib` | Modelo de contexto (opcional) |
| `--cooldown` | `2.5` | Segundos entre dos órdenes al robot |
| `--confirm-window` | `4.0` | Segundos para repetir el gesto cuando el robot pide confirmación |
| `--log` | `runs/<fecha>.csv` | CSV con una fila por decisión |
| `--no-video` | | Sin ventana de vista previa (salir con Ctrl+C) |
| `--virtual-sonar` | | Solo con Choregraphe en `127.0.0.1`: simula los sonares (ver abajo) |

**Qué ocurre.** Cuando aparece un gesto estable nuevo y pasó el tiempo de espera, el robot lee sus sensores, el modelo de contexto y la capa de seguridad deciden, y el robot se mueve, cambia el color de sus ojos y dice el motivo. Un gesto sostenido se ejecuta una sola vez. `NONE` nunca actúa. Si la confianza es baja, el robot pide repetir el gesto: si se repite con confianza de al menos 0.6 dentro de `--confirm-window`, se ejecuta aunque no haya pasado la espera entre órdenes. Si no, la petición caduca y los ojos vuelven a azul.

**Vista previa.** Imagen de la cámara en espejo con los puntos de la mano, gesto y confianza, sonares en cm, postura, movimiento, última decisión con su motivo, color de los ojos, barra de espera, aviso de confirmación pendiente y, con un árbol de decisión como modelo de contexto, las primeras líneas de su explicación.

**Teclas.** `q` cierra el programa (Ctrl+C también). Solo con `--mock`, estas teclas simulan el entorno:

| Tecla | Escenario simulado |
|-------|--------------------|
| `1` | Frente libre (sonares a 1.5 m) |
| `2` | Obstáculo al frente a 0.40 m (avance reducido de 10 cm) |
| `3` | Obstáculo al frente a 0.20 m (se niega a avanzar) |
| `l` / `r` | Activar o quitar un obstáculo a 0.20 m en el sonar izquierdo / derecho |
| `s` | Cambiar la postura simulada entre sentado y de pie |

**Choregraphe.** El robot virtual no tiene sonares reales y reporta 0.0 m. Por seguridad, la capa de seguridad lo trata como lectura no válida y se niega a avanzar y a girar. Con `--virtual-sonar` se reemplazan esas lecturas por valores simulados, controlados con las teclas `1`, `2`, `3`, `l` y `r`. Las decisiones quedan marcadas como `simulated` en el registro. Esta opción **solo funciona con `127.0.0.1`**: con Zyra se rechaza, porque sus sensores reales nunca se reemplazan.

**Al terminar.** Con `q`, Ctrl+C o ante un error, el programa siempre envía `stop`, pone los ojos en azul, cierra el puente y libera la cámara. No suelta la rigidez de los motores.

**Registro.** Cada decisión queda en un CSV dentro de `runs/` con estas columnas: `timestamp, gesture, confidence, sonar_left, sonar_right, hand_size, moving, standing, requested_intent, action, params, reason, eye_color, overridden, bridge_ok, simulated, latency_ms`.

---

## Seguridad

- La capa de seguridad tiene la última palabra. Ninguna salida del modelo puede saltársela.
- Umbrales fijos: distancia segura al frente 0.30 m, distancia segura al costado 0.25 m, avance completo 0.35 m, avance mínimo 0.05 m, giro de 0.785 rad (45°).
- Las lecturas de sonar inválidas (cero, negativas o NaN) se tratan como inseguras.
- Los comandos de movimiento están limitados: `0 <= x <= 0.30` m y `|theta| <= 1.6` rad. No se mueve si el robot no está de pie.
- `OPEN_PALM` (detener) siempre está permitido.
- Las pruebas automatizadas usan solo `--mock`. Las pruebas con el robot real las hace el equipo, con supervisión.

---

## Estado

| Parte | Estado |
|-------|--------|
| Pipeline de gestos (captura, características, entrenamiento, predicción) | En progreso |
| Puente NAOqi y cliente | En progreso |
| Modelo de contexto y capa de seguridad | En progreso |
| `brain/main.py` (webcam, ejecución, confirmación, vista previa, registro CSV) | Terminado; probado solo en `--mock` |
| Puente con el robot real | Pendiente de prueba en el robot |

---

## Notas del repositorio

- **Música.** Los archivos de audio no se versionan. Coloque sus canciones en `songs/` (`.wav` o `.mp3`).
- **Registro interno.** La carpeta `odd/` (planes y registro de tareas del flujo de trabajo) no se versiona.
- **Créditos.** El patrón de subproceso Python 3 → Python 2.7 está tomado de `naoooo/py27-py3.py`, material del curso.
