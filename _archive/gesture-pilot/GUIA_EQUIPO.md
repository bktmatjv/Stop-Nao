# Guía del equipo — NAO Gesture Pilot

Práctica Calificada 2 · Inteligencia Artificial (1ASI0404) · Sección 9138
**Entrega: viernes 13/11/2026** — informe PDF + video de 3 a 5 minutos.

> Lean primero la sección 1 (qué hace el sistema) y la 2 (cómo funciona por dentro).
> Después cada uno va directo a su parte en la sección 4.

---

## 1. ¿Qué hace el programa?

Una persona hace un gesto con el cuerpo frente a la cámara de la laptop. El sistema:

1. **Ve** a la persona (MediaPipe detecta 33 puntos del cuerpo).
2. **Reconoce el gesto** (modelo 1: Random Forest o SVM).
3. **Lee el entorno** del robot: sonares (distancia a obstáculos), si se está moviendo y qué tan lejos está la persona.
4. **Decide qué hacer** (modelo 2: Árbol de decisión o Random Forest), no obedece a ciegas.
5. **Aplica una regla de seguridad**: si hay un obstáculo a menos de 0.30 m, nunca avanza.
6. **Ordena al NAO** la acción: caminar, girar, detenerse, advertir por voz, etc.

| Gesto | Cómo se hace | Qué significa |
|---|---|---|
| `ADVANCE` | Ambos brazos arriba de la cabeza | Avanzar |
| `STOP` | Antebrazos cruzados frente al pecho | Detenerse |
| `TURN_LEFT` | **Tu** brazo izquierdo extendido al costado | Girar a la izquierda |
| `TURN_RIGHT` | **Tu** brazo derecho extendido al costado | Girar a la derecha |
| `NONE` | Brazos relajados, movimientos naturales | Nada (el robot no reacciona) |

| Acción del robot | Qué hace el NAO |
|---|---|
| `ADVANCE` | Camina 20 cm hacia adelante |
| `TURN_LEFT` / `TURN_RIGHT` | Gira unos 45° |
| `STOP` | Se detiene |
| `WARN_OBSTACLE` | Se detiene, dice "Hay un obstáculo, no puedo avanzar" y pone los ojos en rojo |
| `ASK_CLOSER` | Dice "Acércate, no te veo bien" |
| `WAIT` | No hace nada |

**El caso estrella del video:** la persona hace `ADVANCE`, pero hay una caja delante del robot → el NAO **no avanza**, avisa por voz y se le ponen los ojos rojos. Eso demuestra que la decisión depende del contexto y no solo del gesto.

---

## 2. ¿Cómo funciona por dentro?

### 2.1 Arquitectura

```
 Webcam laptop
      │ imagen
      ▼
┌──────────────────────── Python 3.11 (el "cerebro") ────────────────────────┐
│  MediaPipe Pose ──► 33 puntos del cuerpo                                   │
│        │                                                                   │
│        ▼                                                                   │
│  Features (28 números) ──► MODELO 1: gesto (RF / SVM) ──► voto de 7 frames │
│                                         │                                  │
│        sonares + estado del robot ──────┤                                  │
│                                         ▼                                  │
│                         MODELO 2: acción (Árbol / RF)                      │
│                                         │                                  │
│                                         ▼                                  │
│                         Capa de seguridad (regla fija)                     │
└─────────────────────────────────────────┬──────────────────────────────────┘
                                          │ JSON por stdin/stdout
                                          ▼
┌──────────────── Python 2.7 (el "puente", bridge/nao_bridge.py) ────────────┐
│  naoqi: ALMotion, ALTextToSpeech, ALMemory (sonares), ALLeds               │
└─────────────────────────────────────────┬──────────────────────────────────┘
                                          │ WiFi
                                          ▼
                                      Robot NAO
```

**¿Por qué dos programas de Python?** El SDK del NAO (`naoqi`) solo funciona en **Python 2.7**, y MediaPipe y scikit-learn necesitan **Python 3**. No pueden convivir en un mismo proceso. El profesor resolvió esto en `naoooo/py27-py3.py` lanzando Python 2.7 desde Python 3 con `subprocess`, pero abre un proceso nuevo **por cada comando** y eso tarda. Nosotros adaptamos su idea: el puente en Python 2.7 se abre **una sola vez** y queda escuchando órdenes. Así la conexión con el robot se mantiene abierta y podemos leer los sonares continuamente. *(En el informe hay que citar el código del profesor como base.)*

### 2.2 ¿Dónde hay redes neuronales y dónde no?

| Componente | ¿Qué es? | ¿Lo entrenamos nosotros? |
|---|---|---|
| **MediaPipe Pose (BlazePose)** | **Red neuronal convolucional** de Google, en dos etapas: un detector encuentra a la persona y una segunda red estima las coordenadas de 33 puntos del cuerpo | **No.** Viene preentrenada; la usamos como "extractor de características" |
| **Modelo 1: Random Forest vs SVM** | Machine Learning **clásico** (no son redes neuronales) | **Sí**, con nuestros propios datos |
| **Modelo 2: Árbol de decisión vs Random Forest** | Machine Learning **clásico** | **Sí**, con escenarios reales etiquetados por nosotros |
| **Capa de seguridad** | Regla fija (`if`), no es IA | — |

**¿Por qué no entrenamos una red neuronal propia?** Este argumento va en el Marco Teórico:
- Tenemos pocos datos (4 personas). Una red neuronal desde cero necesita miles de ejemplos variados y se sobreajustaría.
- Los datos ya son **tabulares** (28 números por imagen), y ahí RF y SVM rinden muy bien.
- Corren en tiempo real en CPU, sin GPU.
- Son **interpretables**: el árbol de decisión se puede graficar y explicar en el informe.
- Aprovechamos la red neuronal de MediaPipe para lo difícil (ver el cuerpo en una imagen), y el ML clásico para lo específico de nuestro problema.

### 2.3 Modelo 1: reconocer el gesto

**Entrada:** los 33 puntos de MediaPipe (x, y, z, visibilidad).
**Preprocesamiento** (`brain/gestures/features.py`), que produce 28 features:
1. **Centrar** las coordenadas en el centro de la cadera (o de los hombros, si la cadera no se ve). Así da igual dónde esté la persona en la imagen.
2. **Escalar** dividiendo por el ancho de hombros. Así da igual si la persona está cerca o lejos.
3. **Ángulos**: codo (brazo doblado o estirado) y hombro (brazo abajo = 0°, horizontal = 90°, arriba = 180°).
4. **Banderas**: ¿la muñeca está sobre el hombro? ¿Sobre la nariz? Distancia entre las muñecas.
5. **Valores faltantes**: si un codo o una muñeca sale del cuadro, su coordenada se pone en 0 y una bandera `*_visible = 0` le avisa al modelo. Solo se descarta el frame si no se ven los hombros.

**Modelos comparados:**
- **Random Forest:** muchos árboles de decisión, cada uno entrenado con una muestra distinta de los datos. Votan, y gana la mayoría. Hiperparámetros: `n_estimators` (cantidad de árboles), `max_depth` (profundidad), `min_samples_leaf`.
- **SVM con kernel RBF:** busca la frontera que separa las clases con el mayor margen posible. El kernel RBF permite fronteras curvas. Hiperparámetros: `C` (cuánto castiga los errores) y `gamma` (qué tan "local" es la frontera). Necesita datos escalados (`StandardScaler`).

**Cómo se evalúa (importante para el informe):**
- **GroupKFold por persona:** el modelo se evalúa con una persona que **nunca vio** al entrenar. Si mezcláramos frames de la misma persona en entrenamiento y prueba, el accuracy saldría cerca del 99% y sería mentira, porque dos frames seguidos de un video son casi idénticos.
- `GridSearchCV` prueba combinaciones de hiperparámetros y elige con la métrica **F1-macro** (promedia el F1 de cada clase, así ninguna clase queda ignorada).
- Gráficas generadas: matrices de confusión de RF y SVM, curva de aprendizaje, distribución de clases y comparación entre modelos.

**En tiempo real** (`brain/gestures/predictor.py`): se vota sobre los últimos 7 frames para evitar parpadeos, y si la confianza es baja el gesto cuenta como `NONE`.

### 2.4 Modelo 2: decidir la acción según el contexto

**Entrada:** gesto, confianza del gesto, sonar izquierdo, sonar derecho, distancia a la persona y si el robot se está moviendo.
**Salida:** una de las 7 acciones.

**El dataset es lo que lo hace valioso:** ponemos al robot en escenarios reales (con una caja a 20 cm, sin obstáculos, con la persona lejos, etc.), el programa registra lo que miden los sensores y **un humano decide qué debería hacer el robot**. Así el modelo aprende de decisiones humanas y no de reglas que escribimos nosotros. Si solo usáramos reglas, el modelo se limitaría a memorizarlas.

**Modelos comparados:** Árbol de decisión (se puede graficar en el informe) vs Random Forest. La división de datos se hace **por escenario**, por el mismo motivo que la división por persona del modelo 1.

**Capa de seguridad** (`brain/context/safety.py`): pase lo que pase, si el modelo dice `ADVANCE` y algún sonar marca menos de 0.30 m, o da una lectura inválida, la acción se cambia a `WARN_OBSTACLE`. En robótica real ningún modelo de ML tiene la última palabra sobre la seguridad.

### 2.5 Estructura de carpetas

```
pc2/
├── bridge/nao_bridge.py        Puente Python 2.7 ↔ NAO (naoqi)
├── brain/
│   ├── nao_client.py           Cliente Python 3 que habla con el puente
│   ├── main.py                 Programa principal (cámara → gesto → decisión → robot)
│   ├── gestures/               Modelo 1: capture, features, train, predictor
│   └── context/                Modelo 2: collect, train, safety, decider
├── data/gestures/              CSV de gestos grabados (los generan ustedes)
├── data/context/scenarios.csv  Escenarios con el robot (los generan ustedes)
├── models/                     Modelos entrenados (.joblib)
├── report/
│   ├── informe_borrador.md     Borrador del informe con [[PENDIENTE]]
│   ├── guion_video.md          Guion del video
│   ├── figures/                Gráficas generadas al entrenar
│   └── metrics/                Métricas en JSON
├── runs/                       Registro CSV de cada decisión del robot
└── tests/                      Tests automáticos (pytest)
```

---

## 3. Preparación (una sola vez, en la PC del proyecto)

Abrir PowerShell en la carpeta `pc2`:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv311\Scripts\Activate.ps1
python -m pytest -q          # tiene que decir "83 passed"
```

Requisitos que ya están instalados en la PC de Matías: Python 3.11 (`.venv311`), Python 2.7 en `C:\Python27` con el SDK `pynaoqi`.

---

## 4. Tareas por integrante

> El reparto es una **propuesta**: pueden intercambiar roles. Lo importante es que los cuatro aparezcan grabando gestos y hablando en el video.

### 👤 Matías — Robot e integración

1. **Reservar tiempo con Zyra** (el robot real, `192.168.108.90:9559`). Es el recurso más escaso y vale el 35% de la nota.
2. **Primera conexión**, con alguien al lado del robot y el piso despejado:
   ```powershell
   python -m brain.main --robot zyra
   ```
   Al arrancar, el robot se pone rígido y va a la postura StandInit; la primera respuesta tarda unos segundos. **El modo real nunca se probó**: si falla, copiar el error completo (incluye el stderr del puente) y revisarlo.
3. Verificar a mano que funcionen: los sonares (sensores del pecho), la voz en español, los LEDs de los ojos y un paso corto.
4. Ajustar los umbrales si hace falta, anotando antes y después para el informe: `FAR_PERSON = 0.08` en `brain/main.py`, el umbral del sonar de 0.30 m en `brain/context/safety.py` y el cooldown con `--cooldown`.
5. Demo final mientras se graba el video. Cada decisión queda en `runs/*.csv` con su latencia.

### 👤 Vanessa — Dataset de gestos y Modelo 1

1. Coordinar que **los 4 integrantes** graben los 5 gestos. Cada uno ejecuta, cambiando `--person` por su nombre:
   ```powershell
   python -m brain.gestures.capture --person vanessa --label ADVANCE    --seconds 20
   python -m brain.gestures.capture --person vanessa --label STOP       --seconds 20
   python -m brain.gestures.capture --person vanessa --label TURN_LEFT  --seconds 20
   python -m brain.gestures.capture --person vanessa --label TURN_RIGHT --seconds 20
   python -m brain.gestures.capture --person vanessa --label NONE       --seconds 20
   ```
   **Reglas de grabación:**
   - Parados, a **unos 2 m** de la cámara, que se vea de la cabeza a la cintura **con los brazos extendidos dentro del cuadro**.
   - Si aparece el aviso rojo **"FUERA DE CUADRO"**, retrocedan.
   - Al final, el programa avisa si menos de la mitad de los frames salieron completos: en ese caso, repitan.
   - Variar fondo, ropa e iluminación. Moverse un poco durante el gesto.
   - En `NONE`, hagan cosas naturales: rascarse, mirar el celular, caminar.
2. Entrenar (sin `--quick`; puede tardar varios minutos):
   ```powershell
   python -m brain.gestures.train --data data/gestures --out models/gesture_model.joblib
   ```
3. Revisar `report/metrics/gesture_metrics.json` y las gráficas `report/figures/gesture_*.png`. Pasarle a Vivianne: accuracy, F1-macro, mejores hiperparámetros y cuál ganó (RF o SVM).
4. Para el modelo que usa el robot, entrenar una versión final con todas las personas, agregando `--refit-all`.

### 👤 Facundo — Escenarios con el robot y Modelo 2

1. Con Zyra (coordinado con Matías), preparar **entre 15 y 20 escenarios distintos**. Por ejemplo: sin obstáculo, una caja a 20 cm, a 50 cm o a 1 m, la persona cerca o lejos, el robot quieto o caminando.
2. En cada escenario, registrar varias filas con un **id distinto por escena**:
   ```powershell
   python -m brain.context.collect --scenario-id s01 --robot zyra
   ```
   El programa lee los sonares del robot y pregunta el gesto, la confianza y la distancia. Después **ustedes eligen qué debería hacer el robot**. Decidan como lo haría una persona sensata, no copiando una regla.
3. Entrenar:
   ```powershell
   python -m brain.context.train --data data/context/scenarios.csv
   ```
   Opcional: aumentar los datos con sintéticos (`python -m brain.context.make_synthetic --out data/context/synthetic.csv` y después `--extra data/context/synthetic.csv`). **Si lo usan, el informe tiene que decirlo claramente.**
4. Pasarle a Vivianne `report/metrics/context_metrics.json` y las gráficas `context_*.png`. La gráfica del árbol (`context_tree.png`) es muy buena para el informe.

### 👤 Vivianne — Informe y video

1. Partir de `report/informe_borrador.md`, que ya sigue la estructura del Google Doc. Pasarlo al Google Doc con el formato del enunciado: **Times New Roman 12, interlineado 1.5, márgenes de 2.5 cm, entre 8 y 15 páginas**.
2. Reemplazar cada `[[PENDIENTE: ...]]` con los números reales que pasen Vanessa, Facundo y Matías. **No inventar ninguna cifra.**
3. Pendientes conocidos del borrador:
   - Está largo (unas 5.900 palabras): recortar el Marco Teórico y la Discusión para no pasar de 15 páginas.
   - En la sección 3c, cambiar "los frames con baja visibilidad se descartan" por la **imputación con banderas** (ver 2.3, punto 5 de esta guía).
   - Verificar la referencia Gouaillier et al. (2009), a la que le faltan las páginas y el DOI.
   - Mínimo 5 referencias en APA 7, sin Wikipedia.
4. **Video** (guion en `report/guion_video.md`, unos 4 minutos):
   - Micrófono externo (el audio malo baja la nota a 75%).
   - Dos tomas: el robot y la persona, más la grabación de la pantalla de la laptop con el overlay.
   - Tiene que verse el **robot real** (si es solo simulación, ese criterio queda en 50%).
   - Hablan los cuatro integrantes.

---

## 5. Probar sin el robot (cualquiera)

Necesita un modelo de gestos entrenado (sección 4, Vanessa, pasos 1 y 2).

```powershell
python -m brain.main --mock
```

- Hacer gestos: el overlay muestra el gesto, la confianza, los sonares y la acción.
- Tecla **`m`**: pone un obstáculo simulado a 0.2 m. Hacer `ADVANCE` → tiene que salir `WARN_OBSTACLE`.
- Tecla **`q`**: salir.
- Sin modelo de contexto, el programa avisa que usa reglas de respaldo: es normal.

---

## 6. Problemas conocidos y soluciones

| Problema | Causa | Solución |
|---|---|---|
| `gesture model ... not found` | No se entrenó el modelo 1 | Grabar gestos y entrenar (sección 4, Vanessa) |
| `dropped N of N frames` | La persona estaba muy cerca y no se veían los hombros | Retroceder a ~2 m y volver a grabar |
| El robot nunca avanza en Nyx o en el simulador | El robot virtual probablemente no tiene sonares reales; la seguridad trata un 0 como peligro | Usar Zyra o `--mock` |
| El NAO habla en inglés | No tiene el paquete de idioma español | Se puede seguir; avisarlo en el video o en el informe |
| Cambió la IP de Zyra | El programa tiene la IP fija en `bridge/nao_bridge.py` (diccionario `ROBOTS`) | Editar la IP ahí (pendiente: agregar `--ip/--port` a `brain.main`) |
| `NaoBridgeError ... did not answer` | El robot está apagado, en otra red o el puerto es incorrecto | Verificar que la laptop esté en la misma WiFi y que haya `ping` al robot |

---

## 7. Seguridad con el robot

- Siempre alguien **al lado del NAO**, listo para sostenerlo.
- Piso despejado, sin mesas ni bordes cerca.
- Batería cargada antes de grabar.
- Al cerrar el programa (`q` o Ctrl+C), el robot recibe `STOP` automáticamente, pero **no se suelta la rigidez**, para que no se caiga.
