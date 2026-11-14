# Guía del equipo: NAO con gestos de mano y validación del entorno

Inteligencia Artificial (1ASI0404) · Sección 9138
**Entrega:** viernes 13/11/2026: informe en PDF y video de 3 a 5 minutos.

> Leer primero las secciones 1 y 2. Después, ir a la sección del rol asignado dentro de la sección 6.

---

## 1. ¿Qué hace el sistema?

Un robot NAO que **razona antes de obedecer**. La persona hace un gesto con la mano frente a la webcam del computador, y el robot:

1. **Reconoce el gesto** (por ejemplo, pulgar arriba).
2. **Revisa el entorno** antes de moverse: obstáculos al frente o a los lados, si ya se está moviendo, si está sentado o de pie, y qué tan cerca está la persona.
3. **Decide**: ejecuta el movimiento, lo limita (avanza solo una distancia segura) o lo rechaza.
4. **Explica** su decisión en voz alta y con el color de sus ojos.
5. **Pide confirmación** si no está seguro del gesto: en lugar de actuar, le pide a la persona que lo repita.

Ejemplo: pulgar arriba significa "avanza 35 cm". Si hay una caja a 20 cm frente al robot, no avanza, dice por qué y sus ojos se ponen rojos.

## 2. ¿Cómo funciona por dentro?

Hay dos tipos de modelos y una capa de reglas fijas:

| Parte | Qué es | ¿Red neuronal? |
|-------|--------|----------------|
| Detección de la mano (MediaPipe Hands) | Modelo preentrenado por Google. El equipo **no lo entrena**. Devuelve 21 puntos de la mano. | **Sí** (preentrenada) |
| Reconocimiento del gesto | Random Forest o SVM, entrenado por el equipo con sus datos. | No (ML clásico) |
| Decisión según el contexto | Árbol de decisión o Random Forest, entrenado con escenarios etiquetados por personas. | No (ML clásico) |
| Capa de seguridad | **Reglas fijas** con umbrales (distancias en metros). Tiene la última palabra: aunque el modelo diga "avanza", si hay un obstáculo, no avanza. | No (reglas) |

Idea clave: el modelo propone, la **regla de seguridad decide**.

## 3. Gestos, movimientos y comprobaciones

El sistema tiene dos ejes independientes: **modo** (MOVIMIENTO o MÚSICA) y **entrada** (GESTOS o VOZ). El mismo gesto significa cosas distintas según el modo.

### 3.1 Gestos en modo MOVIMIENTO

| Gesto | Cómo se hace | Movimiento del robot | Qué comprueba el robot |
|-------|--------------|----------------------|------------------------|
| `THUMBS_UP` (pulgar arriba) | Puño con el pulgar hacia arriba | Avanzar 35 cm | Que no haya obstáculo al frente (si hay uno cerca, avanza solo la distancia segura) |
| `OPEN_PALM` (palma abierta) | Mano abierta, dedos juntos, palma a la cámara | Detener movimiento, baile, habla y música | Nada: siempre se permite |
| `POINT_LEFT` (señalar a tu izquierda) | Índice estirado apuntando a **tu** izquierda | Girar hacia ese lado (el robot gira a **su** derecha) | Que el sonar de ese lado esté libre |
| `POINT_RIGHT` (señalar a tu derecha) | Índice estirado apuntando a **tu** derecha | Girar hacia ese lado (el robot gira a **su** izquierda) | Que el sonar de ese lado esté libre |
| `FIST` (puño) | Puño cerrado | Sentarse, o pararse si ya está sentado | Que no se esté moviendo |
| `LOVE_YOU` (nueva) | Pulgar, índice y meñique extendidos | Celebrar: "We Are the Champions" y baile especial | Nada: funciona en ambos modos |
| `VICTORY` (dos dedos en V) | Índice y medio en V | Cambia la entrada entre GESTOS y VOZ | Siempre activo, también con voz |
| `NONE` (ninguno) | Mano relajada, movimientos naturales | Nada | No aplica |

### 3.2 Gestos en modo MÚSICA

| Gesto | Acción |
|-------|--------|
| `THUMBS_UP` | Reproducir o reanudar la música, con baile mientras suena |
| `OPEN_PALM` | Pausar música y baile |
| `POINT_RIGHT` | Canción siguiente |
| `POINT_LEFT` | Canción anterior |
| `FIST` | Detener la música |
| `LOVE_YOU` | Celebrar (igual que en MOVIMIENTO) |
| `VICTORY` | Cambia la entrada entre GESTOS y VOZ |

### 3.3 Controles que cambian el modo o la entrada

- **Dos pulgares arriba**, sostenidos unos 1 segundo: cambia el modo entre MOVIMIENTO y MÚSICA. Es un gesto con las dos manos.
- **`VICTORY`**: cambia la entrada entre GESTOS y VOZ.
- **`OPEN_PALM`**: parada de emergencia en cualquier modo y entrada.
- **Voz**: en entrada VOZ, se pulsa la tecla `v` de la vista previa para empezar a grabar y otra vez para terminar. Los LED de las orejas indican que la entrada es voz. Ejemplo de frase: "salúdame" (el saludo ya no es un gesto).

**Importante sobre señalar:** el robot gira **hacia donde señala la persona**. Como la persona mira al robot, su izquierda es la derecha del robot. Por eso `POINT_LEFT` hace girar al robot a su derecha. Esto está controlado por la constante `POINT_TURNS_TOWARD_POINTED_SIDE = True`.

**Gesto poco claro:** si el sistema no está seguro (confianza menor a 0.6), el robot pide repetir el gesto y no se mueve.

**Confusión LOVE_YOU / VICTORY:** estos dos gestos pueden confundirse. Si pasa, el robot puede celebrar cuando se quiso cambiar de entrada, o al revés. Con `--confirm-ambiguous` el robot pide repetir antes de actuar con cualquiera de los dos.

## 4. Colores de los ojos

| Color | Significado |
|-------|-------------|
| Verde | Avanzar o orden aceptada |
| Amarillo | Dudoso o limitado (avance corto por precaución, confirmación) |
| Rojo | Rechazado por seguridad (obstáculo detectado o persona muy cerca) |
| Azul | Modo MOVIMIENTO (en espera) |
| Morado | Modo MÚSICA |
| Cian | Giro a la izquierda o a la derecha |
| Blanco | Cambio de postura (sentarse o ponerse de pie) |

Los LED de las orejas se encienden cuando la entrada es VOZ.

## 5. Cómo ejecutar el sistema (`brain.main`)

Se ejecuta desde la carpeta del proyecto, con el entorno `.venv311` activado:

```powershell
# Simulación: sin robot, con sonares y postura simulados
python -m brain.main --mock

# Robot virtual de Choregraphe (el puerto cambia cada vez que se abre Choregraphe)
python -m brain.main --robot nyx --ip 127.0.0.1 --port <puerto>

# Robot real (Zyra), siempre con alguien al lado
python -m brain.main --robot zyra
```

Necesita el modelo de gestos `models/gesture_model.joblib`. Si no existe, el programa termina con código 2 e indica cómo entrenarlo (`python -m brain.gestures.train`). Si falta `models/context_model.joblib`, avisa y usa solo las reglas por gesto y la capa de seguridad.

**Opciones nuevas.** Las canciones y la clave de voz se configuran en `.env` (ver la sección 6b). Las opciones de línea de comandos tienen prioridad:

| Opción | Descripción |
|--------|-------------|
| `--music-dir <carpeta>` | Carpeta de canciones del modo MÚSICA (por defecto, `MUSIC_DIR` del `.env`, o `songs`) |
| `--champions-song <archivo>` | Audio para el gesto `LOVE_YOU` (por defecto, `CHAMPIONS_SONG` del `.env`) |
| `--confirm-ambiguous` | Pide repetir antes de actuar con `LOVE_YOU` o `VICTORY` |
| `--no-voice` | No activa la entrada por voz |

**Teclas.** `q` cierra el programa (Ctrl+C también). `v` inicia y detiene la grabación de voz (entrada VOZ). Solo con `--mock`, estas teclas simulan el entorno:

| Tecla | Escenario simulado |
|-------|--------------------|
| `1` | Frente libre (sonares a 1.5 m) |
| `2` | Obstáculo al frente a 0.40 m (el robot avanza solo 10 cm) |
| `3` | Obstáculo al frente a 0.20 m (el robot se niega a avanzar) |
| `l` / `r` | Activa o quita un obstáculo a 0.20 m en el sonar izquierdo / derecho |
| `s` | Cambia la postura simulada entre sentado y de pie |

**Con Choregraphe:** el robot virtual **no tiene sonares reales** (marca 0.0 m), así que, por seguridad, se niega a avanzar y a girar. Para verlo moverse, agregar `--virtual-sonar`:

```powershell
python -m brain.main --robot nyx --ip 127.0.0.1 --port <puerto> --virtual-sonar
```

Con esa opción, los sonares se simulan y las teclas `1`, `2`, `3`, `l` y `r` ponen obstáculos virtuales. Las decisiones quedan marcadas como `simulated` en el registro. Esta opción **solo funciona con un robot virtual en la propia PC** (`127.0.0.1`): con Zyra se rechaza, porque sus sensores reales nunca se reemplazan.

**Al terminar** (con `q`, Ctrl+C o ante un error), el programa siempre envía `stop`, pone los ojos en azul, cierra el puente y libera la cámara. No suelta la rigidez de los motores.

Cada decisión queda registrada en un CSV en `runs/` (gesto, confianza, sonares, acción, motivo, color de ojos, si la seguridad corrigió la orden, latencia).

## 6. Roles y pasos de entrenamiento

Cada rol tiene pasos numerados que se siguen en orden. Los roles pueden intercambiarse si el equipo está de acuerdo.

**Orden general del trabajo:** primero el rol 2 graba los gestos y entrena el modelo; el rol 3 recolecta los escenarios con Zyra; el rol 1 integra y prueba el sistema; el rol 4 redacta el informe con las métricas finales.

### Rol 1: Robot e integración

1. Conectar con el **robot virtual de Choregraphe**. El puerto aparece en la ventana de conexión o en *Preferences → Virtual Robot*, y cambia cada vez que se abre Choregraphe.
2. Probar el sistema completo en simulación: `python -m brain.main --mock`.
3. Probar con el robot virtual: `python -m brain.main --robot nyx --ip 127.0.0.1 --port <puerto> --virtual-sonar`.
4. Probar con **Zyra**, con una persona supervisando al lado del robot y el piso despejado.
5. Anotar cada prueba con el gesto, la decisión, el motivo y el resultado. Estas notas alimentan el informe.

### Rol 2: Dataset y modelo de gestos

1. Grabar los 7 gestos con la mano que se usará en la demo: `THUMBS_UP`, `OPEN_PALM`, `POINT_LEFT`, `POINT_RIGHT`, `VICTORY`, `FIST` y `NONE`. Cada grabación dura unos 15 segundos. Además, grabar la clase nueva `LOVE_YOU` (ver la sección 6b).
2. Hacer que **cada integrante del equipo** grabe los 7 gestos con su propio identificador:
   ```powershell
   python -m brain.gestures.capture --person <identificador> --label <ETIQUETA> --seconds 15
   ```
   El identificador no puede tener espacios ni guiones bajos.
3. Respetar las condiciones de grabación:
   - Distancia a la webcam entre 0.5 y 1.5 m.
   - Variar la rotación de la mano, el fondo y la iluminación.
   - Usar siempre la **misma mano** que se usará en la demo.
   - En `NONE`, hacer movimientos naturales con la mano relajada, sin ningún gesto.
4. Entrenar el modelo de gestos:
   ```powershell
   python -m brain.gestures.train --data data/gestures --out models/gesture_model.joblib
   ```
5. Revisar `report/metrics/gesture_metrics.json` y las figuras de `report/figures/`, y entregarlas al rol 4.

### Rol 3: Escenarios de contexto con Zyra

1. Preparar el espacio de prueba: piso despejado, robot de pie y batería cargada.
2. Crear al menos **15 escenarios** con el robot real. Cada escenario tiene un identificador (`--scenario-id`) que comparten todas sus filas. Incluir:
   - Caja a 20, 40 y 80 cm frente al robot.
   - Obstáculo a la izquierda y a la derecha.
   - Robot sentado, de pie y en movimiento.
   - Persona muy cerca del robot.
3. Recolectar cada escenario:
   ```powershell
   python -m brain.context.collect --scenario-id <id> --robot zyra
   ```
4. Entrenar el modelo de contexto:
   ```powershell
   python -m brain.context.train --data data/context/scenarios.csv --out models/context_model.joblib
   ```
5. Revisar las métricas generadas en `report/metrics/` y entregarlas al rol 4.

### Rol 4: Informe y video

1. Completar `report/informe_borrador.md`. Los valores pendientes están marcados con dobles corchetes `[[...]]`: reemplazarlos con las métricas de `report/metrics/`, las figuras de `report/figures/` y las notas de pruebas del rol 1.
2. Formato del informe: PDF, Times New Roman 12, interlineado 1.5, márgenes de 2.5 cm, entre 8 y 15 páginas con figuras.
3. Grabar el video siguiendo `report/guion_video.md`. Duración objetivo: 4 minutos (rango de 3 a 5).
4. Revisar que las cifras del informe y del video coincidan con los archivos de métricas finales.

## 6b. Checklist del equipo

Estado al 13/11/2026. Marcar cuando esté lista.

### Hecho

- [x] **Grabar `LOVE_YOU`** con los tres integrantes (Matías, Vanessa y Vivianne). También se regrabó `VICTORY` de Vanessa y Vivianne.
- [x] **Reentrenar el modelo de gestos** con las tres personas. Con tres o más personas el script cambia solo al protocolo honesto: `GroupKFold` por persona más una persona reservada para prueba. Resultado: 10 813 fotogramas, 8 clases, F1 macro de 0.937 sobre la persona reservada.
- [x] **Reentrenar para la demo** con `--refit-all`, para que el modelo desplegado conozca a los tres integrantes:
  ```powershell
  python -m brain.gestures.train --data data/gestures --out models/gesture_model.joblib --refit-all
  ```
- [x] **Medir la confusión `LOVE_YOU` / `VICTORY`**: el 42 % de los fotogramas de `VICTORY` se leen como `LOVE_YOU` en una persona no vista. Por eso `--confirm-ambiguous` quedó **activada por defecto**.

### Pendiente

- [ ] **Sesión con el robot físico (Zyra).** Es el 35 % de la nota y todavía no hay ninguna: las 19 ejecuciones de `runs/` son contra Choregraphe (`127.0.0.1`). Guardar el CSV de esa sesión como evidencia.
  ```powershell
  python -m brain.main --robot zyra
  ```
- [ ] **Recolectar el conjunto contextual** y entrenar el segundo modelo. Sin él no existe `models/context_model.joblib` y el sistema funciona solo con reglas más la capa de seguridad. Unos 40 escenarios bastan:
  ```powershell
  python -m brain.context.collect --scenario-id <id> --robot zyra
  python -m brain.context.train --data data/context/scenarios.csv --out models/context_model.joblib
  ```
  Si no se alcanza a recolectar, hay que quitar la promesa del modelo contextual del README y del informe, y presentar la capa de seguridad como lo que es: reglas deterministas.
- [ ] **Grabar a más personas** para `VICTORY` y `LOVE_YOU`. La curva de aprendizaje muestra que más fotogramas de las mismas personas no mejoran nada; hacen falta más manos distintas.
- [ ] **Copiar las canciones**: "We Are the Champions" en el archivo de `CHAMPIONS_SONG`, y las canciones de DJ en la carpeta de `MUSIC_DIR` (`.wav` o `.mp3`). Las dos rutas se definen en `.env`. Los audios no se suben al repositorio.
- [ ] **Crear la clave de Groq** y ponerla en `GROQ_API_KEY` del `.env`. El archivo `.env` no se sube al repositorio.
- [ ] **Probar la red del laboratorio**: confirmar que no bloquea `api.groq.com`. Si bloquea, la voz funciona solo con el respaldo por palabras clave.
- [ ] **Completar el informe.** Quedan 31 marcadores `[[PENDIENTE]]`; los resueltos ya traen las métricas reales. Los que faltan necesitan la sesión con el robot, el modelo contextual, las condiciones de grabación y las referencias.
- [ ] **Grabar el video** (3 a 5 minutos) siguiendo `report/guion_video.md`.

Configuración inicial del `.env` (una vez por computador):

```powershell
copy .env.example .env
```

Completar `NAO_IP`, `NAO_PORT`, `GROQ_API_KEY`, `MUSIC_DIR` y `CHAMPIONS_SONG`.

## 7. Idea de demo: "misión"

Para el video, el robot recorre un pequeño circuito de cajas hasta una meta, guiado solo con gestos. En cada paso el robot puede avanzar, girar o rechazar un movimiento. Cuando una caja está demasiado cerca, el robot se niega a chocar y lo explica. Esto muestra el valor del sistema: **obedece, pero solo cuando es seguro**.

## 8. Problemas conocidos

| Problema | Qué hacer |
|----------|-----------|
| La mano no se detecta o está muy lejos | Acercar la mano a 0.5–1.5 m de la webcam; buena luz; fondo simple |
| El puerto de Choregraphe cambia en cada apertura | Leer el puerto nuevo en la ventana de conexión antes de cada prueba |
| El robot virtual no tiene sonares reales | Usar `--virtual-sonar` para simularlos, o probar con Zyra |
| El robot habla en otro idioma | Revisar la configuración de voz (español) antes de grabar |
| El modo real con el puente no está probado | Probar primero en `--mock` y después con el robot virtual; en el robot real, supervisar siempre |

## 9. Seguridad (obligatorio)

- **Nunca** dejar el robot solo mientras se mueve.
- Piso despejado, sin cables ni personas cerca del recorrido, salvo quien supervisa.
- Batería cargada antes de cada sesión con el robot.
- Ante cualquier comportamiento extraño: mostrar `OPEN_PALM` (detenerse) o apagar el robot.
- El robot **no retrocede**: no tiene sensor trasero. Por eso no hay movimiento hacia atrás.
- Las pruebas automatizadas se hacen solo en `--mock`. Las pruebas con el robot real las hace el equipo, con supervisión.
