# Guion del video — NAO Dance AI

Duración objetivo: **4:00** (rango permitido 3:00–5:00).
Personas: **Vanessa (V)**, **Matías (M)**, **Facundo (F)**, **Vivianne (Vi)**.

> Los textos entre corchetes `[[PENDIENTE]]` se reemplazan con cifras reales del informe antes de grabar. Si una cifra no existe, se dice tal cual. No se inventan datos.
> Las canciones de la demo se indican como **Canción A / B / C** hasta que el equipo confirme los títulos (ver la licencia en `data/songs/manifest.csv`).

## 1. Tabla del guion

| Tiempo | Persona | Texto (completo, en español) | Lo que muestra la cámara |
|---|---|---|---|
| 0:00–0:40 | **Vi** | "Hola. Somos el equipo de NAO Dance AI. Muchas coreografías para robots se programan a mano, movimiento por movimiento, y eso no cambia según la música. Nuestra pregunta fue: ¿puede un modelo de inteligencia artificial escuchar una canción y decidir qué estilo de movimiento le corresponde al robot?" | Plano medio de Vi frente a la laptop. Luego plano del NAO en la mesa, quieto. |
| 0:40–1:30 | **V** | "Primero reunimos diez canciones con licencia legal y registramos la procedencia de cada una. Cada canción se divide en fragmentos de ocho segundos, y de cada fragmento extraemos el tempo, la energía, el timbre y los inicios de sonido, con la librería librosa." | Pantalla de la laptop: `manifest.csv` con fuente y licencia, luego `segments.csv` abierto. |
| 0:40–1:30 | **F** | "Las etiquetas las pusimos nosotros, escuchando. Cada integrante anota todos los fragmentos sin ver las respuestas de los demás, y luego combinamos las respuestas por votación. Lo hicimos así para que el modelo no aprenda solamente una regla de BPM. El acuerdo entre anotadores lo medimos con kappa de Fleiss: [[PENDIENTE: kappa]]." | Plano de F anotando en la laptop con el comando `annotate` visible. Pantalla de `merge_labels` con el resultado. |
| 0:40–1:30 | **M** | "El modelo es machine learning clásico: comparamos un Random Forest y una Support Vector Machine. No es una red neuronal. Los entrenamos separando por canción, para que ningún fragmento de una canción aparezca en entrenamiento y en prueba al mismo tiempo." | Pantalla con `python -m dance.model.train ...` ejecutándose. Plano de M señalando la pantalla. |
| 1:30–1:45 | **M** | "Ahora la demostración. Primero, la canción A, una pieza lenta. Vamos a ver lo que el sistema analiza." | Plano de M con la laptop. Comando `python -m dance.main --song <Canción A> --mock` en pantalla. |
| 1:45–2:15 | **V** | "En la pantalla aparece el BPM estimado, las características de cada fragmento, el estilo que predijo el modelo y la coreografía elegida. Aquí ven el estilo predicho para cada fragmento de la canción, en el tiempo." | Zoom sobre el overlay de la laptop: BPM, estilo, coreografía, línea de tiempo. |
| 2:15–2:45 | **F** | "Ahora la canción B, con un tempo medio y más energía. Aquí el modelo predice el estilo que aparece en pantalla. El suavizado evita que el estilo cambie de forma brusca entre fragmentos." | Plano del NAO ejecutando la coreografía moderada (o simulación si no está disponible el robot, indicándolo en pantalla). |
| 2:45–3:15 | **Vi** | "Por último, la canción C, que no usamos para entrenar. Este resultado nos sirve para ver si el modelo generaliza a música nueva. Con ella el sistema predice [[PENDIENTE: estilo predicho]] con confianza [[PENDIENTE: confianza]]." | Plano del NAO con la canción C sonando desde la laptop. Pantalla con el estilo predicho. |
| 3:15–3:45 | **F** | "Resultados sobre las canciones de prueba. El accuracy fue [[PENDIENTE: accuracy]], el macro-F1 [[PENDIENTE: macro-F1]]. Random Forest obtuvo [[PENDIENTE: RF]], SVM [[PENDIENTE: SVM]] y la línea base por reglas [[PENDIENTE: baseline]]." | Gráfica `report/figures/dance_*.png` con la matriz de confusión y la comparación de modelos. |
| 3:15–3:45 | **M** | "Con diez canciones, estos números tienen limitaciones estadísticas. Por eso los presentamos como una primera evaluación y no como una prueba definitiva." | Plano de M frente a la pantalla con la tabla de métricas. |
| 3:45–4:00 | **Vi** | "En conclusión, el sistema analiza una canción, predice un estilo de movimiento y ejecuta una coreografía. Las limitaciones son claras: pocas canciones, sin micrófono del robot y coreografías solo de brazos y cabeza. Gracias." | Plano de los cuatro integrantes con el NAO al centro. Pantalla final con el nombre del proyecto. |

**Nota de honestidad:** si el modo real con el robot no se pudo grabar, el segmento 2:15–2:45 se graba en `--mock` y se dice explícitamente en voz alta: "Esta demostración corresponde a la simulación". El video no puede afirmar que el robot ejecutó movimientos si no los ejecutó.

## 2. Lista de grabación

- [ ] Micrófono externo conectado y probado (el micrófono de la laptop no se usa).
- [ ] La música se escucha con claridad pero no tapa las voces; prueba de volumen hecha antes de grabar.
- [ ] Dos tomas: una del NAO y las personas, otra de la pantalla de la laptop con el overlay.
- [ ] Laptop y NAO con batería cargada; cargador a mano.
- [ ] Zona despejada: sin mesas ni bordes cerca del robot.
- [ ] Alguien al lado del NAO durante cualquier movimiento, listo para sostenerlo.
- [ ] Modo `--mock` probado antes de grabar; si se usa `--robot zyra`, probado antes en esa sesión.
- [ ] Parada probada (Ctrl+C o `q`) antes de la toma.
- [ ] Métricas en pantalla ya reemplazadas por cifras reales (sin `[[PENDIENTE]]`).
- [ ] Los cuatro integrantes han leído su parte y saben qué canción les toca presentar.

## 3. Lista de tomas

| # | Toma | Ángulo | Para qué segmento |
|---|---|---|---|
| T1 | Plano medio de Vi con la laptop | Cámara 1 | 0:00–0:40 |
| T2 | NAO quieto en la mesa | Cámara 1 | 0:00–0:40 (inserto) |
| T3 | Manifest y segments abiertos en la laptop | Cámara 2 (pantalla) | 0:40–1:30 |
| T4 | F anotando; luego `merge_labels` | Cámara 1 y 2 | 0:40–1:30 |
| T5 | Comando de entrenamiento ejecutándose | Cámara 2 (pantalla) | 0:40–1:30 |
| T6 | M inicia `--mock` con la canción A | Cámara 1 y 2 | 1:30–1:45 |
| T7 | Zoom al overlay: BPM, estilo, coreografía, línea de tiempo | Cámara 2 (pantalla) | 1:45–2:15 |
| T8 | NAO ejecutando la coreografía moderada (real o simulada, indicado) | Cámara 1 | 2:15–2:45 |
| T9 | Canción C sonando; estilo predicho en pantalla | Cámara 1 y 2 | 2:45–3:15 |
| T10 | Gráfica de matriz de confusión y comparación de modelos | Cámara 2 (pantalla) | 3:15–3:45 |
| T11 | M frente a la tabla de métricas | Cámara 1 | 3:15–3:45 |
| T12 | Cierre con los cuatro y el NAO al centro | Cámara 1 | 3:45–4:00 |
