# Guion del video de demostración — Control gestual de NAO

Práctica N.° 2 · Inteligencia Artificial · Ciclo 2026-2

**Duración objetivo:** 4:00 (rango permitido: 3:00–5:00)
**Integrantes que hablan:** Vanessa, Matías, Facundo, Vivianne
**Requisito de la rúbrica:** el modelo debe verse funcionando en el robot NAO real. Una grabación en simulación no cumple el criterio "Funcionamiento del modelo en el NAO" (35 %).

> Los valores marcados como `[[PENDIENTE]]` se completan solo con las cifras que salgan de `report/metrics/*.json` tras el entrenamiento. No se deben escribir cifras de memoria.

---

## 1. Guion por segmentos

| Tiempo | Quién habla | Texto (se dice tal cual) | Qué muestra la cámara |
|---|---|---|---|
| 0:00–0:40 | Vanessa | "Hola, somos el equipo de la Práctica 2 de Inteligencia Artificial. Nuestro proyecto busca que el robot NAO responda a gestos humanos. Pero no basta con obedecer: queremos que el robot decida qué hacer según lo que ve y lo que detecta a su alrededor. Por ejemplo, si una persona le indica que avance y hay un obstáculo delante, el robot debe quedarse quieto y advertirlo." | Plano medio de Vanessa frente a la laptop. Luego plano general del NAO sobre la mesa, con el área de trabajo despejada. |
| 0:40–1:30 | Matías | "El sistema tiene tres partes. Una cámara web captura al usuario y MediaPipe Pose extrae 33 puntos del cuerpo. Un clasificador de gestos, entrenado con datos propios de los cuatro integrantes, reconoce cuatro gestos más un estado neutro. Después, un segundo modelo de contexto recibe el gesto, los sonares del robot, la distancia a la persona y si el robot se está moviendo, y decide la acción. Una capa de seguridad tiene la última palabra: si hay un obstáculo a menos de 0,30 metros, nunca avanza. Las acciones se envían al robot mediante un puente en Python 2.7 que usa NAOqi. Los datos de gestos se grabaron con la webcam, un integrante a la vez, y los datos de contexto son escenarios reales etiquetados por personas del equipo, no reglas programadas." | Captura de pantalla de la laptop con el diagrama de tres bloques (visión → gesto → contexto → seguridad → puente NAOqi). Luego plano de Matías señalando el diagrama y, al mencionar los datos, la carpeta `data/gestures` y `data/context` en el explorador de archivos. |
| 1:30–2:10 | Facundo | "Ahora la demostración con gestos. Vamos a probar cuatro gestos y un caso neutro. Primero, avanzar: el robot da un paso corto hacia adelante. Después, detenerse: el robot se detiene. Luego, girar a la izquierda y girar a la derecha. Por último, un movimiento que no es un gesto: el robot no responde." | Plano del NAO de frente y de la persona de cuerpo entero frente a la cámara. Pantalla de la laptop con el esqueleto superpuesto y la etiqueta predicha en pantalla, en cada gesto. Secuencia: ADVANCE → STOP → TURN_LEFT → TURN_RIGHT → NONE. |
| 2:10–2:40 | Vivianne | "Este es el caso principal. La persona hace el gesto de avanzar, pero hay una caja delante del robot. El clasificador de contexto propone avanzar, pero la capa de seguridad detecta que el sonar está por debajo de 0,30 metros y cambia la acción a advertencia. Observen: el robot no se mueve, dice que hay un obstáculo y sus ojos se ponen en rojo." | Plano fijo del NAO con una caja de cartón en frente, a menos de 30 cm. Plano de la persona haciendo el gesto de avanzar. Primer plano de los ojos del NAO en rojo. Pantalla con la acción propuesta y la acción final (WARN_OBSTACLE). |
| 2:40–3:20 | Vivianne | "El segundo caso es cuando la persona está lejos. El gesto se reconoce, pero la distancia es insuficiente para actuar con seguridad, así que el robot pide que se acerque. Retiramos el obstáculo para que quede claro que la decisión depende de la distancia y no del obstáculo." | Persona de pie a distancia (la silueta ocupa poca parte del encuadre). Pantalla con `person_distance` bajo y la acción ASK_CLOSER. Plano del NAO diciendo "Acércate, no te veo bien". Luego, la persona se acerca y el robot responde. |
| 3:20–3:50 | Matías | "Estos son los resultados del clasificador de gestos y del clasificador de contexto. En gestos, la precisión fue de [[PENDIENTE]] y el F1 macro de [[PENDIENTE]], con validación por persona. En contexto, la precisión fue de [[PENDIENTE]] y el F1 macro de [[PENDIENTE]]. La matriz de confusión completa está en el informe." | Pantalla con la tabla de métricas (de `report/metrics/*.json`) y una imagen de la matriz de confusión de `report/figures/`. Plano de Matías señalando la pantalla. |
| 3:50–4:00 | Facundo | "En resumen, el robot responde a gestos y decide según el contexto. Su limitación principal es [[PENDIENTE]]. Como trabajo futuro, queremos ampliar el dataset y probar más personas. Gracias." | Plano de los cuatro integrantes junto al robot. Fin con el logo de la práctica. |

---

## 2. Notas de guion

- Frases en tiempo presente y sin tecnicismos fuera de lo necesario.
- Si una acción no sale como se espera en la grabación, se repite la toma, no se corrige el texto. El video debe mostrar el comportamiento real.
- Los nombres de las acciones (ADVANCE, STOP, TURN_LEFT, TURN_RIGHT, NONE, WARN_OBSTACLE, ASK_CLOSER) aparecen en pantalla, no solo en la voz.
- Si cambian los roles de los integrantes, solo se ajusta la columna "Quién habla".
- Valores de referencia del diseño: avance de 0,2 m por paso, giro de 0,785 rad (45°), umbral de obstáculo de 0,30 m, `person_distance` normalizada por el ancho del cuadro.

---

## 3. Checklist de grabación

**Audio**
- [ ] Micrófono externo (lapel o de solapa) para la voz. No usar el micrófono integrado de la laptop.
- [ ] Prueba de audio antes de grabar: 10 segundos de voz y verificar que se escucha claro sobre el ruido del ventilador del robot.
- [ ] Silenciar notificaciones del equipo y del celular.

**Cámaras**
- [ ] Cámara 1: ángulo sobre el robot (NAO de frente o de tres cuartos, a la altura de sus ojos).
- [ ] Cámara 2: ángulo sobre la persona que hace los gestos (cuerpo entero, sin cortar las manos).
- [ ] Captura de pantalla de la laptop: ventana con el esqueleto de MediaPipe superpuesto y la etiqueta predicha visibles. Grabar con un programa de captura de pantalla, no con la cámara.
- [ ] Ambas cámaras sincronizadas con una palmada al inicio (para editar el corte).

**Iluminación**
- [ ] Luz frontal y uniforme sobre la persona. Evitar contraluz de ventanas.
- [ ] Sin sombras fuertes sobre la cara ni sobre las manos.

**Robot**
- [ ] Batería del NAO cargada al 100 % antes de la grabación, con cargador disponible.
- [ ] Robot con la cabeza en posición neutra y las articulaciones calibradas.
- [ ] Ojos del NAO visibles desde la cámara 1 (para el caso WARN_OBSTACLE).

**Espacio**
- [ ] Área de trabajo despejada, sin cables ni objetos en el suelo, excepto la caja del caso de obstáculo.
- [ ] Marca en el suelo a 30 cm del sonar para la caja.
- [ ] Una persona lista para sujetar el robot en todo momento (ver sección de seguridad).

**Software**
- [ ] Modelos de `models/` cargados y versionados. Anotar la versión usada.
- [ ] Prueba previa en modo `--mock` para confirmar que la secuencia de acciones es la esperada.
- [ ] Prueba previa en el robot real con la secuencia completa, sin grabar.

---

## 4. Lista de tomas (shot list)

| # | Escena | Cámara | Duración aprox. | Notas |
|---|---|---|---|---|
| 1 | Vanessa, intro | Cámara 2 (plano medio) | 0:25 | Fondo despejado |
| 2 | NAO sobre la mesa, plano general | Cámara 1 | 0:10 | Sin movimiento |
| 3 | Diagrama de bloques | Captura de pantalla | 0:30 | Mostrar cada bloque al mencionarlo |
| 4 | Carpetas de datos | Captura de pantalla | 0:20 | Sin datos personales en nombres de archivo |
| 5 | Gesto ADVANCE | Cámara 1 + captura de pantalla | 0:10 | Robot da un paso |
| 6 | Gesto STOP | Cámara 1 + captura de pantalla | 0:10 | Robot se detiene |
| 7 | Gesto TURN_LEFT | Cámara 1 + captura de pantalla | 0:10 | Giro a la izquierda |
| 8 | Gesto TURN_RIGHT | Cámara 1 + captura de pantalla | 0:10 | Giro a la derecha |
| 9 | Movimiento NONE | Cámara 1 + captura de pantalla | 0:10 | Robot no responde |
| 10 | Caso obstáculo: caja delante, gesto ADVANCE | Cámara 1, cámara 2 | 0:20 | Repetir si el robot se mueve |
| 11 | Primer plano de ojos rojos | Cámara 1 (zoom) | 0:05 | Solo tras la toma 10 |
| 12 | Caso lejos: persona alejada | Cámara 2 (plano general) | 0:15 | Pantalla con distancia baja |
| 13 | Robot dice "Acércate, no te veo bien" | Cámara 1 | 0:08 | Audio del robot debe escucharse |
| 14 | Persona se acerca, robot responde | Cámara 2 | 0:12 | Transición a la acción normal |
| 15 | Tabla de métricas | Captura de pantalla | 0:20 | Solo cifras de `report/metrics/*.json` |
| 16 | Matriz de confusión | Captura de pantalla | 0:10 | Imagen de `report/figures/` |
| 17 | Cierre con los cuatro integrantes junto al robot | Cámara 1 | 0:10 | Robot apagado o en posición de reposo |

---

## 5. Seguridad durante la grabación

- Una persona lista para sujetar el robot en todo momento, con las manos cerca del torso del NAO.
- Nadie dentro del radio de giro del robot al ejecutar TURN_LEFT o TURN_RIGHT.
- Probar primero en `--mock` y después en el robot, sin grabar, para confirmar la secuencia.
- Si el robot se mueve de forma inesperada, detener la grabación, usar el comando de parada y repetir la prueba.
