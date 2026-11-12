<!--
BORRADOR T4 (S6). Total de marcadores PENDIENTE (formato doble corchete): 95.
Todos los valores numéricos de resultados, figuras y pruebas con el robot están como marcadores y deben completarse
desde report/metrics/*.json, report/figures/*.png y el protocolo de pruebas con el robot (T7/T8).
Los umbrales de seguridad (0.30 m, 0.25 m, 0.20 m, 0.05 m, 0.785 rad, 0.6, 0.25) provienen de la especificación del proyecto, no de resultados.
Formato final: PDF, Times New Roman 12, interlineado 1.5, márgenes 2.5 cm, 8-15 páginas (con figuras).
-->

# NAO que razona antes de obedecer: control por gestos de mano con validación del entorno mediante Machine Learning

**UNIVERSIDAD PERUANA DE CIENCIAS APLICADAS**
Facultad de Ingeniería

**Curso:** 1ASI0404 - Inteligencia Artificial | **Sección:** 9138
**Práctica N.° 2:** Implementación de un Modelo de Machine Learning en el Robot NAO
**Docente:** Jairo Pinedo Taquia

**Integrantes:**
- Barrientos Villalta, Vanessa Jazmin (u202413287)
- Del Castillo Mendoza, Matias Javier (u202411305)
- Maco Rebatta, Facundo Nicolas (u202312084)
- Ríos Hasegawa, Vivianne Fátima (u202410135)

**Fecha:** Noviembre 2026

---

## 1. Introducción

### a. Contexto general

La inteligencia artificial aplicada a la robótica busca que una máquina perciba su entorno, interprete la información y actúe en consecuencia. En robótica de servicio esta cadena es especialmente exigente porque el robot comparte el espacio con personas no expertas que esperan comunicarse con él de forma natural. Los gestos de la mano son un canal inmediato: una palma abierta se entiende como "detente", un dedo que señala indica una dirección y un pulgar hacia arriba expresa avance o aprobación.

Los avances recientes en la detección de manos en tiempo real sobre CPU (Zhang et al., 2020; Lugaresi et al., 2019) convierten la imagen de una cámara convencional en 21 puntos de referencia de la mano, sobre los cuales pueden entrenarse modelos clásicos de bajo costo (Breiman, 2001; Cortes y Vapnik, 1995). Esto hace viable un sistema de control gestual que se ejecuta en una computadora portátil y gobierna un robot humanoide como el NAO (Gouaillier et al., 2009).

### b. Motivación

Existen entornos en los que la voz, el canal más natural para dar órdenes, deja de ser fiable: fábricas con ruido intenso, pasillos hospitalarios donde se debe guardar silencio y, sobre todo, personas con dificultades del habla, para quienes un asistente controlado por voz no es accesible. El gesto de la mano es en esos casos una alternativa directa, silenciosa e inclusiva.

Sin embargo, un robot que obedece a ciegas es inseguro. La orden "avanza" es correcta en un pasillo despejado y es un accidente si hay una caja a veinte centímetros; "saluda" carece de sentido si el robot está caminando; y una orden captada con baja confianza puede ser un gesto involuntario. La motivación del grupo es un robot que razona antes de obedecer: antes de ejecutar el movimiento asociado a un gesto, el NAO examina su entorno, decide si es adecuado actuar, explica con voz y con el color de sus ojos por qué rechaza o limita una orden, y pide confirmación cuando no está seguro.

### c. Objetivos

**Objetivo general.** Diseñar, entrenar e integrar en el robot NAO un sistema de control por gestos de mano que ejecute el movimiento asociado a cada gesto únicamente después de validar el entorno del robot mediante un modelo de decisión contextual y una capa de seguridad, explicando sus rechazos.

**Objetivos específicos.**

1. **OE1.** Construir un conjunto de datos propio de siete clases de gesto de mano, recolectado por los cuatro integrantes con la cámara de una computadora portátil, y entrenar y comparar dos algoritmos (Random Forest y Máquina de Vector Soporte) sobre atributos derivados de los puntos de referencia de la mano, evaluándolos con validación que separe a las personas.
2. **OE2.** Construir un conjunto de escenarios reales con el robot, etiquetados por humanos con la acción que el NAO debería ejecutar, y entrenar y comparar dos algoritmos (Árbol de Decisión y Random Forest) que decidan la acción en función del gesto, su confianza, los sonares, la cercanía de la mano y el estado del robot, contrastándolos con una política basada solo en reglas.
3. **OE3.** Integrar ambos modelos con el NAO mediante un puente de comunicación entre Python 3 y Python 2.7, incorporando una capa de seguridad determinista, retroalimentación hablada y por color de ojos, y avance graduado, y verificar experimentalmente el comportamiento del robot.

### d. Alcance y limitaciones

El alcance comprende siete clases de gesto (`THUMBS_UP`, `OPEN_PALM`, `POINT_LEFT`, `POINT_RIGHT`, `VICTORY`, `FIST` y `NONE`), reconocidas sobre una sola mano a la vez, ubicada aproximadamente entre 0.5 m y 1.5 m de la cámara de la computadora portátil, en un entorno interior de laboratorio. El cálculo se ejecuta en la computadora y no a bordo del NAO.

Las principales limitaciones son: (i) se usa la cámara de la computadora portátil y no las del robot; (ii) los sonares del NAO solo cubren un cono frontal, por lo que no se detectan obstáculos laterales ni traseros; (iii) el robot no retrocede, pues sin sensor trasero un movimiento hacia atrás no podría validarse; (iv) el SDK de NAOqi requiere Python 2.7, lo que obliga a una arquitectura de dos procesos; y (v) los datos provienen de solo cuatro personas y de un acceso limitado al robot, por lo que la generalización debe interpretarse con cautela (Sección 5.b).

---

## 2. Marco Teórico

### MediaPipe Hands

MediaPipe Hands (Zhang et al., 2020), parte del marco MediaPipe (Lugaresi et al., 2019), resuelve la detección de manos en dos etapas. Un detector de palmas localiza la mano en el fotograma completo y una red de regresión de puntos de referencia predice sobre el recorte resultante 21 puntos tridimensionales de la mano (la muñeca, y cuatro articulaciones por cada uno de los cinco dedos), además de la lateralidad (mano izquierda o derecha). Cada punto se entrega con coordenadas $(x, y)$ normalizadas respecto al ancho y alto de la imagen y una profundidad relativa $z$ medida respecto a la muñeca. El detector se ejecuta solo cuando la mano se pierde y en los fotogramas siguientes se reutiliza la posición anterior, lo que reduce el costo y permite tiempo real en CPU.

Es importante precisar el papel de este componente en el trabajo: se trata de una red neuronal preentrenada por sus autores, que el grupo utiliza como extractor de atributos. El equipo no la entrena ni la modifica. Lo que el grupo entrena son los clasificadores que reciben esos puntos de referencia, es decir, el reconocedor de gestos y el modelo de decisión contextual.

### Ingeniería de atributos de la mano

Los 21 puntos en bruto dependen de dónde se encuentra la mano y de qué tan lejos está de la cámara, factores irrelevantes para decidir qué gesto se hace. Por ello se construye un vector de atributos con tres grupos de información. Primero, la **normalización**: las coordenadas se expresan relativas a la muñeca y se dividen entre el tamaño de la mano (distancia de la muñeca a la articulación metacarpofalángica del dedo medio), lo que elimina la dependencia de la posición y la escala. Segundo, la **extensión de cada dedo**, por ejemplo el cociente entre la distancia de la punta a la muñeca y la de su base a la muñeca, o el ángulo entre sus segmentos. Tercero, la **dirección de señalamiento**: el vector normalizado de la base a la punta del índice, cuyo componente horizontal distingue señalar a un lado del otro. El tamaño de la mano, mayor cuanto más cerca está de la cámara, se reutiliza además como indicador de cercanía en el modelo contextual. La lista exacta de atributos está en el repositorio (`FEATURE_NAMES`).

### Random Forest

Un Random Forest (Breiman, 2001) es un conjunto de $B$ árboles de decisión entrenados sobre muestras *bootstrap* del conjunto de entrenamiento (*bagging*). En cada nodo, cada árbol considera solo un subconjunto aleatorio de $m$ atributos candidatos para la partición, lo que decorrelaciona los árboles y reduce la varianza del conjunto. En clasificación, la predicción se obtiene por voto mayoritario:

$$\hat{y}(\mathbf{x}) = \arg\max_{c} \sum_{b=1}^{B} \mathbb{1}\left[ T_b(\mathbf{x}) = c \right]$$

donde $T_b$ es el árbol $b$-ésimo, y la proporción de votos de cada clase puede interpretarse como una estimación de la confianza. Los hiperparámetros principales son el número de árboles `n_estimators`, la profundidad máxima `max_depth`, el mínimo de muestras por hoja `min_samples_leaf` y el número de atributos por partición `max_features`. Como subproducto, el modelo entrega una importancia de atributos basada en la reducción acumulada de impureza.

### Máquinas de Vector Soporte

Una Máquina de Vector Soporte (SVM; Cortes y Vapnik, 1995) busca el hiperplano que separa las clases con el mayor margen posible. La función de decisión binaria es

$$f(\mathbf{x}) = \operatorname{sign}\left( \sum_{i \in SV} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b \right)$$

donde $SV$ es el conjunto de vectores soporte, $\alpha_i$ son multiplicadores de Lagrange, $y_i \in \{-1,+1\}$ las etiquetas y $K$ un núcleo. Para datos no linealmente separables se emplea el núcleo de base radial (RBF):

$$K(\mathbf{x}, \mathbf{x}') = \exp\left( -\gamma \, \lVert \mathbf{x} - \mathbf{x}' \rVert^{2} \right)$$

El hiperparámetro $C$ controla el compromiso entre maximizar el margen y penalizar errores de entrenamiento (un $C$ grande tolera menos errores y puede sobreajustar), y $\gamma$ define el alcance de influencia de cada vector soporte (un $\gamma$ grande produce fronteras más complejas). El problema multiclase se resuelve combinando clasificadores binarios (Pedregosa et al., 2011). Un SVM no entrega probabilidades de forma nativa; para obtener una confianza utilizable en la regla de confirmación se emplea el escalado de Platt (Platt, 1999), que ajusta una función sigmoide sobre las salidas de la función de decisión y produce probabilidades calibradas. Como el núcleo RBF depende de distancias, los atributos deben estandarizarse previamente.

### Árboles de decisión

Un árbol de decisión (Breiman et al., 1984) particiona recursivamente el espacio de atributos con condiciones del tipo $x_j \le \tau$, eligiendo en cada nodo el par (atributo, umbral) que minimiza la impureza de Gini del nodo hijo:

$$G(t) = 1 - \sum_{c=1}^{C} p_{c,t}^{2}$$

donde $p_{c,t}$ es la proporción de muestras de la clase $c$ en el nodo $t$. Su principal ventaja es la interpretabilidad: el modelo entrenado puede dibujarse y leerse como reglas del tipo "si el sonar izquierdo es menor que cierto valor y el gesto es avanzar, entonces advertir obstáculo". En este trabajo esa propiedad tiene un valor práctico, porque las reglas del árbol son la base de las explicaciones que el robot ofrece sobre sus decisiones. Su desventaja es la alta varianza: un árbol profundo tiende a sobreajustar. Los hiperparámetros principales son `max_depth`, `min_samples_leaf`, `criterion` y `ccp_alpha` (poda por costo-complejidad).

### IA explicable y confirmación con humano en el bucle

La literatura sobre aprendizaje interpretable sostiene que la explicabilidad es necesaria cuando las decisiones de un sistema afectan la seguridad de las personas (Doshi-Velez y Kim, 2017). Aquí se aplican dos ideas. La primera: cada rechazo o limitación de una orden va acompañado de una razón en lenguaje natural, hablada por el robot y mostrada en pantalla, derivada de la regla que lo causó. La segunda, la confirmación con humano en el bucle: si la confianza del reconocimiento es menor que un umbral, el robot no actúa sino que pide repetir el gesto, de modo que una duda se resuelve con una pregunta y no con un movimiento.

### Criterio de selección frente a alternativas

La elección de modelos se justifica por las características del problema y no por la sofisticación del algoritmo:

- **Conjunto de datos pequeño y propio.** Con datos de cuatro personas, una red convolucional de extremo a extremo sobre imágenes requeriría muchos más ejemplos y una GPU para entrenarse, y sería propensa al sobreajuste al fondo y la iluminación. Trabajar sobre puntos de referencia aprovecha la red preentrenada de MediaPipe en lugar de reentrenar una.
- **Atributos tabulares de baja dimensión.** Coordenadas normalizadas, ángulos y lecturas de sonar son atributos tabulares para los que los árboles y los SVM con núcleo suelen ser competitivos con costo bajo (Hastie et al., 2009), con inferencia en tiempo real en CPU.
- **Interpretabilidad en decisiones de seguridad.** El árbol de decisión permite explicar por qué el robot eligió una acción, y la importancia de atributos del Random Forest complementa el análisis.

Se comparan Random Forest y SVM para el reconocimiento de gestos (representaciones continuas y geometría suave) y Árbol de Decisión y Random Forest para la decisión contextual (variables mixtas, umbrales y reglas), de modo que en cada caso se contrasta un modelo más interpretable o de frontera suave con uno más robusto.

### Robot NAO

NAO es un robot humanoide de unos 58 cm de altura desarrollado por Aldebaran y luego SoftBank Robotics (Gouaillier et al., 2009). Dispone de 25 grados de libertad repartidos entre cabeza, brazos, manos, piernas y cadera, lo que permite caminar, girar y gesticular. Su arquitectura sensorial incluye:

- **Dos cámaras** frontales (frente y boca).
- **Dos pares emisor-receptor de sonar** en el pecho, que estiman la distancia a obstáculos delante del robot, con un alcance nominal aproximado de 0.25 a 2.55 m [[PENDIENTE: verificar el rango y el ángulo del cono en la documentación del fabricante]].
- **Una unidad de medición inercial (IMU)** con giroscopio y acelerómetro para estimar orientación y equilibrio.
- **Sensores de fuerza resistivos (FSR)** en las plantas de los pies.
- **Sensores táctiles** en la cabeza y las manos, además de micrófonos y altavoces.

El sistema operativo es NAOqi OS, una distribución Linux sobre la cual corre el middleware NAOqi, que expone la funcionalidad del robot como módulos accesibles por red mediante *proxies* (SoftBank Robotics, s. f.). Los módulos relevantes para este trabajo son `ALMotion` (locomoción y articulaciones, por ejemplo `moveTo` y `stopMove`), `ALRobotPosture` (posturas como Stand, Sit o Crouch), `ALMemory` (lectura de valores de sensores, como los sonares), `ALTextToSpeech` (síntesis de voz) y `ALLeds` (color de los LED de los ojos). El SDK es accesible desde Python y C++, y se dispone también de Choregraphe, un entorno gráfico de programación por bloques. Un punto crítico es que el SDK Python de NAOqi corresponde a Python 2.7, mientras que MediaPipe requiere una versión moderna de Python 3; esta incompatibilidad determina la arquitectura descrita en la Sección 3.f.

---

## 3. Metodología

### a. Descripción del problema

El problema se descompone en dos tareas de aprendizaje supervisado encadenadas y una capa determinista. La primera es **clasificar el gesto** en una de siete clases a partir de los 21 puntos de la mano. La segunda es **decidir qué acción ejecuta el NAO** dados el gesto, su confianza, los sonares izquierdo y derecho, el tamaño de la mano como indicador de cercanía, y si el robot está en movimiento y de pie; la salida es una de catorce acciones, entre ellas conductas que ningún gesto ordena directamente (advertir de un obstáculo o de cercanía, exigir detenerse o ponerse de pie, avanzar solo una distancia corta o pedir confirmación). La tercera pieza es una **capa de seguridad** que valida la decisión con reglas explícitas y tiene la última palabra.

La siguiente tabla resume las clases de gesto, fijadas antes de la recolección para que los cuatro integrantes las realizaran de forma consistente.

| Clase | Descripción del gesto físico | Movimiento asociado |
|---|---|---|
| `THUMBS_UP` | [[PENDIENTE: describir el gesto acordado (pulgar hacia arriba, resto de dedos cerrados)]] | Avanzar 20 cm |
| `OPEN_PALM` | [[PENDIENTE: describir el gesto acordado (palma abierta hacia la cámara)]] | Detenerse |
| `POINT_LEFT` | [[PENDIENTE: describir el gesto acordado (índice señalando al lado izquierdo de la persona)]] | Girar |
| `POINT_RIGHT` | [[PENDIENTE: describir el gesto acordado (índice señalando al lado derecho de la persona)]] | Girar |
| `VICTORY` | [[PENDIENTE: describir el gesto acordado (índice y medio extendidos en V)]] | Alternar la entrada entre gestos y voz (sección 3.g) |
| `FIST` | [[PENDIENTE: describir el gesto acordado (puño cerrado)]] | Sentarse o ponerse de pie (alterna) |
| `NONE` | [[PENDIENTE: describir la mano en reposo, transiciones y posturas no reconocibles incluidas]] | Ninguna |

### b. Origen y características del dataset

Se utilizan dos conjuntos de datos propios, ambos recolectados por el grupo, sin conjuntos públicos.

**Conjunto de gestos.** Los cuatro integrantes realizaron cada uno de los siete gestos frente a la cámara web de una computadora portátil. Cada fotograma pasó por MediaPipe Hands (solución `mp.solutions.hands`, una sola mano) y se guardó como una fila de un archivo CSV por persona, clase y marca de tiempo, con las columnas `person`, `label`, `handedness` y, para cada uno de los 21 puntos, las coordenadas `x`, `y` y `z`. Se solicitó variar la distancia a la cámara (dentro del rango de 0.5 a 1.5 m), la inclinación de la mano, la posición en el encuadre y la iluminación dentro de lo razonable para el laboratorio, y alternar manos izquierda y derecha cuando fuera posible. Tamaño del conjunto: [[PENDIENTE: número total de muestras, por persona y por clase (tabla de distribución)]]; condiciones de captura: [[PENDIENTE: resolución, tasa de cuadros, iluminación y distancia aproximada a la cámara]].

**Conjunto contextual.** Reúne escenarios reales con el robot (modelo Zyra) en los que se registran las variables de entrada y un integrante del grupo etiqueta, según su criterio, la acción que el robot debería ejecutar. Por ejemplo, ante el gesto `THUMBS_UP` con una caja a 20 cm del robot se etiqueta `WARN_OBSTACLE`, y con la caja a 40 cm se etiqueta `ADVANCE_SHORT`. El archivo `scenarios.csv` contiene las columnas `gesture`, `gesture_confidence`, `sonar_left`, `sonar_right`, `hand_size`, `moving`, `standing`, `action` y `scenario_id`. Las etiquetas las asignan personas; no se generan mediante reglas, para evitar que el modelo se limite a reproducir una función conocida. Tamaño: [[PENDIENTE: número de escenarios y distribución por acción]]; etiquetadores y criterio de etiquetado: [[PENDIENTE: quién etiquetó y con qué criterio común]].

**Aumento sintético (declarado).** [[PENDIENTE: confirmar si se usó aumento sintético. Si se usó: describir el procedimiento (por ejemplo, ruido gaussiano pequeño sobre las lecturas de sonar), cuántas muestras sintéticas se agregaron y confirmar que solo se emplearon en entrenamiento]]. Cualquier muestra sintética se identifica como tal, se reporta por separado y no se incluye jamás en los conjuntos de prueba; las métricas de la Sección 4 se calculan únicamente sobre datos reales.

### c. Preprocesamiento

- **Valores faltantes e invarianza.** Si MediaPipe no detecta una mano, `extract_features` devuelve `None` y la ausencia equivale a `NONE` con confianza cero; las lecturas de sonar no positivas o no numéricas se tratan como inválidas y, por seguridad, como no seguras. Las coordenadas se expresan relativas a la muñeca y se dividen entre el tamaño de la mano.
- **Lateralidad.** La lateralidad detectada se incorpora como atributo o se usa para reflejar la mano izquierda sobre la derecha, de modo que los gestos de señalamiento conserven su significado con cualquiera de las dos manos [[PENDIENTE: confirmar cuál de las dos estrategias implementa features.py]].
- **Estandarización.** Los atributos se estandarizan con `StandardScaler` (media cero, varianza unitaria) para el SVM, cuyo núcleo es sensible a la escala; los árboles no la requieren. El escalador forma parte de un `Pipeline` y se ajusta solo con los datos de entrenamiento de cada partición, lo que evita filtrar información de la partición de prueba.
- **Codificación de variables.** En el modelo contextual, el gesto, categórico, se codifica en *one-hot*; `moving` y `standing` se representan como 0/1; la variable objetivo es la acción.
- **Limpieza.** [[PENDIENTE: filas eliminadas por ausencia de mano, duplicados o lecturas de sonar inválidas, con conteos]].

### d. División de datos

Una división aleatoria de filas sería inapropiada: los fotogramas consecutivos de una misma persona y gesto son casi idénticos, de modo que una fila de prueba tendría una gemela casi exacta en el entrenamiento y la métrica sería optimista por fuga de información. Por ello se emplea validación cruzada con `GroupKFold`:

- **Gestos:** los grupos son las personas. Con cuatro integrantes, cada partición reserva a una persona completa para prueba y entrena con las otras tres, lo que mide la generalización a una persona no vista, el escenario real de uso.
- **Contexto:** los grupos son los escenarios (`scenario_id`), de modo que ninguna variante de un mismo escenario aparezca a la vez en entrenamiento y prueba.

Esquema exacto de evaluación (validación anidada con ajuste interno, o conjunto de prueba reservado): [[PENDIENTE: confirmar según el código de entrenamiento de brain/gestures y brain/context]]. Se reporta la media y la desviación estándar de las métricas entre particiones.

**Nota de versión.** Las métricas de esta sección corresponden a las siete clases originales. La clase `LOVE_YOU` se añadió después y se evaluará en la sección 3.h con el mismo esquema.

### e. Entrenamiento y ajuste de hiperparámetros

Cada algoritmo se ajusta con `GridSearchCV` (Pedregosa et al., 2011) usando la validación por grupos y optimizando el F1 macro (`scoring="f1_macro"`), que promedia el F1 de cada clase con el mismo peso y evita que una clase mayoritaria domine la métrica cuando las clases no están perfectamente balanceadas. Se exploran las siguientes rejillas:

| Modelo | Algoritmo | Hiperparámetros explorados | Valores |
|---|---|---|---|
| Gestos | Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` | [[PENDIENTE: rejilla exacta]] |
| Gestos | SVM (RBF) | `C`, `gamma`, con probabilidades calibradas | [[PENDIENTE: rejilla exacta]] |
| Contexto | Árbol de Decisión | `max_depth`, `min_samples_leaf`, `criterion` | [[PENDIENTE: rejilla exacta]] |
| Contexto | Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf` | [[PENDIENTE: rejilla exacta]] |

Las métricas reportadas son exactitud (*accuracy*), F1 macro, matriz de confusión y, para diagnosticar sobreajuste y suficiencia de datos, la curva de aprendizaje del mejor modelo de gestos. La semilla aleatoria es fija para que los experimentos sean reproducibles. Los modelos finales se serializan en `models/gesture_model.joblib` y en un archivo equivalente para el modelo contextual.

### f. Estrategia de integración con el NAO

**Arquitectura de dos procesos.** El SDK `naoqi` solo funciona con Python 2.7, mientras que MediaPipe requiere Python 3.11. Siguiendo el patrón del docente (archivo `py27-py3.py` entregado con la práctica), en el que un programa de Python 3 invoca con `subprocess` el intérprete `C:\Python27\python.exe` para ejecutar un script que usa `ALProxy`, separamos un "cerebro" en Python 3.11 (visión y modelos) de un "puente" en Python 2.7 (acceso al robot). La diferencia es que el ejemplo del docente lanza un proceso nuevo por cada ejecución, lo que obligaría a reconectar con el robot en cada orden; nuestro puente es un proceso persistente que mantiene los *proxies* abiertos y se comunica con el cerebro mediante líneas JSON por la entrada y la salida estándar. El patrón de invocación es una adaptación, con atribución, del código del docente.

```
 Webcam (laptop)
      |
      v
+--------------------------- Python 3.11 ("cerebro") ---------------------------+
| MediaPipe Hands --> extract_features --> GesturePredictor (RF/SVM, voto 7) ----+--> gesto, confianza
|                                                                                |
| NaoClient.sense() <-- sonar_izq, sonar_der, moving, posture                    |
|                                                                                |
| ContextDecider: modelo contextual (DT/RF) --> validate (capa de seguridad)     |
|        --> Decision(accion, razon, color de ojos) --> NaoClient.<comando>      |
+------------------------------------+-------------------------------------------+
                                     |  JSON por linea (stdin/stdout)
                                     v
+------------------------- Python 2.7 ("puente", persistente) -------------------+
| nao_bridge.py: ALProxy(ALMotion, ALRobotPosture, ALMemory, ALTextToSpeech,     |
|                        ALLeds)                                                 |
+------------------------------------+-------------------------------------------+
                                     |  NAOqi (red)
                                     v
                          Robot NAO (Zyra / Nyx / simulador)
```

**Protocolo del puente.** Cada mensaje es un objeto JSON en una línea y toda respuesta incluye `ok`. Los comandos son `ping`, `sense` (devuelve `sonar_left`, `sonar_right`, `moving` y `posture`), `move` (desplazamiento no bloqueante con `moveTo`), `stop`, `wave`, `posture` (sentado o de pie), `say`, `eyes` y `quit`. El puente limita los valores de `move` (avance entre 0 y 0.30 m, giro de a lo sumo 1.6 rad) y lo rechaza si el robot no está de pie; una orden inválida produce `ok: false`, de modo que el cerebro nunca asume un éxito. Admite `--robot nyx|zyra|sim` o `--ip/--port`, y un modo `--mock` sin SDK para probar toda la tubería sin robot. El cliente `NaoClient` lanza el puente una sola vez y expone un método por comando.

**De gesto a intención.** El reconocedor de gestos emite una etiqueta suavizada y su confianza. La etiqueta se traduce en una intención de movimiento, que el modelo contextual y la capa de seguridad posteriormente validan:

| Gesto | Intención | Movimiento en el NAO |
|---|---|---|
| `THUMBS_UP` | Avanzar | `moveTo` de hasta 0.20 m hacia adelante |
| `OPEN_PALM` | Detenerse | `stopMove` |
| `POINT_LEFT` | Girar hacia lo señalado | El robot gira a su derecha (0.785 rad, 45°) |
| `POINT_RIGHT` | Girar hacia lo señalado | El robot gira a su izquierda (0.785 rad, 45°) |
| `VICTORY` | Alternar entrada gestos / voz | Ninguno: cambia el canal de órdenes, no mueve al robot (sección 3.g) |
| `FIST` | Sentarse o pararse | Alterna entre sentado y de pie |
| `NONE` | Ninguna | No hace nada |

La semántica de señalamiento merece una aclaración. Se decidió que el robot gire *hacia donde apunta la persona*. Como la persona está de frente al robot, su izquierda es la derecha del robot: cuando la persona señala a su izquierda (`POINT_LEFT`), el robot ejecuta `TURN_RIGHT`, y cuando señala a su derecha (`POINT_RIGHT`), ejecuta `TURN_LEFT`. Esta convención se controla con una única constante (`POINT_TURNS_TOWARD_POINTED_SIDE`) para poder invertirla si las pruebas con el robot muestran que resulta poco natural.

**Validación del entorno y orden de decisión.** En cada ciclo el cerebro obtiene el gesto y su confianza, lee el contexto con `sense`, el `ContextDecider` propone una acción con el modelo contextual y la capa de seguridad (`validate`) la comprueba y, si corresponde, la sustituye: la seguridad siempre tiene la última palabra. Devuelve una decisión con la acción final, sus parámetros, una razón en español que el robot pronuncia, el color de ojos y un indicador de si modificó la propuesta. Sin un modelo entrenado, el decisor recurre a una política de reglas derivada de la intención del gesto, que también pasa por `validate`; esa política es la línea base con que se compara el modelo aprendido en la Sección 4. Las reglas y umbrales provienen de la especificación del proyecto:

| Acción propuesta | Condición exigida | Si no se cumple | Umbral |
|---|---|---|---|
| Avanzar | Sonares frontales libres | Avance reducido o rechazo | `SAFE_FRONT` = 0.30 m; paso de 0.20 m; mínimo de 0.05 m |
| Girar | Sonar del lado hacia el que se gira, libre | Rechazo con aviso de obstáculo | `SAFE_SIDE` = 0.25 m |
| Saludar | De pie, sin movimiento, persona no demasiado cerca | Aviso de que debe detenerse o aviso de cercanía | `TOO_CLOSE_HAND` = 0.25 (tamaño de mano) |
| Sentarse o pararse | Robot sin movimiento | Aviso de que debe detenerse primero | n. a. |
| Detenerse | Siempre permitido | n. a. | n. a. |
| Cualquier acción | Confianza del gesto suficiente | Pide repetir el gesto | `MIN_CONF` = 0.6 |

Una lectura de sonar inválida (menor o igual que cero, o no numérica) se trata como insegura. Por la ausencia de un sensor trasero, el repertorio no incluye movimiento hacia atrás.

**Avance graduado.** Cuando el obstáculo está cerca pero más allá del margen de seguridad, el robot no se limita a aceptar o rechazar: avanza únicamente la distancia segura. Si la lectura frontal mínima es $d$, la distancia que puede avanzar es $\min(d - \text{SAFE\_FRONT},\ \text{ADVANCE\_STEP})$ [[PENDIENTE: verificar en safety.py la fórmula exacta del avance parcial]]. Si esa distancia es menor que `MIN_STEP`, el movimiento se rechaza. Con una caja a 20 cm el robot rechaza el avance; con la caja a 40 cm avanza una distancia de unos 10 cm y lo comunica.

**Retroalimentación.** El color de los ojos comunica el estado: verde cuando la orden se acepta, amarillo cuando es dudosa o está limitada (por ejemplo, un avance parcial), rojo cuando se rechaza por seguridad y azul cuando espera una orden. Cada rechazo o limitación se acompaña de una explicación hablada, por ejemplo "Hay una caja delante, no puedo avanzar", y de la misma razón en pantalla. Si la confianza es menor que el umbral, el robot pide repetir el gesto en lugar de actuar.

**Suavizado temporal.** Una predicción por fotograma es ruidosa. El `GesturePredictor` mantiene una ventana deslizante de $T = 7$ fotogramas y emite la etiqueta más frecuente:

$$\hat{y}_t = \arg\max_{c} \sum_{k=t-T+1}^{t} \mathbb{1}\left[ \hat{y}_k = c \right]$$

junto con su confianza. Para que un gesto sostenido no dispare órdenes continuas, tras ejecutar una acción se aplica un periodo de espera (*cooldown*) de [[PENDIENTE: segundos, según brain/main.py]] antes de aceptar otra orden. Estos mecanismos introducen una latencia deliberada a cambio de estabilidad, cuyo efecto se mide en la Sección 4.c.

### g. Arquitectura multimodal: gestos y voz, con dos modos

El sistema combina dos canales de entrada que producen intenciones del mismo vocabulario compartido (`brain/vocab.py`): los gestos, clasificados por el modelo de la sección 3.e, y la voz, transcrita y clasificada con Groq (sección 3.i). Se definen dos ejes independientes: el **modo** (MOVIMIENTO o MÚSICA) y la **entrada** (GESTOS o VOZ). El mismo gesto tiene significados distintos según el modo: `THUMBS_UP` avanza en MOVIMIENTO y reproduce música en MÚSICA.

Dos controles funcionan en cualquier combinación, para que el usuario nunca quede atrapado en un canal y la parada no dependa de la voz: `VICTORY` alterna la entrada y `OPEN_PALM` detiene de inmediato movimiento, baile, habla y música. El cambio de modo exige sostener dos pulgares arriba durante unos 1 s, para evitar cambios accidentales. Cada cambio de estado se confirma hablando y con el color de los ojos (azul en MOVIMIENTO, morado en MÚSICA), y los LED de las orejas indican que la entrada es voz. La voz se procesa en un hilo propio: la cámara y la parada siguen respondiendo mientras se transcribe.

### h. Riesgo de confusión entre LOVE_YOU y VICTORY

`LOVE_YOU` (pulgar, índice y meñique extendidos) y `VICTORY` (índice y medio extendidos) comparten el gesto de dedos extendidos y pueden confundirse. `LOVE_YOU` se agregó después de las siete clases originales, por lo que su error se medirá con muestras grabadas por Matías y Vivianne. La métrica se calcula en `brain/gestures/train.py` y se guarda en `report/metrics/gesture_metrics.json`, clave `love_you_victory_confusion`, con las tasas `love_you_as_victory` y `victory_as_love_you`: [[PENDIENTE: valores tras la grabación de LOVE_YOU y el reentrenamiento]].

Mitigación: la opción `--confirm-ambiguous` pide repetir el gesto antes de actuar con cualquiera de los dos. Está desactivada por defecto. Si el error medido resulta alto, la confirmación se activa en la demostración; esa decisión depende del valor medido [[PENDIENTE: decisión del equipo tras la métrica]].

### i. Uso de Groq para la entrada por voz

Al pulsar `v` en entrada VOZ, el audio se graba y se envía a Groq con el modelo Whisper `whisper-large-v3-turbo` (idioma español). La transcripción se clasifica con el modelo `gpt-oss-20b`, con salida JSON restringida a las intenciones de voz permitidas (`vocab.VOICE_INTENTS`), de modo que el modelo no puede proponer una intención fuera de ese conjunto. Si la llamada falla (sin red, clave inválida o tiempo de espera agotado), se usa un respaldo por palabras clave (`brain/voice/keywords.py`). Las llamadas corren en un hilo separado y el ciclo de cámara no espera su respuesta.

---

## 4. Resultados

> Todos los valores de esta sección se completarán a partir de `report/metrics/*.json`, de los archivos de `report/figures/` y del protocolo de pruebas con el robot. No se incluye ningún dato hasta que provenga de una ejecución real.

### a. Métricas

**Reconocimiento de gestos (Random Forest frente a SVM).** Evaluación con `GroupKFold` por persona (Sección 3.d). Se reportan la media y la desviación estándar entre particiones.

| Modelo | Hiperparámetros óptimos | Accuracy (media ± d.e.) | F1 macro (media ± d.e.) |
|---|---|---|---|
| Random Forest | [[PENDIENTE: mejores hiperparámetros del RF de gestos]] | [[PENDIENTE: accuracy RF]] | [[PENDIENTE: F1 macro RF]] |
| SVM (RBF) | [[PENDIENTE: mejores C y gamma]] | [[PENDIENTE: accuracy SVM]] | [[PENDIENTE: F1 macro SVM]] |

Precisión, exhaustividad y F1 por clase de ambos modelos: [[PENDIENTE: tabla por clase para RF y SVM]].

**Decisión contextual (Árbol de Decisión, Random Forest y política de solo reglas).** Evaluación con `GroupKFold` por escenario. La política de solo reglas no se entrena y se evalúa sobre los mismos escenarios.

| Modelo | Hiperparámetros óptimos | Accuracy (media ± d.e.) | F1 macro (media ± d.e.) |
|---|---|---|---|
| Árbol de Decisión | [[PENDIENTE: max_depth, min_samples_leaf y criterion óptimos]] | [[PENDIENTE: accuracy DT]] | [[PENDIENTE: F1 macro DT]] |
| Random Forest | [[PENDIENTE: n_estimators, max_depth y min_samples_leaf óptimos]] | [[PENDIENTE: accuracy RF]] | [[PENDIENTE: F1 macro RF]] |
| Política de solo reglas | no aplica | [[PENDIENTE: accuracy de las reglas]] | [[PENDIENTE: F1 macro de las reglas]] |

**Capa de seguridad.** La regla de seguridad se verificó con casos de prueba con lecturas por debajo y por encima de los umbrales, incluidos casos exactamente en el umbral. Resultado: [[PENDIENTE: número de casos probados y cuántos cumplieron la regla]]. Número de escenarios en los que la capa de seguridad modificó la decisión del modelo aprendido: [[PENDIENTE: conteo]].

### b. Gráficas

**Figura 1.** Matriz de confusión del Random Forest de gestos (`report/figures/gesture_cm_rf.png`).

[[PENDIENTE: insertar gesture_cm_rf.png y leyenda con el conjunto sobre el que se calculó]]

**Figura 2.** Matriz de confusión del SVM de gestos (`report/figures/gesture_cm_svm.png`).

[[PENDIENTE: insertar gesture_cm_svm.png]]

**Figura 3.** Curva de aprendizaje del mejor modelo de gestos (`report/figures/gesture_learning_curve.png`).

[[PENDIENTE: insertar gesture_learning_curve.png]]

**Figura 4.** Comparación de modelos de gestos (`report/figures/gesture_model_comparison.png`).

[[PENDIENTE: insertar gesture_model_comparison.png]]

**Figura 5.** Matriz de confusión del Árbol de Decisión contextual (`report/figures/context_cm_dt.png`).

[[PENDIENTE: insertar context_cm_dt.png]]

**Figura 6.** Matriz de confusión del Random Forest contextual (`report/figures/context_cm_rf.png`).

[[PENDIENTE: insertar context_cm_rf.png]]

**Figura 7.** Estructura del árbol de decisión entrenado (`report/figures/context_tree.png`).

[[PENDIENTE: insertar context_tree.png y comentar las reglas principales que se observan]]

**Figura 8.** Importancia de atributos del modelo contextual (`report/figures/context_feature_importance.png`).

[[PENDIENTE: insertar context_feature_importance.png y comentar los atributos más influyentes]]

### c. Comportamiento observado

Se definió un protocolo de pruebas en el que cada escenario se repite 10 veces con el robot. Un intento se considera exitoso si la acción ejecutada por el robot, su explicación y el color de sus ojos coinciden con lo esperado. La latencia se mide desde que el gesto es sostenido frente a la cámara hasta que el robot inicia la acción.

| Escenario | Gesto | Condición del entorno | Comportamiento esperado | Rep. | Éxitos | Latencia media (s) |
|---|---|---|---|---|---|---|
| E1 | `THUMBS_UP` | Camino libre | Avanza 20 cm, ojos verdes | 10 | [[PENDIENTE: éxitos]] | [[PENDIENTE: latencia]] |
| E2 | `THUMBS_UP` | Caja a 20 cm | Rechaza, explica el obstáculo, ojos rojos | 10 | [[PENDIENTE: éxitos]] | [[PENDIENTE: latencia]] |
| E3 | `THUMBS_UP` | Caja a 40 cm | Avance parcial, ojos amarillos | 10 | [[PENDIENTE: éxitos]] | [[PENDIENTE: latencia]] |
| E4 | `POINT_LEFT` / `POINT_RIGHT` | Obstáculo del lado hacia el que apunta | Rechaza el giro y lo explica | 10 | [[PENDIENTE: éxitos]] | [[PENDIENTE: latencia]] |
| E5 | `POINT_LEFT` / `POINT_RIGHT` | Lado libre | Gira 45° hacia lo señalado | 10 | [[PENDIENTE: éxitos]] | [[PENDIENTE: latencia]] |
| E6 | `VICTORY` | Persona muy cerca | Rechaza el saludo y avisa de la cercanía | 10 | [[PENDIENTE: éxitos]] | [[PENDIENTE: latencia]] |
| E7 | `OPEN_PALM` | Robot en movimiento | Se detiene | 10 | [[PENDIENTE: éxitos]] | [[PENDIENTE: latencia]] |
| E8 | `FIST` | Robot en movimiento | Pide detenerse antes de sentarse | 10 | [[PENDIENTE: éxitos]] | [[PENDIENTE: latencia]] |
| E9 | Gesto ambiguo | Confianza baja | Pide confirmación, no actúa | 10 | [[PENDIENTE: éxitos]] | [[PENDIENTE: latencia]] |

Descripción cualitativa del comportamiento (fluidez, falsos disparos, efecto de la iluminación, estabilidad de la conexión): [[PENDIENTE: observaciones de las pruebas con el robot Zyra]]. Plataforma de las pruebas: [[PENDIENTE: indicar qué escenarios se probaron en robot físico y cuáles en simulador]].

### d. Comparación entre configuraciones

Se comparan tres configuraciones del decisor contextual: solo reglas, Árbol de Decisión más capa de seguridad y Random Forest más capa de seguridad; y dos algoritmos para el reconocimiento de gestos. La comparación contrasta métricas sobre datos grabados con el desempeño en el robot (Sección 4.c) y la tasa de intervención de la capa de seguridad.

| Configuración | F1 macro (escenarios) | Éxito medio en el robot | Intervenciones de la capa de seguridad |
|---|---|---|---|
| Solo reglas + seguridad | [[PENDIENTE: F1 macro]] | [[PENDIENTE: éxito medio]] | [[PENDIENTE: intervenciones]] |
| Árbol de Decisión + seguridad | [[PENDIENTE: F1 macro]] | [[PENDIENTE: éxito medio]] | [[PENDIENTE: intervenciones]] |
| Random Forest + seguridad | [[PENDIENTE: F1 macro]] | [[PENDIENTE: éxito medio]] | [[PENDIENTE: intervenciones]] |

---

## 5. Discusión

### a. Interpretación

La interpretación se basará estrictamente en los valores de la Sección 4. Las pautas siguientes indican cómo se leerá cada resultado.

- **Gestos, Random Forest frente a SVM.** Si el SVM supera al Random Forest, sería coherente con que los atributos normalizados forman fronteras suaves en las que el núcleo RBF es efectivo; si el Random Forest es igual o superior, bastarían umbrales sobre extensión de dedos y ángulos. La diferencia debe contrastarse con la desviación estándar entre personas: si es menor que ella, no puede afirmarse superioridad. Resultado: [[PENDIENTE: interpretación de la comparación con las cifras de la Sección 4.a]].
- **Errores entre clases.** Confusiones entre `POINT_LEFT` y `POINT_RIGHT` indicarían sensibilidad a la lateralidad; entre `FIST` y `THUMBS_UP`, que el pulgar es difícil de distinguir con la mano cerrada; y entre `NONE` y cualquier gesto, que la mano en reposo se parece a una orden. Confusiones observadas: [[PENDIENTE: pares de clases más confundidos según las matrices de confusión]].
- **Curva de aprendizaje.** Si las curvas convergen a un valor alto, el modelo está limitado por sesgo y más datos aportarían poco; si persiste una brecha, hay sobreajuste y más personas ayudarían. Observación: [[PENDIENTE: interpretación de la curva de aprendizaje]].
- **Contexto, Árbol frente a Random Forest y reglas.** Si el Random Forest supera al árbol, la diferencia se atribuye a la menor varianza del conjunto; si son similares, es preferible el árbol por interpretabilidad. Si el modelo aprendido no supera a la política de solo reglas, su valor residiría en los casos que las reglas no cubren; si la supera, en que captura criterios humanos de los etiquetadores. Resultado: [[PENDIENTE: interpretación con la Sección 4.a y las Figuras 7 y 8]].
- **Capa de seguridad y robot real.** Si el modelo ya evitaba avanzar contra obstáculos, la regla es una red de seguridad redundante; si hubo casos en que la corrigió, una garantía determinista no puede sustituirse por un modelo entrenado con pocos escenarios. La brecha entre las métricas sobre datos grabados y el éxito en el robot (Sección 4.c) cuantifica el efecto de condiciones no vistas. Observación: [[PENDIENTE: interpretar intervenciones, éxito y latencia, incluidos el avance parcial y la petición de confirmación]].

### b. Limitaciones

- **SDK en Python 2.7.** Obliga a una arquitectura de dos procesos y a mantener un intérprete obsoleto, lo que agrega complejidad de despliegue y un punto adicional de fallo.
- **Cámara del portátil y una sola mano.** El sistema solo reconoce una mano entre 0.5 m y 1.5 m de la cámara del portátil; si la persona sale del encuadre, el robot no puede reaccionar, y la iluminación deficiente degrada la detección.
- **Cobertura del sonar.** Los sonares cubren solo un cono frontal, tienen una distancia mínima y pueden dar lecturas falsas por reflejos o superficies inclinadas o absorbentes; la capa de seguridad es tan fiable como esa lectura.
- **Sin movimiento hacia atrás.** El robot carece de sensor trasero, por lo que ante un obstáculo solo puede negarse o girar.
- **Conjunto de datos pequeño y latencia.** Cuatro personas y pocos escenarios limitan la generalización, y con cuatro grupos la validación por persona tiene alta varianza. Latencia de red observada: [[PENDIENTE: valores medidos y variabilidad]].

### c. Propuestas de mejora

1. **Enseñanza de gestos nuevos con pocos ejemplos.** Permitir que el usuario enseñe un gesto nuevo con unas pocas repeticiones (por ejemplo, vecinos más cercanos sobre los atributos de la mano) y le asocie un movimiento, sin reentrenar el sistema.
2. **Cámara del NAO y detección de objetos.** Fusionar los sonares con detección de objetos sobre las cámaras del robot (`ALVideoDevice`) para ver obstáculos laterales o a otras alturas.
3. **Varias manos.** Reconocer gestos de las dos manos o de varias personas y decidir a quién obedecer.
4. **Secuencias de gestos como programas.** Interpretar secuencias de gestos como pequeños programas que el robot valide paso a paso antes de ejecutarlos.
5. **Más datos.** Ampliar gestos y escenarios con más personas, iluminaciones y obstáculos laterales.

---

## 6. Conclusiones

1. **Sobre el reconocimiento de gestos (OE1).** [[PENDIENTE: el algoritmo (RF o SVM) obtuvo el mejor F1 macro, de __ ± __ con GroupKFold por persona, frente a __ ± __ del otro; la diferencia es / no es mayor que la desviación entre personas, por lo que se concluye que ...]]. La clase con peor desempeño fue [[PENDIENTE: clase]] (F1 = [[PENDIENTE: valor]]), lo que indica [[PENDIENTE: causa probable según la matriz de confusión]].
2. **Sobre la decisión contextual (OE2).** [[PENDIENTE: el modelo (DT o RF) alcanzó un F1 macro de __ frente a __ del otro y a __ de la política de solo reglas, y los atributos más influyentes fueron __, lo que confirma / contradice la expectativa de que los sonares y la cercanía de la mano determinan la acción]]. Con [[PENDIENTE: n]] escenarios reales, el resultado es [[PENDIENTE: valoración sobre si el tamaño de los datos fue suficiente]].
3. **Sobre la integración y la seguridad (OE3).** [[PENDIENTE: el robot ejecutó el comportamiento esperado en __ de 10 repeticiones en promedio, con latencia media de __ s; con la caja a 20 cm la capa de seguridad evitó el avance en __ de 10 intentos y con la caja a 40 cm el avance parcial se produjo en __ de 10]]. Esto indica que [[PENDIENTE: juicio sobre si la combinación de modelo aprendido, regla determinista, explicación y confirmación es adecuada para un robot que comparte espacio con personas]].
4. **Sobre el enfoque "razonar antes de obedecer".** [[PENDIENTE: valoración de si las explicaciones habladas y la petición de confirmación en gestos ambiguos redujeron los movimientos indebidos, con base en el escenario de gesto ambiguo]].
5. **Sobre la arquitectura.** El puente persistente en Python 2.7 con protocolo JSON permitió usar MediaPipe en Python 3 y el SDK de NAOqi en Python 2.7 simultáneamente; el costo observado fue una latencia adicional de [[PENDIENTE: ms del viaje de ida y vuelta del puente]], aceptable / no aceptable para [[PENDIENTE: el tipo de órdenes del sistema]].

---

## 7. Referencias

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5-32. https://doi.org/10.1023/A:1010933404324

Breiman, L., Friedman, J. H., Olshen, R. A., y Stone, C. J. (1984). *Classification and regression trees*. Wadsworth.

Cortes, C., y Vapnik, V. (1995). Support-vector networks. *Machine Learning, 20*(3), 273-297. https://doi.org/10.1007/BF00994018

Doshi-Velez, F., y Kim, B. (2017). *Towards a rigorous science of interpretable machine learning*. arXiv. https://arxiv.org/abs/1702.08608

Gouaillier, D., Hugel, V., Blazevic, P., Kilner, C., Monceaux, J., Lafourcade, P., Marnier, B., Serre, J., y Maisonnier, B. (2009). Mechatronic design of NAO humanoid. En *2009 IEEE International Conference on Robotics and Automation*. IEEE. [[PENDIENTE: verificar rango de páginas y DOI]]

Hastie, T., Tibshirani, R., y Friedman, J. (2009). *The elements of statistical learning: Data mining, inference, and prediction* (2.ª ed.). Springer.

Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., Zhang, F., Chang, C.-L., Yong, M. G., Lee, J., Chang, W.-T., Hua, W., Georg, M., y Grundmann, M. (2019). *MediaPipe: A framework for building perception pipelines*. arXiv. https://arxiv.org/abs/1906.08172

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., y Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825-2830.

Platt, J. C. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. En A. J. Smola, P. Bartlett, B. Schölkopf y D. Schuurmans (Eds.), *Advances in large margin classifiers* (pp. 61-74). MIT Press.

SoftBank Robotics. (s. f.). *NAO documentation* [Documentación técnica del fabricante]. http://doc.aldebaran.com

Zhang, F., Bazarevsky, V., Vakunov, A., Tkachenka, A., Sung, G., Chang, C.-L., y Grundmann, M. (2020). *MediaPipe Hands: On-device real-time hand tracking*. arXiv. https://arxiv.org/abs/2006.10214
