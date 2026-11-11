<!--
BORRADOR T4 (S6). Total de marcadores PENDIENTE (formato doble corchete): 76.
Todos los valores numericos de resultados, figuras y pruebas del robot estan como marcadores y deben completarse
desde report/metrics/*.json, report/figures/*.png y el protocolo de pruebas con el robot (T7/T8).
Formato final: PDF, Times New Roman 12, interlineado 1.5, margenes 2.5 cm, 8-15 paginas.
-->

# NAO Gesture Pilot: control gestual inteligente con decisión contextual para el robot NAO

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

La inteligencia artificial aplicada a la robótica busca que una máquina perciba su entorno, interprete esa información y actúe en consecuencia. En robótica social y de servicio, esta cadena percepción–decisión–acción es especialmente relevante porque el robot comparte el espacio con personas que no son expertas y que esperan comunicarse con él de forma natural, sin teclados ni interfaces especializadas. Los gestos corporales son uno de los canales más inmediatos para ello: levantar una mano para indicar "detente" o señalar hacia un lado para indicar un giro son convenciones que cualquier persona entiende sin entrenamiento previo.

Los avances recientes en estimación de pose en tiempo real sobre CPU (Bazarevsky et al., 2020; Lugaresi et al., 2019) permiten convertir una imagen de una cámara convencional en un conjunto reducido de coordenadas corporales, sobre el cual pueden entrenarse modelos clásicos de aprendizaje supervisado de bajo costo computacional (Breiman, 2001; Cortes y Vapnik, 1995). Esta combinación hace viable un sistema de control gestual que se ejecute en una computadora portátil y gobierne un robot humanoide como el NAO (Gouaillier et al., 2009) sin hardware especializado.

### b. Motivación

La propuesta de este trabajo parte de una idea sencilla: una persona podrá indicarle al robot, mediante gestos captados por una cámara, que avance, se detenga o gire hacia la izquierda o hacia la derecha. Sin embargo, un robot que obedece cada gesto de forma automática es inseguro: una orden de avanzar es incorrecta si hay un obstáculo delante, y una orden dada por una persona demasiado lejana de la cámara tiene baja confiabilidad. La motivación central del grupo es, por tanto, que el NAO no obedezca automáticamente, sino que considere variables como la distancia, la presencia de obstáculos y su estado actual para decidir qué acción realizar.

Esto da lugar a un sistema con dos modelos de aprendizaje automático encadenados: un clasificador de gestos y un modelo de decisión contextual. Además, el problema permite construir un conjunto de datos propio, recolectado por los integrantes del grupo, lo que aporta originalidad frente a un conjunto de datos público genérico.

### c. Objetivos

**Objetivo general.** Diseñar, entrenar e integrar en el robot NAO un sistema de control gestual inteligente que reconozca gestos humanos mediante una cámara y decida la acción motora del robot considerando el contexto sensorial (sonares, distancia a la persona y estado de movimiento).

**Objetivos específicos.**

1. **OE1.** Construir un conjunto de datos propio de gestos y entrenar y comparar dos algoritmos (Random Forest y Máquina de Vector Soporte) para clasificar cinco clases de gesto a partir de puntos de referencia corporales, evaluándolos con validación que separe a las personas.
2. **OE2.** Construir un conjunto de datos de escenarios reales etiquetados por humanos y entrenar y comparar dos algoritmos (Árbol de Decisión y Random Forest) para decidir la acción del robot en función de gesto, confianza, sonares, distancia a la persona y estado de movimiento.
3. **OE3.** Integrar ambos modelos con el NAO mediante un puente de comunicación Python 3 – Python 2.7, incorporando una capa de seguridad que impida avanzar ante obstáculos, y verificar experimentalmente el comportamiento del robot.

### d. Alcance y limitaciones

El alcance comprende cinco clases de gesto (`ADVANCE`, `STOP`, `TURN_LEFT`, `TURN_RIGHT` y `NONE`), siete acciones de salida del robot (`ADVANCE`, `TURN_LEFT`, `TURN_RIGHT`, `STOP`, `WARN_OBSTACLE`, `ASK_CLOSER` y `WAIT`), una sola persona frente a la cámara a la vez y un entorno interior de laboratorio. La visión se realiza con la cámara de la computadora portátil y no con las cámaras del robot, y el cálculo se ejecuta en la computadora, no a bordo del NAO.

Las principales limitaciones son: (i) el conjunto de datos propio proviene de solo cuatro personas, por lo que la capacidad de generalización a personas nuevas es limitada y debe interpretarse con cautela; (ii) el conjunto de escenarios con el robot es de tamaño reducido por el tiempo disponible de acceso al robot; (iii) el SDK Python de NAOqi requiere Python 2.7, lo que obliga a una arquitectura de dos procesos; y (iv) no se evalúan entornos exteriores ni múltiples personas simultáneas. Se detallan en la Sección 5.b.

---

## 2. Marco Teórico

### Visión por computador y estimación de pose

La estimación de pose humana consiste en localizar, a partir de una imagen, las articulaciones o puntos de referencia (*landmarks*) del cuerpo. MediaPipe Pose, basado en el modelo BlazePose, predice 33 puntos de referencia corporales por persona con un modelo ligero diseñado para ejecutarse en tiempo real en CPU de dispositivos móviles y de escritorio (Bazarevsky et al., 2020; Lugaresi et al., 2019). Cada punto se entrega con coordenadas normalizadas $(x, y)$ respecto al ancho y alto de la imagen, una coordenada de profundidad relativa $z$ y una puntuación de visibilidad $v \in [0,1]$ que estima la probabilidad de que el punto sea visible y no esté ocluido.

Esta representación es el fundamento de nuestro enfoque: en lugar de clasificar píxeles, se clasifican vectores numéricos de baja dimensión que ya codifican la geometría del cuerpo. Esto reduce el volumen de datos necesario, elimina la dependencia del fondo y de la ropa, y permite entrenar modelos clásicos en segundos.

### Random Forest

Un Random Forest (Breiman, 2001) es un conjunto (*ensemble*) de $B$ árboles de decisión entrenados sobre muestras *bootstrap* del conjunto de entrenamiento (*bagging*). En cada nodo, cada árbol considera solo un subconjunto aleatorio de $m$ atributos candidatos para la partición, lo que decorrelaciona los árboles y reduce la varianza del conjunto. En clasificación, la predicción se obtiene por voto mayoritario:

$$\hat{y}(\mathbf{x}) = \arg\max_{c} \sum_{b=1}^{B} \mathbb{1}\left[ T_b(\mathbf{x}) = c \right]$$

donde $T_b$ es el árbol $b$-ésimo. Cada partición se elige para minimizar la impureza de Gini del nodo hijo:

$$G(t) = 1 - \sum_{c=1}^{C} p_{c,t}^{2}$$

siendo $p_{c,t}$ la proporción de muestras de la clase $c$ en el nodo $t$. Los hiperparámetros principales son el número de árboles `n_estimators`, la profundidad máxima `max_depth`, el número mínimo de muestras por hoja `min_samples_leaf`, el número de atributos por partición `max_features` y el criterio de impureza. Un efecto secundario útil es que el modelo produce una medida de importancia de atributos, basada en la reducción acumulada de impureza.

### Máquinas de Vector Soporte

Una Máquina de Vector Soporte (SVM; Cortes y Vapnik, 1995) busca el hiperplano que separa las clases con el mayor margen posible. Para clasificación binaria, la función de decisión es:

$$f(\mathbf{x}) = \operatorname{sign}\left( \sum_{i \in SV} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b \right)$$

donde $SV$ es el conjunto de vectores soporte, $\alpha_i$ son los multiplicadores de Lagrange, $y_i \in \{-1,+1\}$ las etiquetas y $K$ una función núcleo (*kernel*). Para datos no linealmente separables se emplea el núcleo de base radial (RBF):

$$K(\mathbf{x}, \mathbf{x}') = \exp\left( -\gamma \, \lVert \mathbf{x} - \mathbf{x}' \rVert^{2} \right)$$

Los hiperparámetros críticos son $C$, que controla el compromiso entre maximizar el margen y penalizar errores de entrenamiento (un $C$ grande tolera menos errores y puede sobreajustar), y $\gamma$, que define el alcance de influencia de cada vector soporte (un $\gamma$ grande produce fronteras más complejas). El problema multiclase se resuelve combinando clasificadores binarios (estrategia uno contra uno en la implementación de scikit-learn; Pedregosa et al., 2011). Como el núcleo RBF depende de distancias, los atributos deben estandarizarse previamente.

### Árboles de decisión

Un árbol de decisión (Breiman et al., 1984) particiona recursivamente el espacio de atributos mediante condiciones del tipo $x_j \le \tau$, eligiendo en cada nodo el par (atributo, umbral) que maximiza la ganancia de pureza según Gini. Su principal ventaja es la interpretabilidad: el modelo entrenado puede dibujarse y leerse como un conjunto de reglas del tipo "si el sonar izquierdo es menor que cierto valor y el gesto es avanzar, entonces advertir obstáculo". Su principal desventaja es la alta varianza: pequeños cambios en los datos pueden producir árboles distintos y un árbol profundo tiende a sobreajustar. Los hiperparámetros principales son `max_depth`, `min_samples_split`, `min_samples_leaf` y `ccp_alpha` (poda por costo-complejidad).

### Criterio de selección frente a alternativas

La elección de modelos se justifica por las características del problema y no por la sofisticación del algoritmo:

- **Conjunto de datos pequeño y propio.** Con datos de cuatro personas, una red convolucional de extremo a extremo (CNN) entrenada sobre imágenes requeriría muchos más ejemplos para generalizar y es propensa al sobreajuste a fondo, ropa e iluminación. Trabajar sobre *landmarks* ya extrae la información geométrica relevante.
- **Atributos tabulares de baja dimensión.** Coordenadas normalizadas y ángulos son atributos numéricos tabulares, para los que los métodos basados en árboles y los SVM con núcleo suelen ser competitivos con costo bajo (Hastie et al., 2009).
- **Ejecución en tiempo real en CPU.** La inferencia de un Random Forest o un SVM sobre un vector de decenas de atributos toma una fracción pequeña del tiempo de un fotograma, por lo que el cuello de botella es la estimación de pose y no la clasificación.
- **Interpretabilidad.** El árbol de decisión permite explicar por qué el robot eligió una acción, lo cual es deseable en un sistema con una capa de seguridad; la importancia de atributos del Random Forest complementa este análisis.
- **Frente a KNN.** K vecinos más cercanos no requiere entrenamiento, pero la inferencia depende del tamaño del conjunto almacenado, es sensible a la escala de atributos y no ofrece un modelo explicable ni importancia de atributos. Se consideró una alternativa razonable pero no se incluyó en la comparación para mantener el trabajo acotado a dos algoritmos por modelo.

Se comparan Random Forest y SVM para el reconocimiento de gestos (representaciones continuas, geometría suave) y Árbol de Decisión y Random Forest para la decisión contextual (variables mixtas, umbrales y reglas), de modo que en cada caso se contrasta un modelo más interpretable con uno más robusto.

### Robot NAO

NAO es un robot humanoide de unos 58 cm de altura desarrollado por Aldebaran Robotics y luego SoftBank Robotics (Gouaillier et al., 2009). Dispone de 25 grados de libertad (DoF) repartidos en cabeza, brazos, manos, piernas y cadera, lo que permite caminar, girar y gesticular. Su arquitectura sensorial incluye:

- **Dos cámaras** frontales (frente y boca), usadas para visión.
- **Dos pares emisor-receptor de sonar** (izquierdo y derecho) para estimar la distancia a obstáculos frontales, con un cono de detección limitado y una distancia mínima por debajo de la cual la lectura no es confiable (valor a confirmar en la documentación del fabricante).
- **Una unidad de medición inercial (IMU)** con giroscopio y acelerómetro, usada para estimar la orientación y el equilibrio.
- **Sensores de fuerza resistivos (FSR)** en las plantas de los pies para estimar la presión y el centro de presión durante la marcha.
- **Sensores táctiles** en la cabeza y las manos, además de micrófonos y altavoces.

El sistema operativo es NAOqi OS, una distribución Linux sobre la cual corre el middleware NAOqi, que expone la funcionalidad del robot como módulos accesibles localmente o por red mediante *proxies* (SoftBank Robotics, s. f.). Los módulos relevantes para este trabajo son: `ALMotion` (locomoción y articulaciones, por ejemplo `moveTo` y `stopMove`), `ALTextToSpeech` (síntesis de voz), `ALMemory` (lectura de valores de sensores, como los sonares) y `ALVideoDevice` (acceso a las cámaras). El SDK es accesible desde Python y C++, y se dispone también de Choregraphe, un entorno gráfico de programación por bloques. Un punto crítico para este trabajo es que el SDK Python de NAOqi (`naoqi`, módulo `ALProxy`) corresponde a Python 2.7, mientras que MediaPipe requiere una versión moderna de Python 3; esta incompatibilidad determina la arquitectura descrita en la Sección 3.f.

---

## 3. Metodología

### a. Descripción del problema

El problema es doble. Primero, clasificar el gesto que realiza una persona frente a una cámara en una de cinco clases a partir de los 33 *landmarks* de MediaPipe Pose (problema de clasificación supervisada multiclase). Segundo, decidir qué acción motora ejecuta el NAO dados el gesto reconocido y el contexto: gesto, confianza del gesto, lecturas de los sonares izquierdo y derecho (m), distancia estimada a la persona y si el robot se encuentra en movimiento (clasificación multiclase en siete acciones). Las salidas del segundo modelo incluyen conductas que ningún gesto ordena directamente: advertir de un obstáculo (`WARN_OBSTACLE`), pedir a la persona que se acerque (`ASK_CLOSER`) y esperar (`WAIT`).

La tabla siguiente resume las clases de gesto. El gesto físico asociado a cada clase se fijó antes de la recolección para que los cuatro integrantes lo realizaran de forma consistente.

| Clase | Orden transmitida | Descripción del gesto físico |
|---|---|---|
| `ADVANCE` | avanzar | [[PENDIENTE: describir el gesto físico acordado]] |
| `STOP` | detenerse | [[PENDIENTE: describir el gesto físico acordado]] |
| `TURN_LEFT` | girar a la izquierda | [[PENDIENTE: describir el gesto físico acordado]] |
| `TURN_RIGHT` | girar a la derecha | [[PENDIENTE: describir el gesto físico acordado]] |
| `NONE` | sin orden | [[PENDIENTE: describir la postura neutra y movimientos espurios incluidos]] |

### b. Origen y características del dataset

Se utilizan dos conjuntos de datos propios, ambos recolectados por el grupo, sin conjuntos públicos.

**Conjunto de gestos.** Los cuatro integrantes del grupo realizaron cada uno de los cinco gestos frente a la cámara web de una computadora portátil. Cada fotograma pasó por MediaPipe Pose (`mediapipe==0.10.14`, solución `mp.solutions.pose`) y se guardó como una fila de un archivo CSV por persona, clase y marca de tiempo, con las columnas `person`, `label` y, para cada uno de los 33 puntos, `x`, `y`, `z` y `v` (visibilidad). Se solicitó variar la distancia a la cámara, la posición lateral y la iluminación dentro de lo razonable para el entorno de laboratorio. Tamaño del conjunto: [[PENDIENTE: número total de muestras, por persona y por clase (tabla de distribución)]]; condiciones de captura: [[PENDIENTE: resolución, tasa de cuadros, iluminación, distancia aproximada a la cámara]].

**Conjunto contextual.** Reúne escenarios reales con el robot (modelo Zyra, y el simulador cuando fue necesario), en los que se registran las variables de entrada y un integrante del grupo etiqueta, según su criterio, la acción que el robot debería ejecutar. El archivo `scenarios.csv` contiene las columnas `gesture`, `gesture_confidence`, `sonar_left`, `sonar_right`, `person_distance`, `moving`, `action` y `scenario_id`. La variable `person_distance` es el ancho de hombros de la persona dividido por el ancho del fotograma, de modo que un valor mayor indica una persona más cercana. Las etiquetas las asignan humanos; no se generan mediante reglas, para evitar que el modelo se limite a reproducir una función conocida. Tamaño: [[PENDIENTE: número de escenarios y distribución por acción]]; número de etiquetadores y criterio de etiquetado: [[PENDIENTE: quién etiquetó y con qué criterio común]].

**Aumento sintético (declarado).** [[PENDIENTE: confirmar si se usó aumento sintético. Si se usó: describir el procedimiento (por ejemplo, ruido gaussiano pequeño sobre lecturas de sonar o de distancia), cuántas muestras sintéticas se agregaron y confirmar que solo se emplearon en entrenamiento]]. Cualquier muestra sintética se identifica como tal, se reporta por separado y no se incluye jamás en los conjuntos de prueba; las métricas de la Sección 4 se calculan únicamente sobre datos reales.

### c. Preprocesamiento

- **Valores faltantes y filtro de visibilidad.** Un punto con visibilidad menor a 0.5 se considera faltante. Si alguno de los puntos clave de la parte superior del cuerpo (hombros, codos, muñecas, caderas) no supera ese umbral, el fotograma se descarta y no genera predicción (`extract_features` devuelve `None`). En ejecución, la ausencia de una pose válida se trata como ausencia de orden.
- **Invarianza a la posición.** Las coordenadas se centran en el punto medio de las caderas, eliminando la dependencia de dónde se ubica la persona en la imagen.
- **Invarianza a la escala.** Se dividen entre el ancho de hombros, eliminando la dependencia de la distancia a la cámara. Esta misma cantidad, normalizada por el ancho del fotograma, se reutiliza como `person_distance` para el modelo contextual.
- **Ángulos.** Se agregan ángulos de codos y hombros calculados a partir de las articulaciones, que son invariantes a traslación y escala y resumen la postura del brazo.
- **Estandarización.** Los atributos se estandarizan con `StandardScaler` (media cero, varianza unitaria) para el SVM, cuyo núcleo RBF es sensible a la escala. Los modelos basados en árboles no la requieren. El escalador se ajusta solo con los datos de entrenamiento de cada partición y se aplica a los de prueba.
- **Codificación de variables.** En el modelo contextual, el gesto, categórico, se codifica en *one-hot* (`ADVANCE`, `STOP`, `TURN_LEFT`, `TURN_RIGHT`, `NONE`), y `moving` se representa como 0/1. La variable objetivo es la acción de siete clases.
- **Limpieza.** [[PENDIENTE: filas eliminadas por visibilidad baja, duplicados o lecturas de sonar inválidas, con conteos]].

### d. División de datos

Una división aleatoria de filas sería inapropiada: los fotogramas consecutivos de una misma persona y gesto son casi idénticos, de modo que una fila de prueba tendría una gemela casi exacta en el entrenamiento y la métrica sería optimista (fuga de información). Por ello se emplea validación cruzada con `GroupKFold`:

- **Gestos:** los grupos son las personas. Con cuatro integrantes, cada partición reserva a una persona completa para prueba y entrena con las otras tres, lo que mide la generalización a una persona no vista, que es el escenario real de uso.
- **Contexto:** los grupos son los escenarios (`scenario_id`), de modo que ninguna variante de un mismo escenario aparezca a la vez en entrenamiento y prueba.

Esquema exacto de evaluación (anidado con ajuste interno, o con conjunto de prueba reservado): [[PENDIENTE: confirmar según el código de entrenamiento de brain/gestures y brain/context]]. Se reporta la media y la desviación estándar de las métricas entre particiones.

### e. Entrenamiento y ajuste de hiperparámetros

Cada algoritmo se ajusta con `GridSearchCV` (Pedregosa et al., 2011) usando la validación por grupos y optimizando F1 macro (`scoring="f1_macro"`), que promedia el F1 de cada clase con el mismo peso y es apropiada cuando las clases no están perfectamente balanceadas, al evitar que una clase mayoritaria domine la métrica. Se exploran las siguientes rejillas:

| Modelo | Algoritmo | Hiperparámetros explorados | Valores |
|---|---|---|---|
| Gestos | Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` | [[PENDIENTE: rejilla exacta]] |
| Gestos | SVM (RBF) | `C`, `gamma` | [[PENDIENTE: rejilla exacta]] |
| Contexto | Árbol de Decisión | `max_depth`, `min_samples_leaf`, `criterion` | [[PENDIENTE: rejilla exacta]] |
| Contexto | Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf` | [[PENDIENTE: rejilla exacta]] |

Las métricas reportadas son exactitud (*accuracy*), F1 macro, matriz de confusión y, para el diagnóstico de sobreajuste y suficiencia de datos, la curva de aprendizaje del mejor modelo de gestos. La semilla aleatoria es fija para que los experimentos sean reproducibles. Los modelos finales se serializan en `models/gesture_model.joblib` y `models/context_model.joblib`.

### f. Estrategia de integración con el NAO

**Arquitectura de dos procesos.** El SDK `naoqi` solo funciona con Python 2.7, mientras que MediaPipe requiere Python 3.11. Siguiendo el patrón del docente (archivo `py27-py3.py`), en el que un programa de Python 3 invoca con `subprocess` al intérprete de Python 2.7 (`C:\Python27\python.exe`) para ejecutar un script que usa `ALProxy`, nuestro sistema separa un "cerebro" en Python 3.11 (visión y modelos) de un "puente" en Python 2.7 (acceso al robot). La diferencia respecto al ejemplo del docente es que éste lanza un proceso nuevo por cada ejecución, lo que obliga a reconectar con el robot cada vez; en nuestro caso el puente es un proceso persistente que mantiene los *proxies* abiertos y se comunica con el cerebro mediante líneas JSON por entrada y salida estándar. El patrón de invocación se adaptó con atribución al código del docente.

```
 Webcam (laptop)
      |
      v
+--------------------------- Python 3.11 ("cerebro") ---------------------------+
| MediaPipe Pose --> extract_features --> GesturePredictor (RF/SVM, voto 7) -----+--> gesto, confianza
|                                                                                |
| NaoClient.sense() <-- sonar_izq, sonar_der, moving                             |
|                                                                                |
| ContextDecider: modelo contextual (DT/RF) --> apply_safety (umbral 0.30 m)     |
|        --> accion --> cooldown --> NaoClient.act(accion)                       |
+------------------------------------+-------------------------------------------+
                                     |  JSON por linea (stdin/stdout)
                                     v
+------------------------- Python 2.7 ("puente", persistente) -------------------+
| nao_bridge.py: ALProxy(ALMotion, ALTextToSpeech, ALMemory, ALLeds)             |
+------------------------------------+-------------------------------------------+
                                     |  NAOqi (red)
                                     v
                          Robot NAO (Zyra / Nyx / simulador)
```

**Protocolo del puente.** Cada mensaje es un objeto JSON en una línea. Las órdenes son `ping` (verifica el estado y si está en modo simulado), `sense` (devuelve `sonar_left`, `sonar_right` y `moving`), `act` (ejecuta una acción), `say` (síntesis de voz) y `quit`. Una acción desconocida produce una respuesta con `ok: false` y un mensaje de error, de modo que el cerebro nunca asume un éxito. El puente acepta `--robot nyx|zyra|sim` o `--ip/--port` y un modo `--mock` que no requiere el SDK, lo que permite probar toda la tubería sin robot. En Python 3 el cliente `NaoClient` lanza el puente una sola vez y expone `ping`, `sense`, `act`, `say` y `close`.

**Tabla de acciones.** La salida del modelo contextual se traduce en movimientos del robot:

| Acción | Respuesta del NAO |
|---|---|
| `ADVANCE` | `ALMotion.moveTo(0.2, 0, 0)`: avanza 0.2 m |
| `TURN_LEFT` | `moveTo(0, 0, +0.785)`: gira 45° a la izquierda |
| `TURN_RIGHT` | `moveTo(0, 0, -0.785)`: gira 45° a la derecha |
| `STOP` | `stopMove()` |
| `WARN_OBSTACLE` | `stopMove()`, dice "Hay un obstáculo, no puedo avanzar" y enciende los LED de los ojos en rojo |
| `ASK_CLOSER` | dice "Acércate, no te veo bien" |
| `WAIT` | no hace nada |

**Capa de seguridad.** Además del modelo aprendido, se aplica una regla determinista de último recurso: si `min(sonar_left, sonar_right) < 0.30` m y la acción decidida es `ADVANCE`, se sustituye por `WARN_OBSTACLE`. La regla vive en una función independiente (`apply_safety`) aplicada después del modelo, de modo que un error del modelo, o su sobreajuste a pocos escenarios, no pueda producir un avance hacia un obstáculo cercano. El umbral de 0.30 m se fijó por encima de la distancia mínima fiable del sonar y como margen razonable de frenado [[PENDIENTE: justificar o ajustar el umbral con las pruebas de sonar]]. El modelo aprendido y la regla de seguridad cumplen funciones distintas: el primero captura la política general aprendida de escenarios etiquetados y la segunda garantiza una propiedad crítica sin depender de datos.

**Suavizado temporal y *cooldown*.** Una predicción por fotograma es ruidosa. El `GesturePredictor` mantiene una ventana deslizante de $T = 7$ fotogramas y emite el gesto más frecuente (voto mayoritario):

$$\hat{y}_t = \arg\max_{c} \sum_{k=t-T+1}^{t} \mathbb{1}\left[ \hat{y}_k = c \right]$$

junto con su confianza, y se exige una confianza mínima de 0.6. Si no se alcanza, no se emite una orden [[PENDIENTE: verificar el comportamiento exacto en predictor.py]]. Para que un mismo gesto sostenido no dispare órdenes continuas, tras ejecutar una acción se aplica un periodo de espera (*cooldown*) de [[PENDIENTE: segundos, según brain/main.py]] antes de aceptar otra. Estos mecanismos introducen una latencia deliberada a cambio de estabilidad, cuyo efecto se mide en la Sección 4.d.

---

## 4. Resultados

> Todos los valores de esta sección se completarán a partir de `report/metrics/*.json` y los archivos de `report/figures/`. No se incluye ningún dato hasta que provenga de una ejecución real.

### a. Reconocimiento de gestos (Random Forest frente a SVM)

Evaluación con `GroupKFold` por persona (Sección 3.d). Se reportan la media y la desviación estándar entre particiones.

| Modelo | Hiperparámetros óptimos | Accuracy (media ± d.e.) | F1 macro (media ± d.e.) |
|---|---|---|---|
| Random Forest | [[PENDIENTE: mejores hiperparámetros de gesture_rf]] | [[PENDIENTE: accuracy RF]] | [[PENDIENTE: F1 macro RF]] |
| SVM (RBF) | [[PENDIENTE: mejores C y gamma]] | [[PENDIENTE: accuracy SVM]] | [[PENDIENTE: F1 macro SVM]] |

**Figura 1.** Matriz de confusión del Random Forest (`report/figures/gesture_cm_rf.png`).

[[PENDIENTE: insertar figura gesture_cm_rf.png y leyenda con el conjunto sobre el que se calculó]]

**Figura 2.** Matriz de confusión del SVM (`report/figures/gesture_cm_svm.png`).

[[PENDIENTE: insertar figura gesture_cm_svm.png]]

**Figura 3.** Curva de aprendizaje del mejor modelo de gestos (`report/figures/gesture_learning_curve.png`).

[[PENDIENTE: insertar figura gesture_learning_curve.png]]

**Figura 4.** Comparación de modelos (`report/figures/gesture_model_comparison.png`).

[[PENDIENTE: insertar figura gesture_model_comparison.png]]

F1 por clase de los modelos comparados: [[PENDIENTE: tabla de precisión, exhaustividad y F1 por clase para RF y SVM]].

### b. Decisión contextual (Árbol de Decisión frente a Random Forest)

Evaluación con `GroupKFold` por escenario.

| Modelo | Hiperparámetros óptimos | Accuracy (media ± d.e.) | F1 macro (media ± d.e.) |
|---|---|---|---|
| Árbol de Decisión | [[PENDIENTE: max_depth, min_samples_leaf, criterion óptimos]] | [[PENDIENTE: accuracy DT]] | [[PENDIENTE: F1 macro DT]] |
| Random Forest | [[PENDIENTE: n_estimators, max_depth, min_samples_leaf óptimos]] | [[PENDIENTE: accuracy RF]] | [[PENDIENTE: F1 macro RF]] |

**Figura 5.** Matriz de confusión del Árbol de Decisión (`report/figures/context_cm_dt.png`).

[[PENDIENTE: insertar figura context_cm_dt.png]]

**Figura 6.** Matriz de confusión del Random Forest contextual (`report/figures/context_cm_rf.png`).

[[PENDIENTE: insertar figura context_cm_rf.png]]

**Figura 7.** Estructura del árbol de decisión entrenado (`report/figures/context_tree.png`).

[[PENDIENTE: insertar figura context_tree.png y comentar las reglas principales que se observan]]

**Figura 8.** Importancia de atributos del modelo contextual (`report/figures/context_feature_importance.png`).

[[PENDIENTE: insertar figura context_feature_importance.png y comentar los atributos más influyentes]]

### c. Capa de seguridad e integración de ambos modelos

Se verificó el cumplimiento de la regla de seguridad (Sección 3.f) en un conjunto de casos de prueba con lecturas de sonar por debajo y por encima del umbral de 0.30 m, y ante una acción `ADVANCE` propuesta por el modelo. Resultado: [[PENDIENTE: número de casos de seguridad probados y cuántos cumplieron la regla; incluir casos en el umbral exacto]]. Número de casos en los que la capa de seguridad modificó la decisión del modelo en el conjunto de escenarios: [[PENDIENTE: conteo]].

### d. Comportamiento observado del robot

Se definió un protocolo de pruebas en el que cada escenario se repite 10 veces. Un intento se considera exitoso si la acción ejecutada por el robot coincide con la acción esperada para ese escenario. La latencia se mide desde que el gesto es sostenido frente a la cámara hasta que el robot inicia la acción.

| Escenario | Gesto | Condición del entorno | Acción esperada | Repeticiones | Éxitos | Latencia media (s) |
|---|---|---|---|---|---|---|
| E1 | `ADVANCE` | Camino libre | `ADVANCE` | 10 | [[PENDIENTE: éxitos]] | [[PENDIENTE: latencia]] |
| E2 | `ADVANCE` | Obstáculo cercano (< 0.30 m) | `WARN_OBSTACLE` | 10 | [[PENDIENTE: éxitos]] | [[PENDIENTE: latencia]] |
| E3 | `STOP` | Robot en movimiento | `STOP` | 10 | [[PENDIENTE: éxitos]] | [[PENDIENTE: latencia]] |
| E4 | `TURN_LEFT` | Camino libre | `TURN_LEFT` | 10 | [[PENDIENTE: éxitos]] | [[PENDIENTE: latencia]] |
| E5 | `TURN_RIGHT` | Camino libre | `TURN_RIGHT` | 10 | [[PENDIENTE: éxitos]] | [[PENDIENTE: latencia]] |
| E6 | `ADVANCE` | Persona lejos de la cámara | `ASK_CLOSER` | 10 | [[PENDIENTE: éxitos]] | [[PENDIENTE: latencia]] |
| E7 | `NONE` | Sin gesto | `WAIT` | 10 | [[PENDIENTE: éxitos]] | [[PENDIENTE: latencia]] |

Descripción cualitativa del comportamiento observado (fluidez, falsos disparos, comportamiento ante cambios de iluminación, estabilidad de la conexión): [[PENDIENTE: observaciones de las pruebas con el robot Zyra]]. Plataforma de las pruebas (robot físico o simulador) y condiciones: [[PENDIENTE: indicar claramente qué escenarios se probaron en robot físico y cuáles en simulador]].

---

## 5. Discusión

### a. Interpretación

La interpretación de resultados debe basarse estrictamente en los valores de la Sección 4. Se establecen las siguientes pautas, que se completarán según lo observado:

- **Gestos, Random Forest frente a SVM.** Si el SVM supera al Random Forest en F1 macro, es coherente con que los atributos normalizados (coordenadas y ángulos) forman fronteras suaves en las que el núcleo RBF es efectivo; si el Random Forest es igual o superior, sugiere que los umbrales sobre ángulos y coordenadas bastan para separar las clases con mayor robustez ante el ruido de pose. La diferencia debe contrastarse con la desviación estándar entre personas: si es menor que ella, no se puede afirmar superioridad. Resultado: [[PENDIENTE: interpretación de la comparación RF/SVM con las cifras de la Tabla 4.a]].
- **Errores entre clases.** Las confusiones entre `TURN_LEFT` y `TURN_RIGHT`, o entre `STOP` y `NONE`, indicarían simetría en los gestos o posturas neutras parecidas a un gesto. Confusiones observadas: [[PENDIENTE: pares de clases más confundidos según las matrices de confusión]].
- **Curva de aprendizaje.** Si las curvas de entrenamiento y validación convergen a un valor alto, el modelo está limitado por sesgo y más datos aportarían poco; si persiste una brecha, hay sobreajuste y más datos de más personas ayudarían. Observación: [[PENDIENTE: interpretación de la curva de aprendizaje]].
- **Contexto, Árbol frente a Random Forest.** Si el Random Forest supera al árbol, la diferencia se atribuye a la menor varianza del conjunto; si son similares, el árbol es preferible por interpretabilidad. La estructura del árbol y la importancia de atributos permiten comprobar si el modelo aprendió las dependencias esperadas (el sonar para `WARN_OBSTACLE`, la distancia a la persona para `ASK_CLOSER`). Resultado: [[PENDIENTE: interpretación con las Figuras 7 y 8]].
- **Rol de la capa de seguridad.** Si el modelo por sí solo ya evitaba el avance con obstáculos, la regla actúa como red de seguridad redundante; si hubo casos en que la corrigió, es evidencia de que una garantía determinista no puede sustituirse por un modelo entrenado con pocos escenarios. Observación: [[PENDIENTE: interpretar el conteo de intervenciones de la Sección 4.c]].
- **Robot real.** La brecha entre el desempeño sobre datos grabados y el éxito en el robot (Tabla 4.d) cuantifica el efecto de condiciones no vistas en el entrenamiento, como la iluminación o la posición de la persona. Resultado: [[PENDIENTE: interpretar el éxito y la latencia por escenario]].

### b. Limitaciones

- **SDK en Python 2.7.** Obliga a una arquitectura de dos procesos y a mantener un intérprete obsoleto. Aunque el protocolo de líneas JSON desacopla ambos lados, agrega complejidad de despliegue, un punto adicional de fallo y no permite que MediaPipe se ejecute en el mismo proceso que NAOqi.
- **Latencia de red (WiFi).** La lectura de sonares y el envío de órdenes viajan por la red inalámbrica; los retardos y pérdidas elevan la latencia entre gesto y acción, y pueden afectar la lectura de obstáculos justo cuando el robot se mueve. Latencia observada: [[PENDIENTE: valores medidos y variabilidad]].
- **Iluminación y fondo.** La estimación de pose se degrada con poca luz, contraluz o ropa que oculta las articulaciones, y la visibilidad baja descarta fotogramas. Aunque los atributos son invariantes a posición y escala, no lo son a la oclusión.
- **Sonar.** Tiene un cono de detección limitado, no ve obstáculos laterales ni por debajo o encima de su campo, tiene una distancia mínima y puede dar lecturas falsas por reflejos o superficies inclinadas o absorbentes. La capa de seguridad solo es tan fiable como esa lectura.
- **Conjunto de datos pequeño.** Cuatro personas y un número reducido de escenarios limitan la generalización; con cuatro grupos, la validación por personas tiene alta varianza. Los resultados describen el comportamiento con este grupo de personas y este laboratorio, y no se garantizan para otros.
- **Movimientos discretos.** Las acciones son desplazamientos fijos (0.2 m, 45°), no un control continuo, y no se evalúa la marcha en superficies irregulares.

### c. Propuestas de mejora

1. Ampliar el conjunto de gestos con más personas, edades, estaturas e iluminaciones, y evaluar con validación por persona sobre un grupo mayor.
2. Utilizar las cámaras del NAO (módulo `ALVideoDevice`) para eliminar la dependencia de la cámara externa, a cambio de menor resolución y mayor latencia.
3. Incorporar la secuencia temporal de *landmarks* (ventanas de varios fotogramas como atributos, o modelos secuenciales) para reconocer gestos dinámicos, no solo posturas estáticas.
4. Migrar el puente a una versión de NAOqi compatible con Python 3 si el robot lo permite, o ejecutar el cerebro a bordo, para eliminar el proceso intermedio.
5. Aumentar y diversificar los escenarios de contexto, incluyendo obstáculos laterales, e incorporar datos de otros sensores (IMU, FSR) para detectar tropiezos o inestabilidad.
6. Evaluar la calibración de probabilidades del modelo contextual y utilizarla para pedir confirmación al usuario cuando la decisión sea incierta.

---

## 6. Conclusiones

1. **Sobre el reconocimiento de gestos (OE1).** [[PENDIENTE: el algoritmo (RF o SVM) obtuvo el mejor F1 macro de __ ± __ con GroupKFold por persona, frente a __ ± __ del otro; la diferencia es / no es mayor que la desviación entre personas, por lo que se concluye que ...]]. La clase con peor desempeño fue [[PENDIENTE: clase]] (F1 = [[PENDIENTE: valor]]), lo que indica [[PENDIENTE: causa probable según la matriz de confusión]].
2. **Sobre la decisión contextual (OE2).** [[PENDIENTE: el modelo (DT o RF) alcanzó F1 macro de __ frente a __ del otro, y los atributos más influyentes fueron __ , lo que confirma / contradice la expectativa de que el sonar y la distancia a la persona determinan la acción]]. Con [[PENDIENTE: n]] escenarios reales, el resultado es [[PENDIENTE: valoración sobre si el tamaño de datos fue suficiente según la curva de aprendizaje]].
3. **Sobre la integración y la seguridad (OE3).** [[PENDIENTE: el robot ejecutó la acción esperada en __ de 10 repeticiones en promedio, con latencia media de __ s; en el escenario con obstáculo, la capa de seguridad evitó el avance en __ de 10 intentos]]. Esto demuestra que [[PENDIENTE: juicio sobre si la combinación de modelo aprendido y regla determinista es adecuada para un robot que comparte espacio con personas]].
4. **Sobre la arquitectura.** El puente persistente en Python 2.7 con protocolo JSON permitió usar MediaPipe en Python 3 y el SDK de NAOqi en Python 2.7 simultáneamente; el costo observado fue una latencia adicional de [[PENDIENTE: ms medidos del viaje de ida y vuelta del puente]], aceptable / no aceptable para [[PENDIENTE: el tipo de órdenes del sistema]].

---

## 7. Referencias

Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F., y Grundmann, M. (2020). *BlazePose: On-device real-time body pose tracking*. arXiv. https://arxiv.org/abs/2006.10204

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Breiman, L., Friedman, J. H., Olshen, R. A., y Stone, C. J. (1984). *Classification and regression trees*. Wadsworth.

Cortes, C., y Vapnik, V. (1995). Support-vector networks. *Machine Learning, 20*(3), 273–297. https://doi.org/10.1007/BF00994018

Gouaillier, D., Hugel, V., Blazevic, P., Kilner, C., Monceaux, J., Lafourcade, P., Marnier, B., Serre, J., y Maisonnier, B. (2009). Mechatronic design of NAO humanoid. En *2009 IEEE International Conference on Robotics and Automation*. IEEE.

Hastie, T., Tibshirani, R., y Friedman, J. (2009). *The elements of statistical learning: Data mining, inference, and prediction* (2.ª ed.). Springer.

Lugaresi, C., Tang, J., Nash, H., McClanahan, C., Uboweja, E., Hays, M., Zhang, F., Chang, C.-L., Yong, M. G., Lee, J., Chang, W.-T., Hua, W., Georg, M., y Grundmann, M. (2019). *MediaPipe: A framework for building perception pipelines*. arXiv. https://arxiv.org/abs/1906.08172

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., y Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Russell, S., y Norvig, P. (2021). *Artificial intelligence: A modern approach* (4.ª ed.). Pearson.

SoftBank Robotics. (s. f.). *NAO documentation* [Documentación técnica del fabricante]. http://doc.aldebaran.com
