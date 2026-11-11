<!--
BORRADOR T4 (S8). Total de marcadores PENDIENTE (formato doble corchete): 95.
Todos los valores numericos de resultados, figuras, pruebas del robot y parametros finales de entrenamiento estan como marcadores
y deben completarse desde report/metrics/*.json, report/figures/dance_*.png, data/songs/manifest.csv y el protocolo de pruebas con el robot (T7/T8).
Formato final: PDF, Times New Roman 12, interlineado 1.5, margenes 2.5 cm, 8-15 paginas.
-->

# NAO Dance AI: sistema inteligente de adaptación de movimientos robóticos al ritmo musical mediante Machine Learning

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

La inteligencia artificial aplicada a la robótica busca que una máquina perciba su entorno, interprete esa información y actúe en consecuencia. En robótica social y de entretenimiento esta cadena percepción–decisión–acción es especialmente relevante, porque el robot comparte el espacio con personas que esperan que su comportamiento responda a lo que ocurre a su alrededor y no a un guion fijo. La música es un estímulo particularmente rico para este propósito: contiene información sobre el ritmo, la intensidad y el timbre que el oído humano interpreta de forma casi inmediata y que, mediante técnicas de procesamiento digital de señales, puede cuantificarse como un vector de atributos numéricos. El campo de la recuperación de información musical (*music information retrieval*) ha mostrado que esos atributos permiten clasificar géneros, estimar el tempo y detectar eventos rítmicos con modelos de aprendizaje automático de complejidad moderada (Tzanetakis y Cook, 2002; McFee et al., 2015).

El robot NAO, un humanoide de unos 58 cm con 25 grados de libertad (Gouaillier et al., 2009), es una plataforma habitual en docencia e investigación y puede realizar movimientos expresivos con brazos y cabeza. En este trabajo el sistema analiza una canción, decide mediante un modelo de Machine Learning qué estilo de movimiento corresponde a cada fragmento y ordena al NAO ejecutar la coreografía asociada, con la velocidad ajustada al tempo estimado.

### b. Motivación

Los robots que bailan suelen ejecutar coreografías preprogramadas: una secuencia fija de posturas y tiempos que se reproduce igual sin importar qué música suene. Ese enfoque tiene dos problemas. Primero, el robot no analiza la música: si se cambia la canción por otra de carácter distinto, el movimiento sigue siendo el mismo y la presentación pierde sentido. Segundo, cada nueva canción exige que una persona diseñe manualmente una coreografía nueva, lo que no escala.

La pregunta que motiva este proyecto es si un robot puede adaptar su comportamiento a la música a partir de lo aprendido en ejemplos. Se propone sustituir la coreografía rígida por un sistema en el que un modelo entrenado clasifica cada fragmento en un estilo de movimiento (suave, moderado o energético) y se selecciona una coreografía parametrizada por el tempo. El problema requiere un conjunto de datos propio y combina procesamiento de audio, aprendizaje supervisado e integración con una plataforma cuyo SDK impone restricciones de versión de Python.

### c. Objetivos

**Objetivo general.** Diseñar e implementar un sistema que analice archivos de audio, clasifique cada fragmento en un estilo de movimiento mediante un modelo de Machine Learning y haga que el robot NAO ejecute una coreografía de brazos y cabeza adecuada a ese estilo y ajustada al tempo musical.

**Objetivos específicos.**

1. Construir un conjunto de datos propio a partir de diez canciones aportadas por el equipo, extrayendo características acústicas por fragmento y etiquetando cada fragmento con criterios documentados mediante anotación humana independiente.
2. Entrenar y comparar dos clasificadores (Random Forest y SVM) frente a un baseline de reglas, con una evaluación agrupada por canción que evite la fuga de información entre fragmentos de una misma pieza, y seleccionar y persistir el modelo con mejor desempeño.
3. Integrar el modelo seleccionado con el NAO mediante tres coreografías (suave, moderada y energética) con velocidad dependiente del tempo, transiciones suavizadas y un mecanismo de parada accesible, verificando el comportamiento con canciones no vistas durante el entrenamiento.

Además de estos tres objetivos, el proyecto contempla como alcance complementario la implementación de un modo de simulación que permite probar toda la tubería sin robot físico, el registro de errores y estados de ejecución, y la documentación de la procedencia de cada audio utilizado.

### d. Alcance y limitaciones

El alcance de la práctica se acota deliberadamente para que sea verificable en el tiempo disponible:

- **Diez canciones.** El conjunto de datos proviene de diez archivos de audio proporcionados por el equipo, con procedencia y licencia registradas. Esto implica pocas muestras independientes (la unidad real de independencia es la canción, no el fragmento), lo que limita la fuerza estadística de cualquier conclusión.
- **Audio analizado en la laptop.** El análisis acústico se realiza en el computador a partir del archivo de audio, que también se reproduce allí. No se usa el micrófono del NAO, ya que su captura requeriría un módulo remoto adicional sobre `ALAudioDevice`; se documenta como trabajo futuro.
- **Estilo por segmento, no sincronía por pulso.** El modelo predice un estilo por fragmento de 8 segundos y la coreografía se escala al tempo estimado, pero el sistema no garantiza que cada movimiento coincida exactamente con cada pulso de la música.
- **Solo brazos y cabeza.** El robot no camina, no se desplaza y no realiza movimientos que comprometan el equilibrio. Las protecciones de seguridad del robot permanecen activas.
- **Etiquetas subjetivas.** El estilo de movimiento adecuado para un fragmento es una decisión humana; las etiquetas provienen del consenso de cuatro anotadores y su acuerdo se reporta explícitamente.

---

## 2. Marco Teórico

### Procesamiento digital de audio

Una señal de audio digital es una secuencia de muestras $x[n]$ obtenida al muestrear la presión sonora a una frecuencia $f_s$. En este trabajo todo el audio se convierte a una señal monofónica a $f_s = 22\,050$ Hz, que conserva el contenido hasta unos 11 kHz (teorema de Nyquist), suficiente para describir el ritmo y el timbre musical. Las características se calculan con la biblioteca librosa (McFee et al., 2015), que implementa las siguientes representaciones.

**Transformada de Fourier de tiempo corto (STFT).** Para analizar cómo cambia el espectro en el tiempo, la señal se divide en tramas solapadas de $N$ muestras, cada una multiplicada por una ventana $w[n]$ y transformada al dominio de la frecuencia:

$$X[m,k] = \sum_{n=0}^{N-1} x[n + mH]\, w[n]\, e^{-j 2\pi k n / N}$$

donde $m$ es el índice de trama, $k$ el índice de frecuencia y $H$ el salto entre tramas. El espectrograma de magnitud $|X[m,k]|$ es la base de casi todas las características posteriores.

**Envolvente de intensidad de ataques (*onset strength*).** Los ataques (golpes de percusión, inicios de nota) producen incrementos bruscos de energía espectral. La envolvente de intensidad de ataques resume esos incrementos como el flujo espectral positivo entre tramas consecutivas, aproximadamente:

$$o[m] = \sum_{k} \max\left(0,\; \log|X[m,k]| - \log|X[m-1,k]|\right)$$

Su media indica cuán activa es la música a nivel rítmico y su desviación, cuán variable es esa actividad. La tasa de ataques por segundo es una medida complementaria.

**Estimación de tempo y seguimiento de pulso (*beat tracking*).** El tempo se estima a partir de la periodicidad de la envolvente de ataques. El método de Ellis (2007), implementado en librosa, estima un periodo dominante y luego busca por programación dinámica la secuencia de instantes de pulso que maximiza la suma de la envolvente en esos instantes, penalizando que los intervalos se alejen del periodo estimado. El resultado es una estimación de pulsos por minuto (BPM). Los seguidores de pulso presentan con frecuencia errores de octava (estiman la mitad o el doble del tempo percibido), por lo que más adelante se pliega el valor al intervalo [60, 180] BPM.

**Energía RMS.** La raíz del valor cuadrático medio de una trama, $\mathrm{RMS} = \sqrt{\tfrac{1}{N}\sum_n x[n]^2}$, es una medida de la sonoridad. Su media indica el volumen general del fragmento y su desviación, la dinámica.

**Descriptores espectrales.** El *centroide espectral* es el centro de gravedad del espectro, $C = \sum_k f_k |X[k]| / \sum_k |X[k]|$, y se asocia con el brillo del sonido. El *rolloff* es la frecuencia por debajo de la cual se concentra un porcentaje fijo (aquí 85 %) de la energía. El *ancho de banda espectral* mide la dispersión del espectro alrededor del centroide, y la *planitud espectral* (cociente entre las medias geométrica y aritmética del espectro) distingue sonidos tonales de sonidos parecidos al ruido.

**Tasa de cruces por cero (ZCR).** Es la proporción de muestras consecutivas con signo distinto. Valores altos se asocian con sonidos ruidosos o percusivos.

**Coeficientes cepstrales en la escala mel (MFCC).** Los MFCC (Davis y Mermelstein, 1980) describen el timbre. El espectro de potencia se filtra con un banco de filtros triangulares espaciados en la escala mel, que aproxima la percepción auditiva, $m = 2595 \log_{10}(1 + f/700)$; se toma el logaritmo de las energías de cada filtro y se aplica una transformada discreta del coseno (DCT) para decorrelacionarlas:

$$c_i = \sum_{j=1}^{J} \log(E_j)\, \cos\!\left[\frac{\pi i}{J}\left(j - \tfrac{1}{2}\right)\right]$$

donde $E_j$ es la energía del filtro $j$. Los primeros coeficientes capturan la forma gruesa del espectro y son los más usados en clasificación de música (Tzanetakis y Cook, 2002). Se utilizan los 13 primeros.

**Separación armónico-percusiva (HPSS).** Aplicando filtros de mediana sobre el espectrograma, horizontalmente (se realzan componentes sostenidos) y verticalmente (se realzan transitorios), se separa la señal en una parte armónica y otra percusiva (Fitzgerald, 2010). La proporción de energía percusiva sobre la energía total es un descriptor útil de cuán rítmica y marcada es una pieza.

### Random Forest

Un Random Forest (Breiman, 2001) es un conjunto de $B$ árboles de decisión entrenados sobre muestras *bootstrap* del conjunto de entrenamiento. En cada nodo cada árbol considera solo un subconjunto aleatorio de $m$ atributos candidatos, lo que decorrelaciona los árboles y reduce la varianza del conjunto. La predicción se obtiene por voto mayoritario:

$$\hat{y}(\mathbf{x}) = \arg\max_{c} \sum_{b=1}^{B} \mathbb{1}\left[ T_b(\mathbf{x}) = c \right]$$

y la fracción de votos por clase puede usarse como estimación de confianza. Cada partición se elige para minimizar la impureza de Gini, $G(t) = 1 - \sum_{c} p_{c,t}^{2}$, con $p_{c,t}$ la proporción de la clase $c$ en el nodo $t$. Los hiperparámetros principales son el número de árboles (`n_estimators`), la profundidad máxima (`max_depth`), el mínimo de muestras por hoja (`min_samples_leaf`) y el número de atributos por partición (`max_features`). El modelo entrega además una importancia de atributos basada en la reducción acumulada de impureza, que debe leerse con cautela cuando hay atributos correlacionados, como ocurre entre varios descriptores espectrales.

### Máquinas de Vector Soporte

Una Máquina de Vector Soporte (SVM; Cortes y Vapnik, 1995) busca el hiperplano que separa las clases con el mayor margen. La función de decisión en el caso binario es

$$f(\mathbf{x}) = \operatorname{sign}\left( \sum_{i \in SV} \alpha_i y_i K(\mathbf{x}_i, \mathbf{x}) + b \right)$$

donde $SV$ es el conjunto de vectores soporte, $\alpha_i$ los multiplicadores de Lagrange, $y_i \in \{-1,+1\}$ las etiquetas y $K$ un núcleo. Para fronteras no lineales se emplea el núcleo de base radial (RBF), $K(\mathbf{x},\mathbf{x}') = \exp(-\gamma\lVert\mathbf{x}-\mathbf{x}'\rVert^2)$. Los hiperparámetros críticos son $C$, que equilibra la maximización del margen con la penalización de errores (un $C$ grande tolera menos errores y puede sobreajustar), y $\gamma$, que fija el alcance de influencia de cada vector soporte (un $\gamma$ grande produce fronteras más complejas). El problema multiclase se resuelve con clasificadores binarios uno contra uno (Pedregosa et al., 2011).

Como el núcleo RBF depende de distancias, los atributos deben estandarizarse; por ello el SVM se encapsula en un *pipeline* con `StandardScaler`. Además, el SVM no entrega probabilidades de forma nativa. Para obtener una confianza por clase se usa la calibración de Platt (1999), que ajusta una función sigmoide a las salidas de decisión mediante validación cruzada interna. Estas probabilidades son estimaciones y deben interpretarse con prudencia dado el tamaño reducido del conjunto de datos.

### Baseline de reglas

Un baseline es un método simple contra el cual se compara el modelo entrenado, para verificar que el aprendizaje aporta algo. Aquí es una regla determinista con umbrales sobre el BPM y la energía RMS. Se emplea solo como referencia: no sustituye al modelo en la inferencia y las etiquetas no se generan con ella, lo que evita la circularidad de que el modelo solo aprenda a imitar la regla.

### Criterio de selección frente a alternativas

La elección de modelos responde a las características del problema y no a la sofisticación del algoritmo:

- **Conjunto de datos pequeño y propio.** Con diez canciones, una red convolucional entrenada sobre espectrogramas requeriría muchos más ejemplos para generalizar y sería propensa a memorizar canciones concretas. Calcular descriptores acústicos conocidos ya incorpora conocimiento del dominio y reduce la dimensión del problema.
- **Atributos tabulares de baja dimensión.** El vector por fragmento es tabular y de decenas de atributos, escenario en el que los árboles y los SVM con núcleo suelen ser competitivos con bajo costo (Hastie et al., 2009).
- **Tiempo real en CPU.** La inferencia es muy rápida frente a la duración del fragmento; el costo dominante es la extracción de características.
- **Interpretabilidad.** El Random Forest ofrece importancia de atributos y el baseline es transparente, lo que ayuda a explicar por qué el robot elige un estilo.
- **Frente a KNN.** No requiere entrenamiento, pero su inferencia depende del tamaño del conjunto almacenado y es sensible a la escala; se acotó la comparación a dos algoritmos y un baseline.

Random Forest y SVM representan dos familias distintas (conjuntos de árboles y métodos de margen con núcleo) y pueden comportarse de forma diferente con pocas muestras y atributos correlacionados.

### Robot NAO

NAO es un robot humanoide de unos 58 cm de altura desarrollado por Aldebaran Robotics y luego SoftBank Robotics (Gouaillier et al., 2009). Dispone de 25 grados de libertad (DoF) repartidos en cabeza, brazos, manos, piernas y cadera. Su arquitectura sensorial incluye cuatro micrófonos en la cabeza, dos cámaras frontales, sonares, una unidad inercial (giroscopio y acelerómetro), sensores de fuerza en los pies y sensores táctiles; además cuenta con altavoces y LED. En este proyecto solo se utiliza su sistema motor (brazos y cabeza) y, opcionalmente, la síntesis de voz y los LED de los ojos; sus micrófonos no se usan.

El sistema operativo es NAOqi OS, una distribución Linux sobre la cual corre el middleware NAOqi, que expone la funcionalidad del robot como módulos accesibles por red mediante *proxies* (SoftBank Robotics, s. f.). El módulo relevante aquí es `ALMotion`, y en particular `angleInterpolation`, que recibe para cada articulación una lista de ángulos objetivo y los instantes en que deben alcanzarse, y los interpola de forma suave. Cada articulación tiene límites mecánicos que el sistema consulta con `getLimits` antes de mover. El robot incorpora además un gestor de caídas y protecciones de postura y rigidez que este proyecto no desactiva.

El SDK de NAOqi se programa en Python 2.7 (módulo `naoqi`, clase `ALProxy`) y en C++, y existe Choregraphe, un entorno gráfico de bloques. Python 2.7 es incompatible con librosa y scikit-learn (que requieren Python 3), lo que determina la arquitectura de la Sección 3.f.

---

## 3. Metodología

### a. Descripción del problema

El problema es de clasificación supervisada multiclase. Dado un fragmento de audio de 8 segundos, el sistema debe predecir cuál de tres estilos de movimiento le corresponde: **suave** (música tranquila, de baja intensidad rítmica), **moderado** (ritmo marcado de intensidad media) o **energético** (alta intensidad y ritmo vigoroso), junto con el BPM estimado y la confianza. El flujo completo es: archivo de audio → extracción de características → modelo → predicción del estilo → suavizado → selección de coreografía → control del NAO. Se evalúa la calidad del clasificador sobre canciones no vistas y la funcionalidad extremo a extremo con el robot.

### b. Origen y características del dataset

El conjunto de datos es propio. Está formado por diez canciones proporcionadas por el equipo. La procedencia, el autor, la fuente y la licencia o autorización de uso de cada archivo se registran en el archivo `data/songs/manifest.csv` [[PENDIENTE: resumen de procedencia y licencias de las 10 canciones, desde manifest.csv]]. No se descargaron canciones protegidas sin autorización ni se generaron datos ficticios presentados como muestras reales.

Cada canción se divide en fragmentos de 8 segundos; el último fragmento parcial se conserva solo si dura al menos la mitad de la longitud del segmento. El tamaño de 8 s está dentro del rango de 5 a 10 s propuesto para el trabajo y es configurable. El resultado es un total de [[PENDIENTE: número total de fragmentos]] fragmentos, con una duración total de [[PENDIENTE: minutos de audio]] minutos [[PENDIENTE: tabla de las 10 canciones con título, duración, número de fragmentos y BPM mediano]].

**Anotación.** Las etiquetas no se calculan con reglas sobre el BPM o la energía. Cada uno de los cuatro integrantes escucha cada fragmento de forma independiente, sin ver las etiquetas de los demás, y asigna el estilo de movimiento que considera adecuado (suave, moderado o energético) siguiendo criterios documentados: el nivel de energía percibido, la intensidad del ritmo y el carácter general de la música [[PENDIENTE: transcribir los criterios de anotación acordados por el equipo]]. La etiqueta final de cada fragmento es la mayoría de los cuatro votos; los empates se resuelven por [[PENDIENTE: regla de desempate usada]]. El acuerdo entre anotadores se mide con el coeficiente kappa de Fleiss (Fleiss, 1971), que corrige el acuerdo esperado por azar y permite comparar el consenso con un valor de referencia (Landis y Koch, 1977). El resultado se reporta en la Sección 4.a.

La distribución resultante de clases se presenta en la Figura 4 [[PENDIENTE: figura dance_class_distribution.png con el número de fragmentos por estilo]].

### c. Preprocesamiento

Cada fragmento se resume en un vector de 40 atributos, calculado con la misma función de extracción tanto en el entrenamiento como en la inferencia para evitar discrepancias entre ambas fases:

- **Ritmo:** `bpm`, fuerza del pulso (`beat_strength`), media y desviación de la envolvente de ataques (`onset_mean`, `onset_std`) y tasa de ataques (`onset_rate`).
- **Energía:** media y desviación de la energía RMS (`rms_mean`, `rms_std`).
- **Espectro:** media y desviación del centroide espectral, rolloff, ancho de banda, planitud espectral y tasa de cruces por cero.
- **Contenido percusivo:** razón de energía percusiva obtenida con HPSS (`percussive_ratio`).
- **Timbre:** media y desviación de los 13 primeros MFCC (26 atributos).

**Valores faltantes y silencio.** Los fragmentos silenciosos o muy cortos pueden producir valores indefinidos (por ejemplo, planitud espectral con energía nula o ausencia de pulsos). Todo atributo no finito (NaN o infinito) se reemplaza por 0 y, si no se detecta pulso, el BPM toma el valor mínimo del intervalo, de modo que ningún fragmento interrumpa el entrenamiento ni la inferencia.

**Pliegue del BPM.** Para reducir los errores de octava, el BPM se duplica o divide a la mitad hasta quedar en el intervalo [60, 180]. Es una convención de estabilidad y no garantiza coincidir con el tempo percibido.

**Normalización.** Los atributos tienen escalas muy distintas (el centroide en Hz y los MFCC adimensionales). Para el SVM se aplica `StandardScaler` dentro del *pipeline*, de modo que la media y la desviación se estimen solo con los datos de entrenamiento de cada partición. El Random Forest no la requiere, pues sus particiones son invariantes a transformaciones monótonas de cada atributo. La variable objetivo se codifica con tres clases (suave, moderado, energético).

### d. División de datos

Una división aleatoria de fragmentos sería incorrecta aquí. Los fragmentos contiguos de una canción comparten timbre, instrumentación y tempo; si uno está en entrenamiento y su vecino en prueba, el modelo puede acertar identificando la canción y no el estilo, y las métricas resultarían optimistas por fuga de información (*data leakage*).

Por eso la separación se hace **por canción**: ninguna canción aporta fragmentos al entrenamiento y a la prueba a la vez. Con solo diez canciones, un único reparto daría una estimación inestable, por lo que se usa validación cruzada agrupada con **una canción fuera** (*leave-one-song-out*, LOSO): en cada una de las 10 iteraciones una canción completa se reserva como prueba y las otras nueve se usan para entrenar. Las métricas se agregan sobre las 10 iteraciones y se reporta su desviación estándar entre canciones [[PENDIENTE: confirmar el tratamiento de clases ausentes en la canción de prueba en el código de evaluación]].

### e. Entrenamiento y ajuste de hiperparámetros

Los hiperparámetros se ajustan con `GridSearchCV` optimizando el macro-F1, que promedia el F1 de las tres clases con el mismo peso y es apropiado cuando las clases pueden estar desbalanceadas. Para que el ajuste no contamine la estimación del desempeño, se emplea **validación cruzada anidada y agrupada** (Cawley y Talbot, 2010; Varma y Simon, 2006): el bucle externo es LOSO, y dentro de cada partición de entrenamiento (nueve canciones) el bucle interno aplica una validación cruzada agrupada por canción para elegir los hiperparámetros. La canción de prueba externa nunca interviene en esa elección.

Las grillas exploradas son [[PENDIENTE: confirmar las grillas finales con dance/model/train.py]]:

- **Random Forest:** `n_estimators` ∈ {100, 300}; `max_depth` ∈ {None, 5, 10}; `min_samples_leaf` ∈ {1, 2, 4}; con `class_weight="balanced"` y semilla fija.
- **SVM (RBF):** `C` ∈ {0.1, 1, 10, 100}; `gamma` ∈ {`scale`, 0.001, 0.01, 0.1}; con `probability=True` para calibración de Platt y `StandardScaler` previo.

El **baseline de reglas** usa umbrales sobre el BPM y la energía RMS [[PENDIENTE: valores de los umbrales del baseline y cómo se fijaron]]; no tiene entrenamiento.

El modelo seleccionado es el de mayor macro-F1 promedio en la validación externa. Una vez elegido, se reentrena con las diez canciones y se guarda en `models/dance_model.joblib`, junto con los nombres de los atributos, las etiquetas y la duración del segmento, de modo que la inferencia use exactamente las mismas características. El modelo finalmente seleccionado fue [[PENDIENTE: RF o SVM, con hiperparámetros finales]].

### f. Estrategia de integración con el NAO

**Arquitectura de dos procesos.** El SDK `naoqi` solo funciona con Python 2.7, mientras que librosa y scikit-learn requieren Python 3.11. Siguiendo el patrón del docente (`py27-py3.py` y `nao_show.py`), en el que un programa en Python 3 invoca con `subprocess` el intérprete de Python 2.7 (`C:\Python27\python.exe`) para ejecutar un script con `ALProxy`, el sistema separa un "cerebro" en Python 3.11 (audio y modelo) de un "puente" en Python 2.7 (acceso al robot). El ejemplo del docente lanza un proceso nuevo por cada ejecución, lo que obliga a reconectar cada vez; aquí el puente es un proceso persistente que mantiene los *proxies* abiertos y se comunica con el cerebro mediante líneas JSON por entrada y salida estándar. El patrón se adapta con atribución al código del docente, y la verificación de límites articulares con `getLimits` antes de mover sigue la lógica de `nao_show.py`.

```
 Archivo de audio (laptop)
      |
      v
+------------------------ Python 3.11 ("cerebro") -------------------------+
| load_audio --> segmentos de 8 s --> extract_features (40 atributos)        |
|        --> DancePredictor (RF o SVM)  --> estilo, confianza, BPM           |
|        --> suavizado de predicciones (evita cambios bruscos)               |
|        --> build_keyframes(estilo, bpm) --> DanceController                |
|        --> NaoClient (reproduce el audio en la laptop en paralelo)         |
+-------------------------------+--------------------------------------------+
                                |  JSON por línea (stdin/stdout)
                                v
+------------------- Python 2.7 ("puente", persistente) -------------------+
| nao_bridge.py: ALProxy(ALMotion, ALRobotPosture, ALTextToSpeech, ALLeds)   |
+-------------------------------+--------------------------------------------+
                                |  NAOqi (red)
                                v
                      Robot NAO (físico o simulador)
```

**Protocolo del puente.** Cada mensaje es un objeto JSON en una línea y toda respuesta incluye el campo `ok`. Las órdenes son `ping` (verifica el estado), `get_limits` (devuelve los límites articulares), `play_keyframes` (envía nombres de articulaciones, ángulos y tiempos y los ejecuta sin bloquear mediante `post.angleInterpolation`, devolviendo un identificador de tarea), `is_running` (consulta si la tarea sigue activa), `stop_motion` (detiene las tareas de movimiento activas), `posture`, `say`, `set_eyes` y `quit`. El puente admite `--robot` con el nombre del robot o `--ip` y `--port`, y un modo `--mock` que no requiere el SDK, en el que toda respuesta se marca como simulada. Una orden desconocida produce `ok: false` con un mensaje de error, de modo que el cerebro nunca asume un éxito. Como la ejecución es no bloqueante, el cerebro puede consultar el estado y atender una orden de parada en cualquier momento.

**Tres coreografías de brazos y cabeza.** Se definen tres coreografías, una por estilo, construidas con las articulaciones de los hombros, codos, manos y cabeza:

| Estilo | Característica del movimiento |
|---|---|
| Suave | Oscilaciones lentas y amplias de hombros y codos, con balanceo suave de la cabeza |
| Moderado | Elevación alterna de brazos al compás y asentimientos de cabeza más marcados |
| Energético | Mayor amplitud y frecuencia de los movimientos de brazos con giros de cabeza, siempre dentro de los límites articulares |

Los ángulos exactos de cada pose y su amplitud [[PENDIENTE: tabla de poses clave de cada coreografía desde dance/robot/choreographies.py]] se verificaron contra los límites devueltos por `getLimits`.

**Escalado por tempo con tope de velocidad.** La función `build_keyframes(estilo, bpm, duración)` genera la secuencia de ángulos y tiempos a partir de la coreografía y del BPM estimado. La duración de cada ciclo de movimiento se hace proporcional al periodo del pulso, $60/\mathrm{BPM}$, de modo que una música más rápida produce movimientos más rápidos. Para no exigir velocidades inseguras, la duración de cada movimiento entre poses se limita inferiormente a un mínimo $t_{\min}$ de [[PENDIENTE: valor en segundos]] y el BPM efectivo se acota al intervalo [[PENDIENTE: rango de BPM efectivo]]; si la música es más rápida, el robot se mueve a la velocidad máxima permitida en lugar de acelerar. Así, la velocidad angular de cada articulación queda acotada por construcción. Además, las articulaciones se mueven en torno de una postura base estable (`StandInit`), el robot no camina ni cambia de lugar, y no se desactiva ninguna protección de seguridad del robot.

**Suavizado de predicciones.** Un clasificador por fragmento puede alternar de estilo entre fragmentos consecutivos por ruido, lo que generaría cambios bruscos en el movimiento. La función `smooth_predictions` aplica dos reglas: primero, un cambio de estilo solo se acepta cuando la predicción nueva alcanza una confianza mínima (0.5 por defecto) y, si no la alcanza, se mantiene el estilo anterior; después, un filtro de mayoría sobre tres fragmentos elimina los cambios aislados de un solo fragmento, y en caso de empate el fragmento conserva su propio estilo. Las transiciones entre coreografías no usan una postura intermedia: el controlador encadena cada segmento desde la última postura comandada, y `build_keyframes` garantiza una transición inicial de al menos 0.8 s y un límite de velocidad articular, de modo que el robot nunca salta de una pose a otra. Estos mecanismos introducen una latencia deliberada a cambio de estabilidad, cuyo efecto se discute en la Sección 5.

**Mecanismo de parada.** El controlador (`DanceController`) expone una operación `stop()` que envía `stop_motion` al puente y detiene las tareas de movimiento en curso, y puede activarse en cualquier momento desde la línea de comandos: la tecla `q` (leída sin esperar Enter) o `Ctrl+C`. La parada es segura desde otro hilo, deja el robot en la postura StandInit y el controlador rechaza nuevos segmentos hasta que se reactive. Si un segmento falla en mitad del baile, el sistema llama a la parada antes de propagar el error, para no dejar al robot en movimiento. Todos los estados y errores se registran en un registro de eventos.

**Modo de simulación.** Con la opción `--mock` el sistema funciona sin robot físico: no inicia el puente ni contacta al robot, y muestra el BPM, las características, el estilo predicho, la coreografía elegida y una línea de tiempo. El simulador indica explícitamente que no se ejecutaron movimientos físicos reales; se usa para depurar y para las pruebas automáticas, y se distingue de las pruebas con el robot (Sección 4.c).

---

## 4. Resultados

Los resultados numéricos se obtienen de `report/metrics/*.json` y de las pruebas con el robot. En esta versión todos los valores están pendientes de completarse con las ejecuciones reales y no se presentan estimaciones.

### a. Métricas

**Acuerdo entre anotadores.** El coeficiente kappa de Fleiss entre los cuatro anotadores fue $\kappa$ = [[PENDIENTE: valor de kappa de Fleiss]], que corresponde a un acuerdo [[PENDIENTE: interpretación según Landis y Koch, 1977]]. El porcentaje de fragmentos con unanimidad fue [[PENDIENTE: porcentaje de fragmentos con 4/4 votos]].

**Comparación de modelos.** La Tabla 1 resume el desempeño en la validación cruzada anidada con una canción fuera. Los valores se expresan como media ± desviación estándar entre las canciones de prueba.

| Modelo | Accuracy | Precision (macro) | Recall (macro) | Macro-F1 |
|---|---|---|---|---|
| Random Forest | [[PENDIENTE: media ± std]] | [[PENDIENTE: media ± std]] | [[PENDIENTE: media ± std]] | [[PENDIENTE: media ± std]] |
| SVM (RBF) | [[PENDIENTE: media ± std]] | [[PENDIENTE: media ± std]] | [[PENDIENTE: media ± std]] | [[PENDIENTE: media ± std]] |
| Baseline de reglas | [[PENDIENTE: media ± std]] | [[PENDIENTE: media ± std]] | [[PENDIENTE: media ± std]] | [[PENDIENTE: media ± std]] |

*Tabla 1. Desempeño por modelo en validación cruzada anidada agrupada por canción (LOSO).*

**Desempeño por clase del modelo seleccionado.**

| Clase | Precision | Recall | F1 | Soporte (fragmentos) |
|---|---|---|---|---|
| Suave | [[PENDIENTE: valor]] | [[PENDIENTE: valor]] | [[PENDIENTE: valor]] | [[PENDIENTE: n]] |
| Moderado | [[PENDIENTE: valor]] | [[PENDIENTE: valor]] | [[PENDIENTE: valor]] | [[PENDIENTE: n]] |
| Energético | [[PENDIENTE: valor]] | [[PENDIENTE: valor]] | [[PENDIENTE: valor]] | [[PENDIENTE: n]] |

*Tabla 2. Métricas por clase del modelo seleccionado ([[PENDIENTE: nombre del modelo]]).*

### b. Gráficas

Las matrices de confusión se calculan con las predicciones agregadas de las 10 iteraciones LOSO.

- **Figura 1.** Matriz de confusión del Random Forest [[PENDIENTE: figura report/figures/dance_cm_rf.png]].
- **Figura 2.** Matriz de confusión del SVM [[PENDIENTE: figura report/figures/dance_cm_svm.png]].
- **Figura 3.** Matriz de confusión del baseline de reglas [[PENDIENTE: figura report/figures/dance_cm_baseline.png]].
- **Figura 4.** Distribución de clases en el conjunto de datos [[PENDIENTE: figura report/figures/dance_class_distribution.png]].
- **Figura 5.** Comparación de modelos (macro-F1 con barras de error) [[PENDIENTE: figura report/figures/dance_model_comparison.png]].
- **Figura 6.** Importancia de atributos del Random Forest [[PENDIENTE: figura report/figures/dance_feature_importance.png]].

### c. Comportamiento observado

Se probó el sistema con canciones **no vistas** durante el entrenamiento, es decir, que no forman parte de las diez canciones del conjunto de datos, ya que evaluar sobre canciones del entrenamiento sobreestimaría el desempeño. Para cada prueba se registró el BPM estimado, el estilo predicho, la coreografía ejecutada y una observación del comportamiento del robot.

| Canción nueva (no vista) | BPM estimado | Estilo predicho | Coreografía ejecutada | Observación | ¿Éxito? |
|---|---|---|---|---|---|
| [[PENDIENTE: canción 1]] | [[PENDIENTE: BPM]] | [[PENDIENTE: estilo]] | [[PENDIENTE: coreografía]] | [[PENDIENTE: observación]] | [[PENDIENTE: sí/no]] |
| [[PENDIENTE: canción 2]] | [[PENDIENTE: BPM]] | [[PENDIENTE: estilo]] | [[PENDIENTE: coreografía]] | [[PENDIENTE: observación]] | [[PENDIENTE: sí/no]] |
| [[PENDIENTE: canción 3]] | [[PENDIENTE: BPM]] | [[PENDIENTE: estilo]] | [[PENDIENTE: coreografía]] | [[PENDIENTE: observación]] | [[PENDIENTE: sí/no]] |

*Tabla 3. Pruebas con el robot sobre canciones nuevas.*

La tasa de éxito global fue [[PENDIENTE: n de m pruebas]]. Se evaluaron además la estabilidad de las transiciones entre estilos [[PENDIENTE: número de cambios de estilo por canción antes y después del suavizado]], la respuesta de la parada [[PENDIENTE: tiempo medido entre la orden y la detención del movimiento]] y los errores o alertas registrados durante la ejecución [[PENDIENTE: resumen del registro de eventos]]. Las pruebas se realizaron con [[PENDIENTE: robot físico o simulador, identificador del robot]].

### d. Comparación entre configuraciones

La comparación principal es entre Random Forest, SVM y el baseline de reglas (Tabla 1 y Figura 5). Como comparación complementaria se contrasta la configuración con atributos completos frente a [[PENDIENTE: configuración alternativa, por ejemplo solo atributos de ritmo y energía, o con y sin suavizado]], con un macro-F1 de [[PENDIENTE: valor]] frente a [[PENDIENTE: valor]]. La diferencia entre modelos [[PENDIENTE: indicar si la diferencia es mayor que la desviación estándar entre canciones]].

---

## 5. Discusión

### a. Interpretación

Con diez canciones, el número de unidades independientes es diez, y los fragmentos de una misma canción son observaciones correlacionadas; la desviación estándar entre canciones es, por tanto, tan informativa como la media.

Si el Random Forest o el SVM superan de forma consistente al baseline, los atributos acústicos aportan información más allá de umbrales simples de BPM y energía; en caso contrario, el modelo aprendido no justificaría su complejidad [[PENDIENTE: interpretación según la diferencia entre los tres modelos en la Tabla 1]]. Si un modelo supera al otro por menos que la desviación entre canciones, no hay evidencia suficiente para declararlo mejor y la elección debería apoyarse en criterios secundarios como la interpretabilidad [[PENDIENTE: modelo preferido y justificación]].

Cabe esperar que la clase moderada, por ser intermedia, sea la más difícil de separar de las otras dos, pues las fronteras entre estilos son graduales y subjetivas [[PENDIENTE: confirmar o refutar con las matrices de confusión]]. Si en la importancia de atributos (Figura 6) dominan el BPM y la energía RMS, el modelo se parecería al baseline; si aparecen atributos tímbricos o percusivos, estaría aprovechando información adicional [[PENDIENTE: atributos más importantes y su interpretación musical]]. El kappa de Fleiss delimita el techo realista del problema: si los propios anotadores discrepan, no es esperable que un modelo concuerde con la etiqueta más de lo que concuerdan las personas [[PENDIENTE: relación entre kappa, desempeño del modelo y éxito en canciones nuevas, Tabla 3]].

### b. Limitaciones

- **Pocas canciones independientes.** Con diez canciones los intervalos de incertidumbre son amplios, los resultados no deben extrapolarse a la música en general y la selección de hiperparámetros es inestable.
- **Etiquetas subjetivas.** Aunque se usa voto mayoritario y se reporta kappa, las etiquetas representan la opinión de cuatro personas y no una verdad objetiva.
- **Errores de octava en el tempo.** El seguidor de pulso puede estimar la mitad o el doble del tempo percibido; el pliegue a [60, 180] BPM reduce el problema, pero no lo elimina, y una estimación errónea altera la velocidad de la coreografía.
- **SDK en Python 2.7.** Obliga a mantener dos intérpretes y un protocolo entre procesos, con riesgos de mantenimiento.
- **Latencia y sincronía.** El audio se reproduce en la laptop y los movimientos se ejecutan por red, con un desfase variable; el estilo se decide por fragmento de 8 s y el suavizado retrasa deliberadamente los cambios.
- **Micrófono del NAO no utilizado.** Escuchar el ambiente con los micrófonos del robot enfrentaría ruido y reverberación que este trabajo no evalúa.
- **Solo brazos y cabeza.** La seguridad se prioriza sobre la espectacularidad, por lo que el repertorio es reducido.

### c. Propuestas de mejora

Con más tiempo y recursos, el grupo propondría: (1) ampliar el conjunto de datos a más canciones, géneros y anotadores; (2) capturar el audio con los micrófonos del NAO mediante un módulo remoto sobre `ALAudioDevice`; (3) sincronizar los movimientos con los pulsos individuales y no solo con el tempo; (4) incorporar el contexto temporal entre fragmentos y, con más datos, evaluar redes sobre espectrogramas; y (5) ampliar el repertorio de coreografías.

---

## 6. Conclusiones

1. Respecto al primer objetivo específico (conjunto de datos propio): la construcción de un conjunto de datos a partir de diez canciones y la anotación independiente de [[PENDIENTE: número de fragmentos]] fragmentos por cuatro personas produjo un acuerdo de $\kappa$ = [[PENDIENTE: kappa]], lo que indica que [[PENDIENTE: juicio sobre la fiabilidad de las etiquetas y su efecto en el techo del modelo]].
2. Respecto al segundo objetivo específico (comparación de modelos): en validación cruzada agrupada por canción, el modelo [[PENDIENTE: RF o SVM]] obtuvo un macro-F1 de [[PENDIENTE: valor]] frente a [[PENDIENTE: valor]] del otro algoritmo y [[PENDIENTE: valor]] del baseline de reglas, lo que [[PENDIENTE: juicio: si el aprendizaje supera o no al baseline y si la diferencia entre modelos es significativa frente a la variabilidad entre canciones]].
3. Respecto al tercer objetivo específico (integración con el NAO): el sistema [[PENDIENTE: juicio sobre si el robot ejecutó correctamente la coreografía adecuada en las canciones nuevas, con la tasa de éxito observada]], con velocidad limitada y parada funcional, lo que [[PENDIENTE: juicio sobre la viabilidad de la arquitectura de dos procesos y su latencia]].
4. Respecto al objetivo general: la cadena completa audio → características → modelo → coreografía [[PENDIENTE: juicio global de si el sistema adapta de forma perceptible el movimiento a la música frente a una coreografía fija]]; sin embargo, dado el tamaño del conjunto, esta conclusión se limita al contexto de las canciones evaluadas.
5. Respecto a la metodología: la separación de datos por canción fue necesaria para no sobreestimar el desempeño; [[PENDIENTE: comparación, si se hizo, con una división aleatoria de fragmentos que muestre la diferencia de métricas]].

---

## 7. Referencias

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Cawley, G. C., y Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107.

Cortes, C., y Vapnik, V. (1995). Support-vector networks. *Machine Learning, 20*(3), 273–297. https://doi.org/10.1007/BF00994018

Davis, S. B., y Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. *IEEE Transactions on Acoustics, Speech, and Signal Processing, 28*(4), 357–366. https://doi.org/10.1109/TASSP.1980.1163420

Ellis, D. P. W. (2007). Beat tracking by dynamic programming. *Journal of New Music Research, 36*(1), 51–60. https://doi.org/10.1080/09298210701653344

Fitzgerald, D. (2010). Harmonic/percussive separation using median filtering. En *Proceedings of the 13th International Conference on Digital Audio Effects (DAFx-10)*. Graz, Austria.

Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters. *Psychological Bulletin, 76*(5), 378–382. https://doi.org/10.1037/h0031619

Gouaillier, D., Hugel, V., Blazevic, P., Kilner, C., Monceaux, J., Lafourcade, P., Marnier, B., Serre, J., y Maisonnier, B. (2009). Mechatronic design of NAO humanoid. En *2009 IEEE International Conference on Robotics and Automation* (pp. 769–774). IEEE.

Hastie, T., Tibshirani, R., y Friedman, J. (2009). *The elements of statistical learning: Data mining, inference, and prediction* (2.ª ed.). Springer.

Landis, J. R., y Koch, G. G. (1977). The measurement of observer agreement for categorical data. *Biometrics, 33*(1), 159–174. https://doi.org/10.2307/2529310

McFee, B., Raffel, C., Liang, D., Ellis, D. P. W., McVicar, M., Battenberg, E., y Nieto, O. (2015). librosa: Audio and music signal analysis in Python. En *Proceedings of the 14th Python in Science Conference* (pp. 18–25).

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., y Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Platt, J. C. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. En A. J. Smola, P. Bartlett, B. Schölkopf, y D. Schuurmans (Eds.), *Advances in large margin classifiers* (pp. 61–74). MIT Press.

SoftBank Robotics. (s. f.). *NAO documentation* [Documentación técnica del fabricante]. http://doc.aldebaran.com

Tzanetakis, G., y Cook, P. (2002). Musical genre classification of audio signals. *IEEE Transactions on Speech and Audio Processing, 10*(5), 293–302. https://doi.org/10.1109/TSA.2002.800560

Varma, S., y Simon, R. (2006). Bias in error estimation when using cross-validation for model selection. *BMC Bioinformatics, 7*, Artículo 91. https://doi.org/10.1186/1471-2105-7-91
