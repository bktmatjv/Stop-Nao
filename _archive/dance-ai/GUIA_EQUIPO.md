# Guía del equipo — NAO Dance AI

Inteligencia Artificial (1ASI0404) · Sección 9138
**Entrega: viernes 13/11/2026** — informe PDF + video de 3 a 5 minutos.

> Lean primero las secciones 1 y 2 (qué hace el sistema y cómo funciona). Después vayan directo a su parte en la sección 4.

---

## 1. ¿Qué hace el programa?

El sistema recibe un archivo de audio (una canción) y:

1. **Divide** la canción en fragmentos de 8 segundos.
2. **Analiza** cada fragmento: tempo (BPM), energía y características del sonido.
3. **Predice** un estilo de movimiento para cada fragmento: `SOFT` (suave), `MODERATE` (moderado) o `ENERGETIC` (energético).
4. **Selecciona** una coreografía de brazos y cabeza para cada estilo y ajusta la velocidad según el tempo.
5. **Ejecuta** la coreografía en el NAO (o la simula en modo `--mock`, sin robot).

**Lo que el robot hace:** mueve brazos y cabeza en tres coreografías. No camina ni se desplaza.

**Lo que el sistema NO hace:** no sincroniza cada golpe de la música con precisión; no escucha con el micrófono del NAO; no reconoce voz.

---

## 2. ¿Cómo funciona por dentro?

### 2.1 Arquitectura

```
 Archivo de audio (laptop)
      │
      ▼
┌──────────────────── Python 3.11 (el "cerebro") ────────────────────┐
│  Extracción de características (librosa)                            │
│  BPM, RMS, MFCC, onset strength, spectral centroid                  │
│        │                                                            │
│        ▼                                                            │
│  Modelo de ML: Random Forest o SVM ──► estilo por fragmento        │
│        │                                                            │
│        ▼                                                            │
│  Suavizado (evita cambios bruscos de estilo)                        │
│        │                                                            │
│        ▼                                                            │
│  Selección de coreografía y escalado de velocidad por tempo         │
└────────────────────────────────────┬───────────────────────────────┘
                                     │ JSON por stdin/stdout
                                     ▼
┌──────────── Python 2.7 (el "puente", bridge/nao_bridge.py) ─────────┐
│  naoqi: ALMotion, ALTextToSpeech, ALLeds                            │
└────────────────────────────────────┬───────────────────────────────┘
                                     │ Wi-Fi
                                     ▼
                                 Robot NAO
```

**¿Por qué dos programas de Python?** El SDK del NAO (`naoqi`) solo funciona en **Python 2.7**, y el análisis de audio y el modelo necesitan **Python 3**. No pueden convivir en el mismo proceso. El profesor resolvió esto en `naoooo/py27-py3.py` lanzando Python 2.7 desde Python 3 con `subprocess`. Nosotros adaptamos esa idea: el puente se abre **una sola vez** y queda esperando órdenes, en lugar de abrir un proceso nuevo por cada comando. *(En el informe hay que citar ese código como base.)*

### 2.2 Audio: qué es cada medida

Las características se calculan con `librosa`, una biblioteca de **procesamiento de señales**. Calcula números a partir de la onda de sonido. No es una red neuronal.

| Medida | Qué describe, en palabras simples |
|---|---|
| **BPM (tempo)** | Cuántos pulsos por minuto tiene la música. Se estima y se limita al rango 60–180 BPM. |
| **RMS** | Volumen medio del fragmento. |
| **Onset strength** | Qué tan marcados son los inicios de los sonidos (golpes, ataques). |
| **Spectral centroid** | Qué tan "agudo" es el sonido en promedio (brillo). |
| **MFCC** | Una descripción compacta del timbre; cada coeficiente resume una parte del espectro. |

### 2.3 ¿Dónde hay machine learning y dónde no?

| Componente | ¿Qué es? | ¿Lo entrenamos nosotros? |
|---|---|---|
| **Extracción de características** (`librosa`) | Procesamiento de señales | No hay entrenamiento |
| **Modelo: Random Forest vs SVM** | Machine learning **clásico** (no son redes neuronales) | **Sí**, con nuestras canciones y etiquetas humanas |
| **Regla de BPM** | Regla fija (`if`), solo sirve como **línea base** para comparar | No se entrena; **nunca reemplaza** al modelo |
| **Suavizado y coreografía** | Lógica del programa | No es IA |

**Importante para el informe:** no usamos una red neuronal. Con 10 canciones y poco volumen de datos, los modelos clásicos son más adecuados y más fáciles de explicar. Esto va en el Marco Teórico.

### 2.4 ¿Por qué las etiquetas son humanas?

Si las etiquetas se generaran con reglas de BPM o energía, el modelo solo aprendería esa misma regla: sería un modelo circular, sin valor. Por eso **cada integrante escucha los fragmentos y decide** a qué estilo de movimiento corresponde cada uno. Después se combinan las respuestas por votación mayoritaria y se mide el acuerdo entre anotadores con **kappa de Fleiss**.

### 2.5 ¿Por qué la división es por canción?

Dos fragmentos de la misma canción suenan casi igual. Si unos fragmentos de una canción quedan en entrenamiento y otros en prueba, el modelo parecería muy bueno, pero solo estaría reconociendo la canción. Por eso **todos los fragmentos de una canción van al mismo conjunto** (entrenamiento o prueba). Con solo 10 canciones, el resultado tiene limitaciones estadísticas que hay que informar.

### 2.6 Modelos comparados

- **Random Forest:** muchos árboles de decisión, cada uno entrenado con una muestra distinta. Votan y gana la mayoría.
- **SVM (Support Vector Machine):** busca la frontera que separa las clases con el mayor margen. Necesita datos normalizados (el pipeline ya lo hace).
- **Línea base por reglas:** clasifica solo con BPM y energía. Sirve para comparar: si el modelo no supera la regla, hay que decirlo en el informe.

### 2.7 Estructura de carpetas

```
pc2/
├── dance/
│   ├── audio/         Extracción de características, dataset, anotación, unión de etiquetas
│   ├── model/         Entrenamiento (RF, SVM), inferencia, suavizado
│   └── robot/         Cliente del NAO, coreografías, controlador
├── bridge/            Puente Python 2.7 ↔ NAO (nao_bridge.py)
├── data/
│   ├── songs/         Canciones y manifest.csv (procedencia y licencia)
│   ├── labels/        Una etiqueta por integrante (<nombre>.csv)
│   └── dataset/       segments.csv y labels.csv (los genera el programa)
├── models/            Modelo entrenado (dance_model.joblib)
├── report/            Informe, guion del video, figuras y métricas
├── runs/              Registros de ejecución
├── tests/             Pruebas automáticas (pytest)
└── _archive/gesture-pilot/   Proyecto anterior (archivado, no se usa)
```

---

## 3. Preparación (una sola vez, en la PC del proyecto)

Abrir PowerShell en la carpeta `pc2`:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv311\Scripts\Activate.ps1
python -m pytest -q
```

Requisitos: Python 3.11 con el entorno `.venv311` (`pip install -r requirements.txt`) y Python 2.7 en `C:\Python27` con el SDK `pynaoqi` (solo para el robot).

---

## 4. Tareas por integrante

> El reparto es una **propuesta**: pueden intercambiar roles. Lo importante es que los cuatro aparezcan en el video y que cada uno pueda explicar el sistema completo.

### 👤 Matías — Robot e integración

1. **Reservar tiempo con Zyra**, el robot real. Es el recurso más escaso del proyecto.
2. **Primera prueba en simulación**, sin robot:
   ```powershell
   python -m dance.main --song <archivo> --mock
   ```
   Revisar que aparezcan canción, BPM, características, estilo predicho, coreografía y línea de tiempo.
3. **Primera ejecución real**, con alguien al lado del robot y el piso despejado:
   ```powershell
   python -m dance.main --song <archivo> --robot zyra
   ```
   **Este modo está implementado pero todavía no se probó en el robot.** Es el paso que falta: la primera conexión real con Zyra. Si falla, copiar el error completo (incluido el mensaje del puente) y revisarlo antes de volver a intentar. Opciones útiles: `--no-audio` (analiza sin reproducir la canción), `--ip`/`--port` (si el robot cambia de dirección) y `--log-dir` (por defecto `runs/`).
4. Probar las **tres coreografías** (suave, moderada y energética) con poca amplitud y registrar el resultado de cada una.
5. Probar la **parada** (Ctrl+C o `q`) y confirmar que el robot se detiene.
6. Grabar el **video demo** junto con Vivianne (ver sección 5 del guion).

### 👤 Vanessa — Dataset

1. **Elegir 10 canciones con procedencia legal**: de uso libre (royalty-free), licencia Creative Commons o propias. Deben ser variadas en tempo e intensidad.
2. **Completar `data/songs/manifest.csv`** con las columnas `song_id, file, title, artist, source, license, notes`. Cada canción debe tener su fuente y su licencia.
3. **Reservar 2 o 3 canciones extra** que no se usen en el entrenamiento. Sirven para la demo de generalización en el video.
4. Generar el dataset:
   ```powershell
   python -m dance.audio.build_dataset --template
   python -m dance.audio.build_dataset --songs data/songs --manifest data/songs/manifest.csv --out data/dataset/segments.csv
   ```
5. Revisar que cada fragmento tenga todas las características y que no haya canciones duplicadas.

**No descargar canciones protegidas sin autorización** y no generar datos ficticios que se presenten como muestras reales.

### 👤 Facundo — Etiquetado y modelo

1. **Coordinar el etiquetado a ciegas:** los cuatro integrantes anotan todos los fragmentos sin ver las respuestas de los demás.
   ```powershell
   python -m dance.audio.annotate --annotator <nombre>
   ```
2. Unir las etiquetas y calcular el acuerdo entre anotadores (kappa de Fleiss):
   ```powershell
   python -m dance.audio.merge_labels --labels-dir data/labels --out data/dataset/labels.csv
   ```
3. Entrenar el modelo (Random Forest y SVM) y guardar el seleccionado:
   ```powershell
   python -m dance.model.train --segments data/dataset/segments.csv --labels data/dataset/labels.csv --out models/dance_model.joblib
   ```
4. Entregar a Vivianne las **métricas reales** que genere el programa: accuracy, precision, recall, macro-F1, matriz de confusión, comparación RF vs SVM vs línea base, y el acuerdo entre anotadores.

### 👤 Vivianne — Informe y video

1. Partir de `report/informe_borrador.md`. Formato del enunciado: **Times New Roman 12, interlineado 1.5, márgenes de 2.5 cm, entre 8 y 15 páginas**, con al menos 5 referencias en APA 7.
2. Reemplazar cada `[[PENDIENTE: ...]]` con los números reales que entreguen Facundo y Matías. **No inventar ninguna cifra.** Si un dato no existe, dejarlo declarado como pendiente y explicarlo.
3. Revisar que el informe describa el sistema tal como funciona: el modelo es de machine learning clásico, no una red neuronal; las etiquetas son humanas; la división es por canción.
4. Verificar la referencia de Gouaillier et al. (2009), a la que le faltan páginas y DOI, si se usa.
5. **Video** (guion en `report/guion_video.md`, unos 4 minutos). Ver la sección 5.

---

## 5. Video: requisitos

- **Micrófono externo.** El audio malo baja la nota a 75 %.
- **Dos tomas:** el robot y la persona, más la grabación de la pantalla de la laptop con el BPM y el estilo predicho.
- **Debe verse el robot real.** Si el video es solo simulación, el criterio se evalúa al 50 %.
- **Hablan los cuatro integrantes.**
- La música debe escucharse, pero no tapar las voces.
- Batería cargada, zona despejada y alguien al lado del robot.

---

## 6. Problemas conocidos

| Problema | Causa probable | Qué hacer |
|---|---|---|
| Errores de **doble o mitad de tempo** (el BPM sale como el doble o la mitad del real) | El estimador de tempo puede confundir subdivisiones del pulso | Documentarlo como limitación; no corregirlo a mano en los datos |
| El NAO **habla en inglés** | No tiene el paquete de idioma español instalado | Se puede seguir; avisarlo en el video o en el informe |
| El **micrófono del NAO no se usa** | Captura desde el micrófono del robot está fuera del alcance del MVP (requiere un módulo remoto de audio) | El audio se reproduce y se analiza en la laptop; declararlo como limitación |
| El **modo real con el robot no se ha probado** | Falta la primera conexión con Zyra | Probar en modo `--mock` primero y registrar el resultado real en el informe |
| Pocas canciones (10) | Limitación del proyecto | Informar los resultados con cautela; no afirmar generalización fuerte |

---

## 7. Seguridad con el robot

- Siempre alguien **al lado del NAO**, listo para sostenerlo.
- Piso despejado, sin mesas ni bordes cerca.
- Batería cargada antes de grabar.
- Probar siempre primero con `--mock`.
- Para detener: **Ctrl+C** o la tecla **q**.
- Al cerrar el programa **no se suelta la rigidez** de las articulaciones, para que el robot no se caiga. No desactivar las protecciones de seguridad del robot.
