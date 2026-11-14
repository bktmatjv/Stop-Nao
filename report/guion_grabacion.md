# Guion de grabación — qué hacer frente al robot

Lista práctica para el día de la grabación, con lo que el sistema **hace hoy**.
Complementa a [`guion_video.md`](guion_video.md), que tiene el reparto y los textos
largos; este archivo es el orden de acciones y lo que debe verse en pantalla.

Duración objetivo: **4:45** (rango 3:00–5:00). Si hay que recortar, el giro rechazado de la
escena 5 es lo primero que se cae; la escena 3 y la 9 no se tocan.

---

## Antes de empezar: dos cosas que rompen la toma

**La voz necesita `.env`.** Hoy ese archivo no existe, así que no hay `GROQ_API_KEY` y el
micrófono se queda mudo. Antes de grabar, en PowerShell:

```powershell
Copy-Item .env.example .env
```

La clave ya viene escrita en `.env.example`. `sounddevice` está instalado (0.5.6), así que la
grabación funciona.

**Sin `.env`, el micrófono graba pero el robot no obedece nunca**, y es engañoso: el overlay
reacciona, así que parece que te oye. Lo que pasa por dentro es que sin clave no se pide la
transcripción, el texto llega vacío y la intención sale `UNKNOWN`, que se descarta sin hacer
nada. El respaldo por palabras clave tampoco salva la situación, porque necesita texto para
buscar palabras. En los registros de `runs/` de hoy **no hay una sola orden de voz
ejecutada**: solo filas de `VICTORY` y `OPEN_PALM`, que son gestos.

Dos comprobaciones de 30 segundos antes de grabar:

1. Al arrancar, mira la consola. Si aparece `AVISO: sin GROQ_API_KEY la voz no puede
   transcribir`, no has creado el `.env`. Para ahí.
2. Entra en modo voz, di "salúdame" y mira el overlay. Tiene que decir
   `Oido: 'saludame' -> WAVE`. Si dice `Oido: '' -> UNKNOWN`, no hay transcripción (falta la
   clave o la red bloquea `api.groq.com`).

**El modo MÚSICA no tiene audio.** En `songs/` solo hay `manifest.csv`, ningún `.wav`. Puedes
entrar al modo y eso se ve bien (ojos morados, *"Modo música"*), pero `THUMBS_UP` o
"reproduce" contestan *"No hay canciones para reproducir."* No pidas canciones en cámara.

Lo que sí funciona y da para un video completo: **modo MOVIMIENTO con gestos**, más la
**entrada por voz** de la escena 9.

---

## Preparación

- [ ] `.env` creado y voz probada (ver arriba).
- [ ] Piso despejado y alguien junto al NAO durante toda la grabación.
- [ ] Batería del NAO cargada.
- [ ] Una caja o libro a mano para los obstáculos.
- [ ] Micrófono externo, no el de la laptop.
- [ ] Iluminación uniforme sobre la mano; fondo simple.
- [ ] Grabar **pantalla y robot a la vez** (ver nota abajo).
- [ ] Ensayar la demo completa una vez antes de grabar en serio.

**Por qué grabar también la pantalla:** la ventana de vista previa muestra gesto, confianza,
sonares en cm, postura, última acción, motivo, color de ojos y, en voz, la frase transcrita y
la intención elegida. Es la evidencia de que decide el modelo y no la persona. Si solo grabas
al robot, se ve un robot moviéndose pero no se ve el modelo, que es justo lo que califica la
rúbrica.

**Posición:** a 0.5–1.5 m de la webcam.

**Ritmo:** hay 2.5 s de espera entre órdenes **por gestos**. Entre gesto y gesto **baja la
mano**; un gesto sostenido se ejecuta una sola vez. Cuenta "mil uno, mil dos, mil tres" antes
del siguiente. Las órdenes por voz no pasan por esa espera: cada pulsación de `v` es una
orden.

---

## Comando

```powershell
python -m brain.main --robot nyx --ip 192.168.137.45 --port 9559
```

Ajusta `--ip` si cambió. En la ventana de vista previa, `q` cierra el programa y `v` inicia y
envía la grabación de voz.
Las teclas `1/2/3/l/r/s` que simulan obstáculos **solo existen con `--mock`**; con el robot
real los sonares son de verdad, así que los obstáculos se ponen con la caja.

---

## Escena por escena

### 1. Presentación — 20 s, sin gestos

> "Somos [nombres]. Nuestro NAO obedece gestos de mano y órdenes por voz, pero antes de
> obedecer revisa su entorno: si la orden no es segura, se niega y explica por qué."

Señalar en pantalla: gesto detectado, confianza, sonares, **ojos azules** = esperando.

- [ ] Grabada

### 2. Avance libre — 25 s · `THUMBS_UP`, camino despejado

Esperado: avanza 35 cm · **ojos verdes** · *"Avanzo, el camino está libre."*

> "Pulgar arriba significa avanzar. Los sonares leen más de 65 centímetros, así que da el
> paso completo."

- [ ] Grabada

### 3. Rechazo por obstáculo — 35 s · caja a ~20 cm · `THUMBS_UP`

Esperado: **no se mueve** · **ojos rojos** · *"Hay un obstáculo a 20 centímetros. No puedo avanzar."*

> "El mismo gesto, entorno distinto. La capa de seguridad tiene la última palabra: el modelo
> puede proponer avanzar, pero si el sonar ve menos de 30 centímetros, no avanza."

**Es la escena más importante del video.** Repetirla dos veces y grabar varias tomas.

- [ ] Grabada (toma 1)
- [ ] Grabada (toma 2)

### 4. Avance parcial — 30 s · caja a ~40 cm · `THUMBS_UP`

Esperado: avanza solo ~10 cm · **ojos amarillos** · *"Solo puedo avanzar 10 centímetros."*

> "Y no es solo sí o no: calcula cuánto puede avanzar sin llegar al margen de seguridad, y
> avanza esa distancia."

- [ ] Grabada

### 5. Giros — 30 s · `POINT_LEFT`, luego `POINT_RIGHT`

Con el lado libre. Esperado: gira 45° **hacia donde señalas** · **ojos cian**.

> "Señalo a mi izquierda y el robot gira hacia ahí. Como está de frente a mí, su izquierda es
> mi derecha; el código invierte la referencia a propósito."

Después, caja al lado hacia el que vas a señalar y repetir: **se niega, ojos rojos**.

- [ ] Giro permitido
- [ ] Giro rechazado

### 6. Cambio de postura — 25 s · `FIST`

Esperado: se sienta · **ojos blancos** · *"Me siento a descansar."* Otro `FIST` lo levanta.

> "El puño alterna sentarse y levantarse. Si estuviera caminando, primero exigiría detenerse."

- [ ] Grabada

### 7. Parada de emergencia — 25 s · `THUMBS_UP` y, mientras camina, `OPEN_PALM`

Esperado: se detiene **en el acto**, sin esperar los 2.5 s.

> "La palma abierta es paro de emergencia. No pasa por el modelo ni por el umbral de
> confianza: va directa a detener."

- [ ] Grabada

### 8. Confirmación de gesto ambiguo — 30 s · `LOVE_YOU`

- Primera vez: **ojos amarillos** · *"No estoy seguro del gesto. ¿Puedes repetirlo?"*
- Repetirlo **dentro de 4 s**: celebra y baila.

> "Medimos que el 42 % de los `VICTORY` se confunden con este gesto en una persona que el
> modelo nunca vio. Por eso pide confirmación antes de actuar: prefiere preguntar a
> equivocarse."

Vale mucho para la nota de creatividad: muestra que midieron un error y respondieron a él.

- [ ] Grabada

### 9. Entrada por voz — 50 s · `VICTORY` y la tecla `v`

Cómo se habla: pulsa `v`, di la frase, vuelve a pulsar `v`. No hay que mantener la tecla.
El overlay va mostrando `Grabando... (v para enviar)` → `Procesando...` →
`Oido: 'avanza' -> ADVANCE`. Esa línea es la mejor prueba en pantalla de toda la escena:
aparece la frase tal como la entendió y la intención que eligió.

**9a. Cambiar de entrada.** Muestra `VICTORY`.
Esperado: *"Te escucho por voz"* · **LED de las orejas encendidos** · overlay `Entrada: VOICE`.

> "El mismo robot acepta otra entrada. Muestro la V y el micrófono reemplaza a los gestos: se
> le encienden los LED de las orejas. Para hablar pulso 'v', doy la orden y vuelvo a pulsar
> 'v'."

Ojo: si arrancaste con `--confirm-ambiguous`, el primer `VICTORY` contesta
*"No estoy seguro del gesto. ¿Puedes repetirlo?"* y **no cambia la entrada**. Es lo que pasa
en los registros de hoy. Repite la V dentro de 4 s y entonces sí cambia. No es un fallo, es
la escena 8 aplicada a la V; si no quieres explicarlo dos veces, arranca sin esa opción.

**9b. "Salúdame".** Esperado: saluda con el brazo · **ojos magenta** ·
*"¡Hola! Un gusto saludarte."*

> "El saludo no tiene gesto: existe solo por voz. La voz no repite lo que ya hacían las manos,
> añade órdenes propias."

**9c. "Avanza", con la caja a ~20 cm.** Esperado: **no se mueve** · **ojos rojos** ·
*"Hay un obstáculo a 20 centímetros. No puedo avanzar."*

> "Y la seguridad no depende de cómo llegó la orden. Dicha en voz alta, pasa por la misma capa
> de reglas y se niega igual."

**9d. "Avanza" con el camino libre y, mientras camina, `OPEN_PALM`.** Esperado: se detiene en
el acto, aunque la entrada activa sea la voz.

> "Aunque el micrófono esté al mando, la palma abierta sigue viva: el paro de emergencia nunca
> se desactiva."

**9e. Volver a gestos.** Di "gestos", o muestra `VICTORY` otra vez.
Esperado: *"Volvemos a los gestos"* · LED de las orejas apagados.

- [ ] 9a cambio de entrada
- [ ] 9b "salúdame"
- [ ] 9c "avanza" rechazado
- [ ] 9d palma abierta en modo voz
- [ ] 9e vuelta a gestos

#### Frases que el sistema entiende

Di estas, tal cual. La lista de intenciones es cerrada: lo que no encaja sale como `UNKNOWN`
y el robot no hace nada.

| Lo que dices | Qué hace | Qué se ve |
|---|---|---|
| "avanza" / "adelante" | `ADVANCE`, con la misma capa de seguridad | verde, amarillo o rojo según el sonar |
| "detente" | `STOP`, paro inmediato · *"Me detengo."* | verde |
| "gira a la izquierda" / "gira a la derecha" | `TURN_LEFT` / `TURN_RIGHT`, 45° | cian |
| "siéntate" | `SIT` · *"Me siento a descansar."* | blanco |
| "levántate" / "párate" | `STAND` · *"Me pongo de pie y estoy listo."* | blanco |
| "salúdame" | `WAVE` · *"¡Hola! Un gusto saludarte."* | magenta |
| "festeja" / "somos campeones" | `CELEBRATE`, baile de campeones (sin canción, ver avisos) | — |
| "modo música" / "modo movimiento" | cambia de modo | morado / azul |
| "gestos" | vuelve a la entrada por gestos | — |

**Cuatro trampas que hay que saber antes de hablar:**

1. **Por voz, izquierda y derecha son las del robot.** Con gestos, señalar a tu izquierda lo
   gira hacia donde señalas; por voz no hay inversión. Si dices "gira a la izquierda" gira a
   **su** izquierda, que estando de frente es tu derecha. Narra el giro mirando al robot, no a
   ti.
2. **"Siéntate" y "levántate" son órdenes explícitas**, no un interruptor como el puño. No
   digas "siéntate" dos veces esperando que se levante.
3. **Cualquier frase con la palabra "gesto" te saca del modo voz.** No digas "ahora sin
   gestos" estando en modo voz: eso es, justamente, la orden de volver a gestos.
4. **Una orden a la vez.** Si pulsas `v` mientras procesa la anterior, el overlay dice
   `Espera: procesando la voz anterior` y la nueva no se graba.

#### Si te oye pero no obedece

El overlay dice en qué punto se rompió la cadena. Lee **dos líneas**: la de `Voz:` (qué
entendió) y las de `Ultima accion:` y `Motivo:` (qué decidió). Hay tres casos y se distinguen
a simple vista.

**Caso 1 — `Oido: '' -> UNKNOWN`, el texto está vacío.** No hubo transcripción. No es el
micrófono: graba bien, pero nadie convirtió el audio en texto. Falta `.env`, o la red bloquea
`api.groq.com`. Es el caso de hoy. Crea el `.env` y reinicia; si la red del laboratorio lo
bloquea, la escena 9 no es grabable y hay que saltarla.

**Caso 2 — `Oido: 'lo que dijiste' -> UNKNOWN`, con texto pero sin intención.** Te transcribió
y no supo qué querías. La lista de intenciones es cerrada y descartar es el comportamiento
correcto: prefiere no hacer nada antes que adivinar. Usa las frases de la tabla, tal cual, y
di una sola orden por grabación ("avanza", no "a ver, avanza un poco hacia mí").

**Caso 3 — `Oido: 'avanza' -> ADVANCE` pero el robot no se mueve.** Esto no es un problema de
voz: la orden llegó entera y **la capa de seguridad la rechazó**, igual que rechaza un gesto.
Mira `Motivo:` y el color de ojos:

| Motivo en pantalla | Por qué | Qué haces |
|---|---|---|
| *"Primero debo ponerme de pie."* (amarillo) | Está sentado; avanzar, girar y saludar exigen estar de pie | Di "levántate" y repite la orden. **Es el fallo más probable si vienes de la escena 6.** |
| *"Hay un obstáculo a N centímetros."* (rojo) | Hay algo delante | Es correcto: es la escena 9c. Quita la caja. |
| *"No puedo medir la distancia al frente."* (rojo) | El sonar no respondió o el dato es viejo | Falla hacia el lado seguro. Espera un segundo y repite. |
| *"Me estoy moviendo. Primero debo detenerme..."* | Pediste sentarse o saludar mientras caminaba | Di "detente" y repite. |
| *"No pude ejecutar la orden."* (rojo) | El puente rechazó el comando; en la consola sale `WARNING: bridge refused` | Revisa que el NAO siga conectado. |

**Y un caso que no es un fallo:** si repites la misma orden antes de 3 s, el robot **se queda
callado** aunque la haya ejecutado, porque no repite la misma frase tan seguido. Comprueba
`Ultima accion:`, no el altavoz.

Para la revisión posterior, cada orden deja una fila en `runs/<fecha>.csv` con la intención
pedida, la acción ejecutada, el motivo y si el puente la aceptó. Las órdenes de voz son las
filas cuya columna `gesture` dice `VOICE`.

### 10. Cierre — 20 s

> "El reconocimiento de gestos es un SVM entrenado con 10 813 fotogramas propios de tres
> personas, evaluado reservando una persona entera: 94 % de exactitud sobre alguien que el
> modelo nunca vio. La voz transcribe con Whisper y elige la intención con un modelo de
> lenguaje, siempre de una lista cerrada. Pero la decisión final nunca es del modelo, ni del
> de gestos ni del de voz: siempre pasa por la misma capa de reglas."

- [ ] Grabada

---

## Si algo sale mal

| Qué ves | Qué pasa | Qué haces |
|---|---|---|
| Ojos rojos y *"No puedo medir la distancia"* | Sonar devolvió 0 o inválido | Es correcto: falla hacia el lado seguro. Sirve como escena. |
| No reacciona a un gesto | Faltan los 2.5 s, o no bajaste la mano | Baja la mano, espera, repite. |
| Confianza por debajo de 0.60 | Mano mal iluminada o muy lejos | Acércate, mejora la luz. |
| Se queda mudo tras `VICTORY` | Pasó a entrada VOZ y no has pulsado `v` | Pulsa `v`, habla, pulsa `v`. O muestra `VICTORY` para volver a gestos. |
| `Voz: Voz no disponible` | Arrancó con `--no-voice`, o el escucha no se pudo crear | Reinicia sin `--no-voice`. |
| `Voz: Error de microfono: ...` | Micrófono ocupado o sin permiso | Cierra Zoom/Teams, revisa los permisos de Windows. |
| `Oido: '' -> UNKNOWN` | No hubo transcripción: falta `.env` o la red bloquea Groq | Crea `.env`. Si la red lo bloquea, salta la escena 9. |
| `Oido: '<frase>' -> UNKNOWN` | La frase no encaja en ninguna intención | Usa las frases de la tabla, tal cual. |
| Te entiende la frase y la intención, pero no se mueve | La capa de seguridad rechazó la orden | Lee `Motivo:`; casi siempre está sentado. Ver [Si te oye pero no obedece](#si-te-oye-pero-no-obedece). |
| `VICTORY` contesta *"¿Puedes repetirlo?"* y no cambia la entrada | Arrancaste con `--confirm-ambiguous` | Repite la V dentro de 4 s. |
| Ejecuta la orden pero no dice nada | Repetiste la misma frase antes de 3 s | Normal. Mira `Ultima accion:`, no el altavoz. |
| Se sienta y la ventana se congela | El arreglo del puente no funcionó | `git stash` vuelve al código anterior. |

---

## Dos avisos honestos

**El arreglo del puente se estrena hoy.** `goToPosture` pasó a ser no bloqueante para corregir
el cuelgue que mataba las sesiones (`SIT` tardaba 20 s de mediana). Vigila el primer `FIST`: el
robot debe empezar a sentarse de inmediato y la ventana seguir respondiendo.

**No hay modelo de contexto entrenado.** Lo que se ve en el video son las **reglas por gesto
más la capa de seguridad**. No digas en la narración que hay un segundo modelo entrenado; el
conjunto `data/context/scenarios.csv` todavía no se ha recolectado. Lo mismo con "festeja":
sin audio en `songs/`, el robot baila pero no suena la canción; no prometas música.

---

## Colores de ojos, para narrar sin equivocarse

| Color | Significado |
|---|---|
| Azul | Esperando, modo MOVIMIENTO |
| Verde | Orden aceptada |
| Amarillo | Limitada (avance corto) o pide confirmación |
| Rojo | Rechazada por seguridad |
| Cian | Giro |
| Blanco | Sentarse o levantarse |
| Magenta | Saludo (solo por voz) |
| Morado | Modo MÚSICA |

**LED de las orejas:** encendidos cuando la entrada es VOZ. Es la señal visual de que el
micrófono está al mando.
