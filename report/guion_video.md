# Guion del video — NAO con gestos de mano y validación del entorno

Duración objetivo: **4:00** (rango 3:00–5:00).
Participantes: Vanessa, Matías, Facundo, Vivianne.

## Tabla de segmentos

| Tiempo | Quién | Texto hablado (completo) | Qué muestra la cámara |
|--------|-------|--------------------------|------------------------|
| 0:00–0:40 | Vivianne | "Un robot que obedece cualquier orden puede chocar. Nosotros queremos un robot que entienda una señal con la mano, pero que antes de moverse revise si es seguro. Así, el robot no solo obedece: razona." | Plano medio de Vivianne frente a la cámara. Después, plano del NAO en el piso. |
| 0:40–1:20 | Matías | "Una webcam capta la mano. MediaPipe, un modelo preentrenado que no entrenamos nosotros, encuentra 21 puntos de la mano. Un modelo de Random Forest o SVM reconoce el gesto. Luego, un modelo de árbol de decisión o Random Forest revisa el contexto, y una capa de reglas fijas de seguridad tiene la última palabra." | Diagrama del flujo en pantalla (webcam → MediaPipe → gesto → contexto → seguridad → NAO). Matías señala cada bloque. |
| 1:20–1:45 | Vanessa | "Estos son los gestos. Pulgar arriba: avanzar 20 centímetros. Palma abierta: detenerse. Señalar a la izquierda o a la derecha: girar hacia donde señalo. Victoria: cambiar entre gestos y voz. Puño: sentarse o pararse." | Vanessa hace cada gesto a la cámara; en pantalla aparece la etiqueta del gesto reconocido. |
| 1:45–2:20 | Facundo | "Ahora, el primer caso. Pulgar arriba con una caja a 20 centímetros." | Escenario: caja a 20 cm frente al NAO. Vanessa hace pulgar arriba. |
| 2:20–2:40 | Facundo | "El robot no avanza. Sus ojos se ponen rojos y explica el motivo: hay un obstáculo al frente." | Plano del NAO con ojos rojos. Subtítulo con la razón en pantalla. |
| 2:40–3:00 | Vanessa | "Ahora, la caja a 40 centímetros. El robot avanza solo la distancia segura, y sus ojos quedan amarillos." | Caja a 40 cm. El NAO avanza una distancia corta. Ojos amarillos. |
| 3:00–3:10 | Matías | "Señalamos hacia un lado con un obstáculo. El robot rechaza ese giro." | Caja al costado. Matías señala. El NAO no gira y explica. |
| 3:10–3:20 | Vivianne | "Si el gesto no es claro, el robot no adivina: pide que lo repitamos." | Gesto ambiguo. El NAO dice "¿puedes repetir el gesto?". |
| 3:20–3:30 | Matías | "Dos pulgares arriba cambian el modo a música. Ahora el pulgar arriba reproduce la canción, la palma abierta la pausa y señalar cambia de canción. El gesto de pulgar, índice y meñique celebra con 'We Are the Champions'." | Matías hace los gestos. En pantalla, modo MÚSICA y ojos morados. El NAO baila mientras suena la música. |
| 3:30–3:38 | Vivianne | "También podemos dar órdenes por voz. Con la victoria cambiamos a voz, y le decimos 'salúdame'." | Vivianne habla frente a la cámara. Se ven los LED de las orejas encendidos y la intención reconocida en el overlay. |
| 3:38–3:50 | Matías, Facundo | Misión: "Ahora una misión corta: el robot debe llegar a la meta, guiado solo con gestos, sin chocar con las cajas." [Narración de la misión] Métricas: "Resultados del modelo de gestos: [[PENDIENTE]]. Resultados del modelo de contexto: [[PENDIENTE]]." | Recorrido corto entre cajas hasta la meta. Overlay con métricas. |
| 3:50–4:00 | Vivianne | "Un robot que razona antes de obedecer es más seguro y más útil. Gracias." | Plano medio final de los 4 integrantes con el NAO. |

> Nota: los tiempos de la misión y de las métricas pueden ajustarse al grabar. Las métricas se llenan solo con resultados observados; mientras tanto, quedan como [[PENDIENTE]].

## Lista de grabación

- [ ] Micrófono externo (no el de la laptop).
- [ ] Dos ángulos: (1) cámara sobre el NAO; (2) captura de pantalla de la laptop con el overlay (gesto reconocido, distancia del sonar, color de ojos).
- [ ] Batería del NAO cargada.
- [ ] Piso despejado, marcado con cinta.
- [ ] Alguien al lado del robot durante todo el video.
- [ ] Iluminación uniforme; fondo simple detrás de la persona que hace los gestos.
- [ ] Probar la demo completa una vez antes de grabar.

## Lista de tomas

1. Plano medio de Vivianne (intro).
2. Plano del NAO en el piso con el recorrido.
3. Plano de la pantalla con el diagrama (Matías).
4. Plano de la mano frente a la webcam, con la etiqueta del gesto (Vanessa).
5. Plano general del NAO con caja a 20 cm, y plano de sus ojos rojos.
6. Plano de la caja a 40 cm, y ojos amarillos.
7. Plano del giro rechazado.
8. Plano del gesto ambiguo y la pregunta del NAO.
9. Plano de Matías con dos pulgares arriba y pantalla en modo MÚSICA (ojos morados), y plano de los pulgares, palma abierta y celebración.
10. Plano de Vivianne con la victoria, LED de las orejas encendidos y overlay de la intención por voz.
11. Plano general de la misión.
12. Pantalla con métricas (cuando estén disponibles).
13. Plano final de los 4 integrantes.
