"""Spanish keyword fallback used when Groq is unavailable. Ordered rules, first match wins."""

from __future__ import annotations

import re
import unicodedata

from brain import vocab


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9 ]+", " ", text)


_MUSIC = r"(musica|cancion|canciones)"
# Most specific first: "deten la musica" must win over STOP, "modo musica" over MUSIC_PLAY.
_RULES = [
    (rf"\b(deten\w*|para\w*|alto|stop|apaga\w*)\b.*\b{_MUSIC}\b", vocab.MUSIC_STOP),
    (r"\bmodo\b.*\bmusica\b", vocab.SET_MODE_MUSIC),
    (r"\bmodo\b.*\bmovimiento\b", vocab.SET_MODE_MOVEMENT),
    (r"\b(gestos|gesto)\b", vocab.SET_INPUT_GESTURES),
    (r"\bpausa\w*\b", vocab.MUSIC_PAUSE),
    (r"\b(siguiente|proxima)\b", vocab.MUSIC_NEXT),
    (r"\banterior\b", vocab.MUSIC_PREV),
    (rf"\b(reproduce|reproducir|play)\b|\bpon\b.*\b{_MUSIC}\b", vocab.MUSIC_PLAY),
    (r"\b(festeja\w*|campeones|celebra\w*)\b", vocab.CELEBRATE),
    (r"\bsalud\w*\b", vocab.WAVE),
    (r"\b(sientate|sentarte|sientese)\b", vocab.SIT),
    (r"\b(parate|levantate|levantarte|ponte de pie)\b", vocab.STAND),
    (r"\b(detente|detener|alto|para|stop)\b", vocab.STOP),
    (r"\bizquierda\b", vocab.TURN_LEFT),
    (r"\bderecha\b", vocab.TURN_RIGHT),
    (r"\b(avanza\w*|adelante)\b", vocab.ADVANCE),
]
_COMPILED = [(re.compile(p), i) for p, i in _RULES]


def keyword_intent(text: str) -> str:
    """Return a member of vocab.VOICE_INTENTS; UNKNOWN when nothing matches."""
    t = _norm(text or "")
    for rx, intent in _COMPILED:
        if rx.search(t):
            return intent
    return vocab.UNKNOWN
