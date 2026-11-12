"""Voice pipeline tests. No network, no microphone: everything is injected."""

import json
import threading
import time
import urllib.error

import pytest

from brain import vocab
from brain.voice.groq_client import GroqClient, GroqError
from brain.voice.keywords import keyword_intent
from brain.voice.listener import VoiceListener


@pytest.mark.parametrize("text,expected", [
    ("avanza", vocab.ADVANCE), ("Adelante", vocab.ADVANCE),
    ("para", vocab.STOP), ("detente", vocab.STOP), ("¡Alto!", vocab.STOP),
    ("gira a la izquierda", vocab.TURN_LEFT), ("gira a la derecha", vocab.TURN_RIGHT),
    ("siéntate", vocab.SIT), ("sientate", vocab.SIT),
    ("párate", vocab.STAND), ("levántate", vocab.STAND),
    ("salúdame", vocab.WAVE), ("saluda", vocab.WAVE),
    ("festeja", vocab.CELEBRATE), ("campeones", vocab.CELEBRATE),
    ("reproduce", vocab.MUSIC_PLAY), ("pon música", vocab.MUSIC_PLAY),
    ("pausa", vocab.MUSIC_PAUSE), ("siguiente", vocab.MUSIC_NEXT),
    ("anterior", vocab.MUSIC_PREV),
    ("detén la música", vocab.MUSIC_STOP), ("deten la musica", vocab.MUSIC_STOP),
    ("modo música", vocab.SET_MODE_MUSIC), ("cambia a modo música", vocab.SET_MODE_MUSIC),
    ("modo movimiento", vocab.SET_MODE_MOVEMENT),
    ("volvamos a los gestos", vocab.SET_INPUT_GESTURES),
    ("", vocab.UNKNOWN), ("blah blah", vocab.UNKNOWN),
])
def test_keyword_table(text, expected):
    assert keyword_intent(text) == expected


def test_keyword_ambiguous_cases():
    assert keyword_intent("detén la música") == vocab.MUSIC_STOP != vocab.STOP
    assert keyword_intent("pon música") == vocab.MUSIC_PLAY
    assert keyword_intent("cambia a modo música") == vocab.SET_MODE_MUSIC
    assert keyword_intent("detente") == vocab.STOP


def test_keyword_always_in_vocab():
    for t in ["avanza", "xyz", "modo movimiento", "gira"]:
        assert keyword_intent(t) in vocab.VOICE_INTENTS


class FakeResp:
    def __init__(self, payload):
        self._b = payload if isinstance(payload, bytes) else json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def make_opener(responses, calls=None):
    it = iter(responses)

    def opener(req, timeout=None):
        if calls is not None:
            calls.append((req, timeout))
        r = next(it)
        if isinstance(r, Exception):
            raise r
        return FakeResp(r)
    return opener


def chat(content):
    return {"choices": [{"message": {"content": content}}]}


def test_transcribe_builds_multipart_and_parses():
    calls = []
    c = GroqClient("KEY", opener=make_opener([{"text": " avanza "}], calls))
    assert c.transcribe(b"RIFFwav", language="es") == "avanza"
    req, timeout = calls[0]
    assert req.full_url.endswith("/audio/transcriptions")
    assert req.get_header("Authorization") == "Bearer KEY"
    assert "multipart/form-data; boundary=" in req.get_header("Content-type")
    assert b"whisper-large-v3-turbo" in req.data and b"RIFFwav" in req.data
    assert timeout and timeout <= 15


def test_classify_uses_enum_schema_and_validates():
    calls = []
    c = GroqClient("K", opener=make_opener([chat('{"intent": "SIT"}')], calls))
    assert c.classify("siéntate") == "SIT"
    body = json.loads(calls[0][0].data)
    assert body["model"] == "openai/gpt-oss-20b"
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["intent"]["enum"] == list(vocab.VOICE_INTENTS)


@pytest.mark.parametrize("content", ['{"intent": "FLY"}', "not json", '{"x": 1}', '{"intent": 3}'])
def test_classify_out_of_set_is_unknown(content):
    c = GroqClient("K", opener=make_opener([chat(content)]))
    assert c.classify("x") == vocab.UNKNOWN


def test_errors_raise_groqerror():
    for exc in (urllib.error.URLError("down"), TimeoutError(), OSError("x")):
        c = GroqClient("K", opener=make_opener([exc]))
        with pytest.raises(GroqError):
            c.classify("x")
    c = GroqClient("K", opener=make_opener([b"<html>"]))
    with pytest.raises(GroqError):
        c.transcribe(b"w")


class FakeRecorder:
    def __init__(self):
        self.started = 0

    def start(self):
        self.started += 1

    def stop(self):
        return b"WAV"


class FakeClient:
    def __init__(self, text="avanza", intent="ADVANCE", exc=None, gate=None, classify_exc=None):
        self.text, self.intent, self.exc, self.gate = text, intent, exc, gate
        self.classify_exc = classify_exc

    def transcribe(self, wav, language="es"):
        if self.gate:
            self.gate.wait(2)
        if self.exc:
            raise self.exc
        return self.text

    def classify(self, text):
        if self.classify_exc:
            raise self.classify_exc
        return self.intent


def run(client, recorder=None):
    got, done = [], threading.Event()

    def cb(intent, text):
        got.append((intent, text))
        done.set()
    l = VoiceListener(cb, client=client, recorder=recorder or FakeRecorder())
    l.begin()
    l.end()
    assert done.wait(2)
    return l, got


def test_flow_uses_groq():
    _, got = run(FakeClient("hola", "WAVE"))
    assert got == [("WAVE", "hola")]


def test_fallback_on_classify_error_uses_keywords():
    _, got = run(FakeClient("siéntate", classify_exc=GroqError("boom")))
    assert got == [(vocab.SIT, "siéntate")]


def test_transcribe_error_without_transcript_is_unknown():
    _, got = run(FakeClient(exc=GroqError("boom")))
    assert got == [(vocab.UNKNOWN, "")]


def test_no_client_uses_keywords_on_empty_transcript_unknown():
    class R(FakeRecorder):
        pass
    # no client and no transcript possible -> UNKNOWN
    _, got = run(None)
    assert got == [(vocab.UNKNOWN, "")]


def test_end_does_not_block_and_busy_ignores_begin():
    gate = threading.Event()
    rec = FakeRecorder()
    got, done = [], threading.Event()
    l = VoiceListener(lambda i, t: (got.append(i), done.set()),
                      client=FakeClient(gate=gate), recorder=rec)
    l.begin()
    t0 = time.time()
    l.end()
    assert time.time() - t0 < 0.5
    assert l.busy
    l.begin()
    assert rec.started == 1
    gate.set()
    assert done.wait(2)
    for _ in range(100):
        if not l.busy:
            break
        time.sleep(0.01)
    assert not l.busy
    assert got == ["ADVANCE"]
    l.begin()
    assert rec.started == 2


def test_end_without_begin_is_noop():
    l = VoiceListener(lambda i, t: None, client=None, recorder=FakeRecorder())
    l.end()
    assert not l.busy


def test_soundevice_recorder_missing_dep_clear_error(monkeypatch):
    import builtins
    from brain.voice.listener import SoundDeviceRecorder
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "sounddevice":
            raise ImportError("no")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(RuntimeError, match="sounddevice"):
        SoundDeviceRecorder().start()
