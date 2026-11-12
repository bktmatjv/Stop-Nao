"""Push-to-talk listener: record -> transcribe -> classify in a daemon thread."""

from __future__ import annotations

import io
import threading
import wave

from brain import vocab
from brain.voice.groq_client import GroqError
from brain.voice.keywords import keyword_intent

SAMPLE_RATE = 16000


class SoundDeviceRecorder:
    """16 kHz mono 16-bit WAV capture. `sounddevice` is optional and imported lazily."""

    def __init__(self, samplerate: int = SAMPLE_RATE):
        self.samplerate = samplerate
        self._stream = None
        self._chunks: list = []

    def start(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "Voice input needs the optional 'sounddevice' package (pip install sounddevice)"
            ) from exc
        self._chunks = []
        self._stream = sd.InputStream(
            samplerate=self.samplerate, channels=1, dtype="int16",
            callback=lambda indata, frames, t, status: self._chunks.append(indata.copy()),
        )
        self._stream.start()

    def stop(self) -> bytes:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._chunks:
            import numpy as np
            pcm = np.concatenate(self._chunks).astype("int16").tobytes()
        else:
            pcm = b""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.samplerate)
            w.writeframes(pcm)
        return buf.getvalue()


class VoiceListener:
    def __init__(self, on_intent, client=None, recorder=None):
        self._on_intent = on_intent
        self._client = client
        self._recorder = recorder or SoundDeviceRecorder()
        self._lock = threading.Lock()
        self._recording = False
        self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy or self._recording

    def begin(self) -> None:
        with self._lock:
            if self._busy or self._recording:
                return
            self._recording = True
        try:
            self._recorder.start()
        except Exception:
            with self._lock:
                self._recording = False
            raise

    def end(self) -> None:
        with self._lock:
            if not self._recording:
                return
            self._recording = False
            self._busy = True
        threading.Thread(target=self._work, daemon=True, name="voice-listener").start()

    def _work(self) -> None:
        intent, text = vocab.UNKNOWN, ""
        try:
            wav = self._recorder.stop()
            if self._client is not None:
                try:
                    text = self._client.transcribe(wav, language="es")
                    intent = self._client.classify(text) if text else vocab.UNKNOWN
                except GroqError:
                    intent = keyword_intent(text) if text else vocab.UNKNOWN
        except Exception:
            intent = vocab.UNKNOWN
        finally:
            self._busy = False
        try:
            self._on_intent(intent, text)
        except Exception:
            pass
