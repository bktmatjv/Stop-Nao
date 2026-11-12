"""Minimal Groq client (OpenAI-compatible API) using only the standard library."""

from __future__ import annotations

import json
import urllib.request
import uuid

from brain import vocab

BASE_URL = "https://api.groq.com/openai/v1"
STT_MODEL = "whisper-large-v3-turbo"
LLM_MODEL = "openai/gpt-oss-20b"
TIMEOUT_S = 8.0

_SYSTEM = (
    "You classify a short Spanish voice command for a NAO robot into exactly one intent. "
    "Use UNKNOWN if it does not clearly match. Intents: " + ", ".join(vocab.VOICE_INTENTS)
)


class GroqError(Exception):
    """Any network, HTTP or protocol failure talking to Groq."""


class GroqClient:
    def __init__(self, api_key: str, opener=None, timeout: float = TIMEOUT_S):
        self.api_key = api_key
        self._open = opener or urllib.request.urlopen
        self.timeout = timeout

    def _send(self, req: urllib.request.Request) -> dict:
        try:
            with self._open(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except Exception as exc:  # urllib errors, timeouts, bad JSON
            raise GroqError(str(exc) or type(exc).__name__) from exc

    def transcribe(self, wav_bytes: bytes, language: str = "es") -> str:
        boundary = uuid.uuid4().hex
        parts = []
        for name, value in (("model", STT_MODEL), ("language", language), ("response_format", "json")):
            parts.append(
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
            )
        parts.append(
            (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.wav"\r\n'
             "Content-Type: audio/wav\r\n\r\n").encode() + wav_bytes + b"\r\n"
        )
        parts.append(f"--{boundary}--\r\n".encode())
        req = urllib.request.Request(
            f"{BASE_URL}/audio/transcriptions", data=b"".join(parts), method="POST",
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        data = self._send(req)
        text = data.get("text") if isinstance(data, dict) else None
        if not isinstance(text, str):
            raise GroqError("transcription response has no text")
        return text.strip()

    def classify(self, text: str) -> str:
        body = {
            "model": LLM_MODEL,
            "temperature": 0,
            "messages": [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": text}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "intent",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"intent": {"type": "string", "enum": list(vocab.VOICE_INTENTS)}},
                        "required": ["intent"],
                        "additionalProperties": False,
                    },
                },
            },
        }
        req = urllib.request.Request(
            f"{BASE_URL}/chat/completions", data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        data = self._send(req)
        try:
            intent = json.loads(data["choices"][0]["message"]["content"])["intent"]
        except (KeyError, IndexError, TypeError, ValueError):
            return vocab.UNKNOWN
        return intent if isinstance(intent, str) and intent in vocab.VOICE_INTENTS else vocab.UNKNOWN
