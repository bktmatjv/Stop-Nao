"""Runtime configuration loaded from environment variables and an optional ``.env`` file.

The ``.env`` parser is deliberately tiny (KEY=VALUE, ``#`` comments, optional quotes)
so the project needs no extra dependency. Real environment variables win over the file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def parse_env_file(path: Path | str) -> dict[str, str]:
    """Parse a ``.env`` file; a missing file yields an empty dict."""
    values: dict[str, str] = {}
    p = Path(path)
    if not p.is_file():
        return values
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


@dataclass(frozen=True)
class Config:
    nao_ip: str = "127.0.0.1"
    nao_port: int = 9559
    groq_api_key: str = ""
    music_dir: str = "songs"
    champions_song: str = ""

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key)


def load_config(env_path: Path | str | None = None, environ: dict | None = None) -> Config:
    """Build a Config: process environment overrides the ``.env`` file, which overrides defaults."""
    merged = parse_env_file(env_path if env_path is not None else DEFAULT_ENV_PATH)
    merged.update(os.environ if environ is None else environ)
    defaults = Config()
    try:
        port = int(merged.get("NAO_PORT", defaults.nao_port))
    except ValueError:
        port = defaults.nao_port
    return Config(
        nao_ip=merged.get("NAO_IP", defaults.nao_ip),
        nao_port=port,
        groq_api_key=merged.get("GROQ_API_KEY", ""),
        music_dir=merged.get("MUSIC_DIR", defaults.music_dir),
        champions_song=merged.get("CHAMPIONS_SONG", ""),
    )
