"""Background music player for celebrations and dances.

Finds audio files in archive/dance-ai and songs/ directories, and plays
them asynchronously without blocking the robot control loop.
"""

from __future__ import annotations

import os
import random
import sys
import threading
from pathlib import Path

# Directories to search for songs
SONG_SEARCH_DIRS = [
    Path("songs"),
    Path("_archive/dance-ai/data/songs"),
    Path("_archive/dance-ai/data/demo_songs"),
    Path("data/songs"),
]

AUDIO_EXTENSIONS = ("*.wav", "*.mp3", "*.ogg", "*.flac")


class MusicPlayer:
    """Non-blocking audio player for celebratory robot moments."""

    def __init__(self, search_dirs: list[Path | str] | None = None):
        if search_dirs is None:
            self.search_dirs = [Path(p) for p in SONG_SEARCH_DIRS]
        else:
            self.search_dirs = [Path(p) for p in search_dirs]
        self._playing = False
        self._lock = threading.Lock()
        self._current_track: str | None = None

    def scan_songs(self) -> list[Path]:
        """Return a sorted list of all unique audio files found."""
        found = []
        seen = set()
        for directory in self.search_dirs:
            if not directory.exists() or not directory.is_dir():
                continue
            for ext in AUDIO_EXTENSIONS:
                for path in directory.glob(ext):
                    abs_p = path.resolve()
                    if abs_p not in seen:
                        seen.add(abs_p)
                        found.append(path)
        return sorted(found)

    def play_random(self) -> str | None:
        """Select a random song and play it asynchronously.

        Returns the filename if playback started, or None if no songs found.
        """
        songs = self.scan_songs()
        if not songs:
            print("[Musica] No se encontraron canciones en songs/ ni en archive/dance-ai.", file=sys.stderr)
            return None

        chosen = random.choice(songs)
        self.play_file(chosen)
        return chosen.name

    def play_file(self, path: Path | str) -> bool:
        """Play a specific song asynchronously without blocking."""
        file_path = Path(path)
        if not file_path.is_file():
            print(f"[Musica] Archivo no encontrado: {file_path}", file=sys.stderr)
            return False

        self.stop()
        track_name = file_path.name
        self._current_track = track_name
        print(f"[Musica] Reproduciendo: {track_name}", flush=True)

        def _play_worker():
            with self._lock:
                self._playing = True
            played = False

            # Try sounddevice + soundfile first
            try:
                import sounddevice as sd
                import soundfile as sf
                data, sr = sf.read(str(file_path), dtype="float32")
                sd.play(data, sr)
                played = True
                # Wait until playback finishes or stop is requested
                sd.wait()
            except Exception as exc:
                # If sounddevice failed or not available, fallback to winsound on Windows
                if sys.platform == "win32" and file_path.suffix.lower() == ".wav":
                    try:
                        import winsound
                        winsound.PlaySound(str(file_path), winsound.SND_ASYNC | winsound.SND_FILENAME)
                        played = True
                    except Exception as w_exc:
                        print(f"[Musica] Error con winsound: {w_exc}", file=sys.stderr)
                else:
                    print(f"[Musica] Error al reproducir audio: {exc}", file=sys.stderr)

            with self._lock:
                self._playing = False

        thread = threading.Thread(target=_play_worker, daemon=True, name="MusicPlayerThread")
        thread.start()
        return True

    def stop(self) -> None:
        """Stop any ongoing audio playback immediately."""
        with self._lock:
            self._playing = False
            self._current_track = None

        # Stop sounddevice if active
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass

        # Stop winsound if active on Windows
        if sys.platform == "win32":
            try:
                import winsound
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._playing

    @property
    def current_track(self) -> str | None:
        with self._lock:
            return self._current_track
