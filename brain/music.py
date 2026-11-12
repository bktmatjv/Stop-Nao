"""Background music player for celebrations and dances.

Finds audio files in archive/dance-ai and songs/ directories, and plays
them asynchronously without blocking the robot control loop.
"""

from __future__ import annotations

import os
import random
import sys
import threading
import time
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


DJ_AUDIO_SUFFIXES = (".wav", ".mp3", ".ogg", ".flac")


class SoundBackend:
    """Default audio output for DJPlayer: non-blocking sounddevice playback.

    Loading happens in a daemon thread so the caller never blocks. Pause keeps
    the elapsed position and resume restarts from it. On Windows, when
    sounddevice is unavailable, .wav files fall back to winsound (async); that
    fallback cannot seek, so resume restarts the track.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._path: Path | None = None
        self._offset = 0.0  # seconds played before the current segment
        self._segment_start: float | None = None  # monotonic start of the live segment
        self._generation = 0

    def play(self, path: Path | str) -> None:
        self._start(Path(path), 0.0)

    def resume(self) -> None:
        if self._path is not None:
            self._start(self._path, self._offset)

    def pause(self) -> None:
        with self._lock:
            if self._segment_start is not None:
                self._offset += time.monotonic() - self._segment_start
                self._segment_start = None
            self._generation += 1
            self._halt()

    def stop(self) -> None:
        with self._lock:
            self._generation += 1
            self._path = None
            self._offset = 0.0
            self._segment_start = None
            self._halt()

    def _start(self, path: Path, offset: float) -> None:
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._path = path
            self._offset = offset
            self._segment_start = None
        threading.Thread(
            target=self._load_and_play,
            args=(path, offset, generation),
            daemon=True,
            name="DJPlayerThread",
        ).start()

    def _load_and_play(self, path: Path, offset: float, generation: int) -> None:
        try:
            import sounddevice as sd
            import soundfile as sf

            data, sr = sf.read(str(path), dtype="float32")
            start_frame = int(offset * sr)
            with self._lock:
                if generation != self._generation:
                    return  # superseded by pause/stop/next before loading finished
                self._segment_start = time.monotonic()
                sd.play(data[start_frame:], sr)
        except Exception as exc:
            if sys.platform == "win32" and path.suffix.lower() == ".wav":
                try:
                    import winsound

                    with self._lock:
                        if generation == self._generation:
                            winsound.PlaySound(str(path), winsound.SND_ASYNC | winsound.SND_FILENAME)
                except Exception as w_exc:
                    print(f"[Musica] Error con winsound: {w_exc}", file=sys.stderr)
            else:
                print(f"[Musica] Error al reproducir audio: {exc}", file=sys.stderr)

    @staticmethod
    def _halt() -> None:
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass
        if sys.platform == "win32":
            try:
                import winsound

                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass


class DJPlayer:
    """Playlist player for the laptop DJ: scan, transport controls and a champions song.

    All methods return a bool or None and never raise for a missing or empty
    music folder. ``is_playing`` reflects the requested state; the player does
    not detect the end of a track.
    """

    _STOPPED = "stopped"
    _PLAYING = "playing"
    _PAUSED = "paused"

    def __init__(self, music_dir: Path | str, champions_song: Path | str | None = None, backend=None):
        self.music_dir = Path(music_dir)
        self.champions_song = Path(champions_song) if champions_song else None
        self._backend = backend if backend is not None else SoundBackend()
        self._lock = threading.Lock()
        self._state = self._STOPPED
        self._index = 0
        self._on_champions = False
        self._pending_load = False  # track changed while paused; next resume must load it

    def scan(self) -> list[Path]:
        """Return the sorted audio files directly inside ``music_dir`` (empty if missing)."""
        if not self.music_dir.is_dir():
            return []
        try:
            return sorted(
                p for p in self.music_dir.iterdir()
                if p.is_file() and p.suffix.lower() in DJ_AUDIO_SUFFIXES
            )
        except OSError:
            return []

    def play(self) -> bool:
        """Start the current track, or resume it if paused."""
        with self._lock:
            if self._state == self._PLAYING:
                return True
            if self._state == self._PAUSED:
                self._resume_locked()
                return True
            path = self._current_path()
            if path is None:
                return False
            self._backend.play(path)
            self._on_champions = False
            self._state = self._PLAYING
            return True

    def pause(self) -> bool:
        with self._lock:
            if self._state != self._PLAYING:
                return False
            self._backend.pause()
            self._state = self._PAUSED
            return True

    def resume(self) -> bool:
        with self._lock:
            if self._state != self._PAUSED:
                return False
            self._resume_locked()
            return True

    def next(self) -> bool:
        return self._move(1)

    def previous(self) -> bool:
        return self._move(-1)

    def stop(self) -> None:
        """Stop playback and reset the playlist to its first track."""
        with self._lock:
            if self._state != self._STOPPED:
                self._backend.stop()
            self._state = self._STOPPED
            self._index = 0
            self._on_champions = False
            self._pending_load = False

    def play_champions(self) -> bool:
        """Play the champions song, replacing any current playback. The playlist cursor is kept."""
        with self._lock:
            if self.champions_song is None or not self.champions_song.is_file():
                return False
            self._backend.play(self.champions_song)
            self._on_champions = True
            self._pending_load = False
            self._state = self._PLAYING
            return True

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._state == self._PLAYING

    @property
    def is_paused(self) -> bool:
        with self._lock:
            return self._state == self._PAUSED

    @property
    def current_track(self) -> str | None:
        with self._lock:
            if self._state == self._STOPPED:
                return None
            if self._on_champions:
                return self.champions_song.name
            path = self._current_path()
            return path.name if path is not None else None

    def _current_path(self) -> Path | None:
        tracks = self.scan()
        if not tracks:
            return None
        return tracks[self._index % len(tracks)]

    def _resume_locked(self) -> None:
        if self._pending_load:
            self._backend.play(self._current_path())
            self._pending_load = False
            self._on_champions = False
        else:
            self._backend.resume()
        self._state = self._PLAYING

    def _move(self, step: int) -> bool:
        with self._lock:
            tracks = self.scan()
            if not tracks:
                return False
            self._index = (self._index + step) % len(tracks)
            if self._state == self._PLAYING:
                self._backend.play(tracks[self._index])
                self._on_champions = False
            elif self._state == self._PAUSED:
                self._pending_load = True
                self._on_champions = False
            return True
