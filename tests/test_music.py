"""Tests for brain.music (MusicPlayer)."""

import os
from pathlib import Path
import pytest
from brain.music import MusicPlayer


def test_music_player_scan_finds_songs():
    player = MusicPlayer()
    songs = player.scan_songs()
    assert isinstance(songs, list)
    assert len(songs) > 0
    assert any(s.name.endswith(".wav") for s in songs)


def test_music_player_empty_dir_returns_none(tmp_path):
    player = MusicPlayer(search_dirs=[tmp_path])
    assert player.scan_songs() == []
    assert player.play_random() is None


def test_music_player_stop_idempotent():
    player = MusicPlayer()
    player.stop()
    player.stop()
    assert player.is_playing is False
    assert player.current_track is None


# ---------------------------------------------------------------------------
# DJPlayer: playlist + transport controls on top of an injectable backend.
# ---------------------------------------------------------------------------

from brain.music import DJPlayer  # noqa: E402


class FakeBackend:
    """Records transport calls instead of touching real audio hardware."""

    def __init__(self):
        self.calls = []

    def play(self, path):
        self.calls.append(("play", Path(path).name))

    def pause(self):
        self.calls.append(("pause",))

    def resume(self):
        self.calls.append(("resume",))

    def stop(self):
        self.calls.append(("stop",))


def _make_tracks(folder, names):
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        (folder / name).write_bytes(b"RIFF")
    return folder


@pytest.fixture
def music_dir(tmp_path):
    return _make_tracks(tmp_path / "music", ["b.wav", "a.mp3", "c.flac"])


@pytest.fixture
def backend():
    return FakeBackend()


def test_dj_scan_returns_sorted_audio_files_only(tmp_path, backend):
    folder = _make_tracks(tmp_path / "m", ["b.wav", "a.mp3", "notes.txt", "d.OGG"])
    (folder / "sub.wav").mkdir()  # a directory named like a track is not a track
    dj = DJPlayer(folder, backend=backend)
    assert [p.name for p in dj.scan()] == ["a.mp3", "b.wav", "d.OGG"]


def test_dj_missing_folder_is_safe(tmp_path, backend):
    dj = DJPlayer(tmp_path / "does-not-exist", backend=backend)
    assert dj.scan() == []
    assert dj.play() is False
    assert dj.pause() is False
    assert dj.resume() is False
    assert dj.next() is False
    assert dj.previous() is False
    assert dj.play_champions() is False
    dj.stop()
    assert dj.is_playing is False
    assert dj.is_paused is False
    assert dj.current_track is None
    assert backend.calls == []


def test_dj_play_starts_first_track(music_dir, backend):
    dj = DJPlayer(music_dir, backend=backend)
    assert dj.play() is True
    assert dj.is_playing is True
    assert dj.is_paused is False
    assert dj.current_track == "a.mp3"
    assert backend.calls == [("play", "a.mp3")]


def test_dj_play_while_playing_is_noop(music_dir, backend):
    dj = DJPlayer(music_dir, backend=backend)
    dj.play()
    assert dj.play() is True
    assert backend.calls == [("play", "a.mp3")]


def test_dj_pause_then_play_resumes_same_track(music_dir, backend):
    dj = DJPlayer(music_dir, backend=backend)
    dj.play()
    assert dj.pause() is True
    assert dj.is_paused is True
    assert dj.is_playing is False
    assert dj.current_track == "a.mp3"
    assert dj.play() is True
    assert dj.is_playing is True
    assert dj.is_paused is False
    assert backend.calls == [("play", "a.mp3"), ("pause",), ("resume",)]


def test_dj_pause_and_resume_invalid_states_return_false(music_dir, backend):
    dj = DJPlayer(music_dir, backend=backend)
    assert dj.pause() is False  # nothing playing
    assert dj.resume() is False  # not paused
    dj.play()
    assert dj.resume() is False  # playing, not paused
    assert backend.calls == [("play", "a.mp3")]


def test_dj_resume_explicit(music_dir, backend):
    dj = DJPlayer(music_dir, backend=backend)
    dj.play()
    dj.pause()
    assert dj.resume() is True
    assert dj.is_playing is True
    assert backend.calls[-1] == ("resume",)


def test_dj_next_advances_and_wraps(music_dir, backend):
    dj = DJPlayer(music_dir, backend=backend)
    dj.play()
    assert dj.next() is True
    assert dj.current_track == "b.wav"
    assert dj.next() is True
    assert dj.current_track == "c.flac"
    assert dj.next() is True
    assert dj.current_track == "a.mp3"  # wrapped around
    assert dj.is_playing is True
    assert backend.calls[-1] == ("play", "a.mp3")


def test_dj_previous_wraps_to_last(music_dir, backend):
    dj = DJPlayer(music_dir, backend=backend)
    dj.play()
    assert dj.previous() is True
    assert dj.current_track == "c.flac"  # wrapped from first to last
    assert dj.is_playing is True
    assert backend.calls[-1] == ("play", "c.flac")


def test_dj_next_while_paused_changes_track_without_playing(music_dir, backend):
    dj = DJPlayer(music_dir, backend=backend)
    dj.play()
    dj.pause()
    assert dj.next() is True
    assert dj.current_track == "b.wav"
    assert dj.is_paused is True
    assert dj.is_playing is False
    assert backend.calls == [("play", "a.mp3"), ("pause",)]


def test_dj_next_while_stopped_does_not_start_playback(music_dir, backend):
    dj = DJPlayer(music_dir, backend=backend)
    assert dj.next() is True
    assert dj.is_playing is False
    assert dj.current_track is None
    assert backend.calls == []


def test_dj_stop_resets_to_start(music_dir, backend):
    dj = DJPlayer(music_dir, backend=backend)
    dj.play()
    dj.next()
    dj.stop()
    assert dj.is_playing is False
    assert dj.is_paused is False
    assert dj.current_track is None
    assert backend.calls[-1] == ("stop",)
    assert dj.play() is True
    assert dj.current_track == "a.mp3"  # reset to the first track


def test_dj_stop_from_paused_clears_pause(music_dir, backend):
    dj = DJPlayer(music_dir, backend=backend)
    dj.play()
    dj.pause()
    dj.stop()
    assert dj.is_paused is False
    assert dj.resume() is False


def test_dj_stop_is_idempotent(music_dir, backend):
    dj = DJPlayer(music_dir, backend=backend)
    dj.stop()
    dj.stop()
    assert dj.is_playing is False
    assert dj.current_track is None


def test_dj_champions_missing_setting_returns_false(music_dir, backend):
    dj = DJPlayer(music_dir, champions_song=None, backend=backend)
    assert dj.play_champions() is False
    assert backend.calls == []


def test_dj_champions_missing_file_returns_false(music_dir, backend):
    dj = DJPlayer(music_dir, champions_song=str(music_dir / "nope.mp3"), backend=backend)
    assert dj.play_champions() is False
    assert dj.is_playing is False
    assert backend.calls == []


def test_dj_champions_plays_song_and_keeps_playlist_cursor(tmp_path, music_dir, backend):
    champions = tmp_path / "champions.mp3"
    champions.write_bytes(b"ID3")
    dj = DJPlayer(music_dir, champions_song=str(champions), backend=backend)
    dj.play()
    dj.next()  # cursor on b.wav
    assert dj.play_champions() is True
    assert dj.is_playing is True
    assert dj.current_track == "champions.mp3"
    assert backend.calls[-1] == ("play", "champions.mp3")
    dj.stop()
    dj.play()
    assert dj.current_track == "a.mp3"  # stop resets the playlist cursor
