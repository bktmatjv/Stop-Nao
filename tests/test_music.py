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


# ---------------------------------------------------------------------------
# One audio source, one track at a time (real SoundBackend over a fake sounddevice).
# ---------------------------------------------------------------------------
import sys  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

from brain.music import SoundBackend  # noqa: E402


class FakeSoundDevice:
    def __init__(self):
        self.active = []
        self.plays = []
        self.overlaps = 0

    def play(self, data, sr):
        if self.active:  # real sounddevice would cut the old one; we want it halted BEFORE
            self.overlaps += 1
        self.active.append(data[0])
        self.plays.append(data[0])

    def stop(self):
        self.active.clear()


@pytest.fixture
def fake_audio(monkeypatch):
    sd = FakeSoundDevice()
    sf = types.SimpleNamespace(read=lambda path, dtype=None: ([Path(path).name], 1))
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    monkeypatch.setitem(sys.modules, "soundfile", sf)
    return sd


def _wait(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_play_next_previous_never_leave_two_tracks_sounding(music_dir, fake_audio):
    dj = DJPlayer(music_dir, backend=SoundBackend())
    assert dj.play()
    assert _wait(lambda: len(fake_audio.plays) == 1)
    for step, expected in ((dj.next, 2), (dj.next, 3), (dj.previous, 4)):
        assert step()
        assert _wait(lambda: len(fake_audio.plays) == expected)
        assert len(fake_audio.active) == 1
    assert fake_audio.overlaps == 0
    assert fake_audio.plays == ["a.mp3", "b.wav", "c.flac", "b.wav"]
    dj.stop()
    assert fake_audio.active == []


def test_champions_replaces_the_current_track(tmp_path, music_dir, fake_audio):
    champions = tmp_path / "champions.mp3"
    champions.write_bytes(b"x")
    dj = DJPlayer(music_dir, champions, backend=SoundBackend())
    dj.play()
    assert _wait(lambda: len(fake_audio.plays) == 1)
    assert dj.play_champions()
    assert _wait(lambda: len(fake_audio.plays) == 2)
    assert fake_audio.active == ["champions.mp3"] and fake_audio.overlaps == 0


def test_rapid_skips_leave_only_the_last_track(music_dir, fake_audio):
    dj = DJPlayer(music_dir, backend=SoundBackend())
    dj.play()
    for _ in range(4):
        dj.next()
    time.sleep(0.2)
    assert len(fake_audio.active) <= 1 and fake_audio.overlaps == 0


def test_a_track_ending_never_starts_another(music_dir, fake_audio):
    dj = DJPlayer(music_dir, backend=SoundBackend())
    dj.play()
    assert _wait(lambda: len(fake_audio.plays) == 1)
    fake_audio.active.clear()  # the track finishes by itself
    time.sleep(0.3)
    assert fake_audio.plays == ["a.mp3"]  # no chaining, no random pick
    assert dj.current_track == "a.mp3"
