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
