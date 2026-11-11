"""Shared pytest fixtures.

Synthetic audio generated here is ONLY for tests; it is always written under
pytest's ``tmp_path`` and never into ``data/``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEST_SR = 22050


def make_click_track(bpm: float, seconds: float, sr: int = TEST_SR, noise: float = 0.01, seed: int = 0):
    """Click track at a known BPM (short decaying bursts) plus low-level noise."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    y = noise * rng.standard_normal(n)
    click_len = int(0.03 * sr)
    t = np.arange(click_len) / sr
    click = np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 150)
    period = 60.0 / bpm
    pos = 0.0
    while int(pos * sr) + click_len < n:
        i = int(pos * sr)
        y[i : i + click_len] += click
        pos += period
    return y.astype(np.float32)


@pytest.fixture
def click_track():
    """Factory: ``click_track(bpm, seconds)`` -> float32 array at TEST_SR."""
    return make_click_track


@pytest.fixture
def test_sr():
    return TEST_SR


@pytest.fixture
def songs_dir(tmp_path):
    """Two short synthetic songs (90 and 140 BPM, 20 s) with a valid manifest."""
    d = tmp_path / "songs"
    d.mkdir()
    sf.write(d / "a.wav", make_click_track(90, 20, seed=1), TEST_SR)
    sf.write(d / "b.wav", make_click_track(140, 20, seed=2), TEST_SR)
    (d / "manifest.csv").write_text(
        "song_id,file,title,artist,source,license,notes\n"
        "song01,a.wav,Click 90,Synth,synthetic test,test-only,\n"
        "song02,b.wav,Click 140,Synth,synthetic test,test-only,\n",
        encoding="utf-8",
    )
    return d
