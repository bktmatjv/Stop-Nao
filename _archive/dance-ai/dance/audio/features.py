"""Audio loading, segmentation and feature extraction.

The same functions are used offline (dataset building) and online (inference),
so the feature vector is always computed identically.

Features computed per segment (all scalar, aligned with ``FEATURE_NAMES``):

* ``bpm``: tempo estimated by ``librosa.beat.beat_track`` on the onset
  envelope, folded into ``[BPM_MIN, BPM_MAX]`` by doubling/halving (removes
  the usual octave ambiguity of beat trackers). Silence maps to ``BPM_MIN``.
* ``beat_strength``: mean onset-strength value at the tracked beat frames
  (how pronounced the beat is). 0 when no beat is found.
* ``onset_mean`` / ``onset_std``: mean and standard deviation of the onset
  strength envelope (overall rhythmic activity and its variability).
* ``onset_rate``: detected onsets per second.
* ``rms_mean`` / ``rms_std``: mean and standard deviation of the frame RMS
  energy (loudness and its dynamics).
* ``spectral_centroid_mean`` / ``spectral_centroid_std``: mean/std of the
  spectral centroid in Hz (brightness of the sound).
* ``spectral_rolloff_mean``: mean frequency (Hz) under which 85 % of the
  spectral energy lies.
* ``spectral_bandwidth_mean``: mean spectral bandwidth in Hz.
* ``spectral_flatness_mean``: mean spectral flatness (noisy vs tonal).
* ``zcr_mean``: mean zero-crossing rate (percussiveness / noisiness).
* ``percussive_ratio``: energy of the percussive component of a harmonic-
  percussive source separation divided by the total energy (0..1).
* ``mfcc1_mean..mfcc13_mean`` and ``mfcc1_std..mfcc13_std``: mean and
  standard deviation over time of the first 13 MFCCs (timbre).

Every feature is forced to be finite (NaN/inf replaced by 0) so silent or very
short segments never break training or inference. The code is deterministic.
"""

from __future__ import annotations

import warnings

import librosa
import numpy as np

from dance.config import BPM_MAX, BPM_MIN, MIN_SEGMENT_FRACTION, SAMPLE_RATE, SEGMENT_SECONDS

N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

FEATURE_NAMES: list[str] = (
    [
        "bpm",
        "beat_strength",
        "onset_mean",
        "onset_std",
        "onset_rate",
        "rms_mean",
        "rms_std",
        "spectral_centroid_mean",
        "spectral_centroid_std",
        "spectral_rolloff_mean",
        "spectral_bandwidth_mean",
        "spectral_flatness_mean",
        "zcr_mean",
        "percussive_ratio",
    ]
    + [f"mfcc{i}_mean" for i in range(1, N_MFCC + 1)]
    + [f"mfcc{i}_std" for i in range(1, N_MFCC + 1)]
)


def load_audio(path, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load an audio file as mono float32 resampled to ``sr``."""
    y, sr_out = librosa.load(str(path), sr=sr, mono=True)
    return np.asarray(y, dtype=np.float32), int(sr_out)


def segment_audio(
    y: np.ndarray, sr: int, seconds: float = SEGMENT_SECONDS
) -> list[tuple[float, float, np.ndarray]]:
    """Split ``y`` into consecutive ``seconds``-long segments.

    Returns ``(start_s, end_s, samples)`` tuples. A trailing partial segment is
    kept only if it is at least ``MIN_SEGMENT_FRACTION`` of a full segment.
    """
    size = int(round(seconds * sr))
    if size <= 0:
        raise ValueError("segment length must be positive")
    segments = []
    for start in range(0, len(y), size):
        chunk = y[start : start + size]
        if len(chunk) < size and len(chunk) < MIN_SEGMENT_FRACTION * size:
            continue
        segments.append((start / sr, (start + len(chunk)) / sr, chunk))
    return segments


def _fold_bpm(bpm: float) -> float:
    """Fold a tempo into [BPM_MIN, BPM_MAX] by doubling/halving."""
    if not np.isfinite(bpm) or bpm <= 0:
        return float(BPM_MIN)
    while bpm < BPM_MIN:
        bpm *= 2.0
    while bpm > BPM_MAX:
        bpm /= 2.0
    return float(bpm)


def _pad(y: np.ndarray) -> np.ndarray:
    """Pad very short signals so that STFT-based features are defined."""
    y = np.asarray(y, dtype=np.float32)
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))
    return y


def _tempo_and_beats(y: np.ndarray, sr: int) -> tuple[float, np.ndarray, np.ndarray]:
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)
    tempo = float(np.atleast_1d(tempo)[0])
    return _fold_bpm(tempo), onset_env, np.asarray(beats, dtype=int)


def estimate_bpm(y: np.ndarray, sr: int) -> float:
    """Estimate the tempo in BPM, folded into [BPM_MIN, BPM_MAX]."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bpm, _, _ = _tempo_and_beats(_pad(y), sr)
    return bpm


def _finite(value: float) -> float:
    value = float(value)
    return value if np.isfinite(value) else 0.0


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Return the feature vector of one segment, aligned with ``FEATURE_NAMES``."""
    y = _pad(y)
    duration = len(y) / sr
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        bpm, onset_env, beats = _tempo_and_beats(y, sr)
        beats = beats[beats < len(onset_env)]
        beat_strength = float(np.mean(onset_env[beats])) if len(beats) else 0.0
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=HOP_LENGTH)

        rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
        magnitude = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
        centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
        rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr)[0]
        bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)[0]
        flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]

        harmonic, percussive = librosa.decompose.hpss(magnitude)
        e_perc = float(np.sum(percussive**2))
        e_harm = float(np.sum(harmonic**2))
        percussive_ratio = e_perc / (e_perc + e_harm) if (e_perc + e_harm) > 0 else 0.0

        mel = librosa.feature.melspectrogram(S=magnitude**2, sr=sr)
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=N_MFCC)

        values = [
            bpm,
            beat_strength,
            np.mean(onset_env),
            np.std(onset_env),
            len(onsets) / duration,
            np.mean(rms),
            np.std(rms),
            np.mean(centroid),
            np.std(centroid),
            np.mean(rolloff),
            np.mean(bandwidth),
            np.mean(flatness),
            np.mean(zcr),
            percussive_ratio,
        ]
        values += list(np.mean(mfcc, axis=1)) + list(np.std(mfcc, axis=1))

    features = np.array([_finite(v) for v in values], dtype=np.float64)
    assert len(features) == len(FEATURE_NAMES)
    return features


def feature_dict(y: np.ndarray, sr: int) -> dict[str, float]:
    """Same as ``extract_features`` but as a name -> value mapping."""
    return dict(zip(FEATURE_NAMES, extract_features(y, sr).tolist()))


__all__ = [
    "FEATURE_NAMES",
    "load_audio",
    "segment_audio",
    "estimate_bpm",
    "extract_features",
    "feature_dict",
]
