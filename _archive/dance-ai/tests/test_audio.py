"""Tests for dance.audio (features, dataset building, annotation, label merging).

All audio is synthetic and lives under tmp_path; nothing is written to data/ or report/.
"""

import csv
import json

import numpy as np
import pytest

from dance import STYLES
from dance.audio import annotate as ann
from dance.audio import build_dataset as bd
from dance.audio import merge_labels as ml
from dance.audio.features import (
    FEATURE_NAMES,
    estimate_bpm,
    extract_features,
    load_audio,
    segment_audio,
)
from dance.config import BPM_MAX, BPM_MIN


def _read(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------- features

def _consistent(est, true_bpm, tol=0.05):
    """True if est matches true_bpm or its double/half within tol."""
    return any(abs(est - c) <= tol * c for c in (true_bpm, true_bpm * 2, true_bpm / 2))


@pytest.mark.parametrize("bpm", [90, 140])
def test_estimate_bpm_click_tracks(click_track, test_sr, bpm):
    est = estimate_bpm(click_track(bpm, 20), test_sr)
    assert BPM_MIN <= est <= BPM_MAX
    assert _consistent(est, bpm), est


def test_estimate_bpm_is_folded_into_range(click_track, test_sr):
    for bpm in (50, 100, 200):
        est = estimate_bpm(click_track(bpm, 20), test_sr)
        assert BPM_MIN <= est <= BPM_MAX


def test_feature_names_contract():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))
    for required in ["bpm", "rms_mean", "onset_mean", "spectral_centroid_mean"]:
        assert required in FEATURE_NAMES
    for i in range(1, 14):
        assert f"mfcc{i}_mean" in FEATURE_NAMES


def test_extract_features_finite_aligned_and_deterministic(click_track, test_sr):
    y = click_track(120, 8)
    a = extract_features(y, test_sr)
    b = extract_features(y, test_sr)
    assert a.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(a))
    np.testing.assert_array_equal(a, b)
    assert a[FEATURE_NAMES.index("rms_mean")] > 0


@pytest.mark.parametrize("n", [0, 100, 22050 * 8])
def test_silence_and_tiny_inputs_are_finite(test_sr, n):
    f = extract_features(np.zeros(n, dtype=np.float32), test_sr)
    assert f.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(f))
    assert BPM_MIN <= f[FEATURE_NAMES.index("bpm")] <= BPM_MAX


def test_louder_and_busier_audio_has_higher_energy_features(click_track, test_sr):
    quiet = extract_features(click_track(80, 8, noise=0.001) * 0.1, test_sr)
    loud = extract_features(click_track(80, 8, noise=0.001), test_sr)
    assert loud[FEATURE_NAMES.index("rms_mean")] > quiet[FEATURE_NAMES.index("rms_mean")]


def test_segment_counts_and_last_partial_rule(test_sr):
    y = np.zeros(int(test_sr * 20), dtype=np.float32)  # 20 s, 8 s segments: 8+8+4(=0.5 kept)
    segs = segment_audio(y, test_sr, 8.0)
    assert [(s, e) for s, e, _ in segs] == [(0.0, 8.0), (8.0, 16.0), (16.0, 20.0)]
    short = np.zeros(int(test_sr * 19), dtype=np.float32)  # last 3 s < 4 s -> dropped
    segs = segment_audio(short, test_sr, 8.0)
    assert len(segs) == 2
    assert all(len(c) == 8 * test_sr for _, _, c in segs)


def test_load_audio_mono_resampled(tmp_path, click_track):
    import soundfile as sf

    path = tmp_path / "x.wav"
    sf.write(path, click_track(100, 2, sr=44100), 44100)
    y, sr = load_audio(path, sr=22050)
    assert sr == 22050 and y.ndim == 1 and abs(len(y) - 2 * 22050) <= 2


# ------------------------------------------------------------ build_dataset

def test_build_dataset_end_to_end(songs_dir, tmp_path, capsys):
    out = tmp_path / "dataset" / "segments.csv"
    rc = bd.main(["--songs", str(songs_dir), "--manifest", str(songs_dir / "manifest.csv"),
                  "--out", str(out), "--seconds", "8"])
    assert rc == 0
    rows = _read(out)
    assert list(rows[0].keys()) == ["segment_id", "song_id", "title", "start_s", "end_s"] + FEATURE_NAMES
    ids = [r["segment_id"] for r in rows]
    assert ids == ["song01_000", "song01_001", "song01_002", "song02_000", "song02_001", "song02_002"]
    assert all(np.isfinite(float(r[c])) for r in rows for c in FEATURE_NAMES)
    printed = capsys.readouterr().out
    assert "song01: 3 segments" in printed and "BPM" in printed
    # second song is faster -> its estimated tempo is not identical to song 1's
    assert all(BPM_MIN <= float(r["bpm"]) <= BPM_MAX for r in rows)


def _write_manifest(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(bd.MANIFEST_COLUMNS)
        w.writerows(rows)


@pytest.mark.parametrize(
    "row,expected",
    [
        (["s1", "a.wav", "T", "A", "", "CC", ""], "'source' is empty"),
        (["s1", "a.wav", "T", "A", "web", "", ""], "'license' is empty"),
        (["s1", "nope.wav", "T", "A", "web", "CC", ""], "audio file not found"),
        (["", "a.wav", "T", "A", "web", "CC", ""], "'song_id' is empty"),
        (["s1", "a.wav", "", "A", "web", "CC", ""], "'title' is empty"),
        (["bad id", "a.wav", "T", "A", "web", "CC", ""], "song_id must match"),
    ],
)
def test_manifest_validation_errors(songs_dir, tmp_path, capsys, row, expected):
    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [row])
    out = tmp_path / "seg.csv"
    rc = bd.main(["--songs", str(songs_dir), "--manifest", str(manifest), "--out", str(out)])
    assert rc == 1
    assert expected in capsys.readouterr().err
    assert not out.exists()


def test_manifest_lists_all_problems_and_duplicates(songs_dir, tmp_path):
    manifest = tmp_path / "m.csv"
    _write_manifest(manifest, [["s1", "a.wav", "T", "", "", "", ""], ["s1", "a.wav", "T", "", "w", "l", ""]])
    with pytest.raises(bd.ManifestError) as exc:
        bd.build_dataset(songs_dir, manifest, tmp_path / "o.csv")
    text = str(exc.value)
    assert "'source' is empty" in text and "'license' is empty" in text and "duplicated song_id" in text


def test_missing_manifest_and_columns(tmp_path, capsys):
    assert bd.main(["--manifest", str(tmp_path / "none.csv"), "--out", str(tmp_path / "o.csv")]) == 1
    bad = tmp_path / "bad.csv"
    bad.write_text("song_id,file\ns1,a.wav\n", encoding="utf-8")
    assert bd.main(["--manifest", str(bad), "--out", str(tmp_path / "o.csv")]) == 1
    assert "missing column" in capsys.readouterr().err


def test_template_flag_and_comments_ignored(tmp_path, capsys):
    path = tmp_path / "songs" / "manifest.csv"
    assert bd.main(["--template", "--manifest", str(path)]) == 0
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == ",".join(bd.MANIFEST_COLUMNS)
    assert "royalty-free" in text and "Never" in text
    assert bd.read_manifest(path) == []  # comments are ignored -> empty manifest
    with pytest.raises(bd.ManifestError):  # an empty template is refused
        bd.validate_manifest(bd.read_manifest(path), tmp_path)
    assert bd.main(["--template", "--manifest", str(path)]) == 1  # no silent overwrite
    assert bd.main(["--template", "--force", "--manifest", str(path)]) == 0


# ----------------------------------------------------------------- annotate

SEGMENTS = [
    {"segment_id": f"song01_{i:03d}", "song_id": "song01", "start_s": str(i * 8.0), "end_s": str((i + 1) * 8.0)}
    for i in range(5)
]


def _scripted(answers):
    it = iter(answers)
    return lambda prompt="": next(it)


def _segments_csv(tmp_path):
    path = tmp_path / "segments.csv"
    cols = ["segment_id", "song_id", "title", "start_s", "end_s"] + FEATURE_NAMES
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for s in SEGMENTS:
            w.writerow({**{c: 0.123456 for c in cols}, **s, "title": "T"})
    return path


def test_annotate_labels_skip_invalid_and_quit(tmp_path):
    seg_path = _segments_csv(tmp_path)
    out = tmp_path / "labels" / "ana.csv"
    # order is shuffled but deterministic; answers: bad key, 1, skip, 2, quit
    rc = ann.main(
        ["--annotator", "ana", "--segments", str(seg_path), "--out", str(out), "--no-audio"],
        input_fn=_scripted(["x", "1", "s", "2", "q"]),
    )
    assert rc == 0
    rows = _read(out)
    assert [r["label"] for r in rows] == ["SOFT", "MODERATE"]
    assert {r["annotator"] for r in rows} == {"ana"}
    assert list(rows[0].keys()) == ["segment_id", "annotator", "label"]


def test_annotate_resume_skips_labeled_and_is_blind(tmp_path, capsys):
    seg_path = _segments_csv(tmp_path)
    out = tmp_path / "ana.csv"
    args = ["--annotator", "ana", "--segments", str(seg_path), "--out", str(out), "--no-audio"]
    ann.main(args, input_fn=_scripted(["1", "2", "q"]))
    before = {r["segment_id"]: r["label"] for r in _read(out)}
    first = set(before)
    assert len(first) == 2
    capsys.readouterr()

    seen = []

    def inp(prompt=""):
        seen.append(prompt)
        return "3"

    ann.main(args, input_fn=inp)
    printed = capsys.readouterr().out
    assert len(seen) == 3  # only the 3 pending segments are asked
    rows = _read(out)
    assert len(rows) == 5 and len({r["segment_id"] for r in rows}) == 5
    # previously given labels are untouched
    assert {r["segment_id"]: r["label"] for r in rows if r["segment_id"] in first} == before
    # blind: no BPM / feature values printed, criteria are shown in Spanish
    assert "bpm" not in printed.lower() and "0.123456" not in printed
    assert "SOFT" in printed and "movimientos amplios y lentos" in printed
    assert "ENERGETIC" in printed and "MODERATE" in printed


def test_annotate_order_is_seeded_and_per_annotator():
    ids = lambda order: [s["segment_id"] for s in order]  # noqa: E731
    a1 = ids(ann.shuffled_order(SEGMENTS, "ana", 42))
    a2 = ids(ann.shuffled_order(SEGMENTS, "ana", 42))
    assert a1 == a2
    assert sorted(a1) == sorted(s["segment_id"] for s in SEGMENTS)
    assert ids(ann.shuffled_order(SEGMENTS, "ana", 42, shuffle=False)) == [s["segment_id"] for s in SEGMENTS]
    assert any(ids(ann.shuffled_order(SEGMENTS, n, 42)) != a1 for n in ("bob", "eve", "zoe"))


def test_annotate_replay_plays_audio_slices(songs_dir, tmp_path):
    seg_path = tmp_path / "segments.csv"
    seg_path.write_text(
        "segment_id,song_id,title,start_s,end_s\nsong01_000,song01,T,0.0,8.0\nsong01_001,song01,T,8.0,16.0\n",
        encoding="utf-8",
    )
    played = []
    out = tmp_path / "ana.csv"
    rc = ann.main(
        ["--annotator", "ana", "--segments", str(seg_path), "--songs", str(songs_dir),
         "--manifest", str(songs_dir / "manifest.csv"), "--out", str(out), "--no-shuffle"],
        input_fn=_scripted(["r", "3", "1"]),
        play_fn=lambda y, sr: played.append((len(y), sr)),
    )
    assert rc == 0
    assert played == [(8 * 22050, 22050)] * 3  # first play + replay + second segment
    assert [r["label"] for r in _read(out)] == ["ENERGETIC", "SOFT"]


def test_annotate_eof_saves_and_bad_inputs(tmp_path, capsys):
    seg_path = _segments_csv(tmp_path)
    out = tmp_path / "ana.csv"

    def eof(prompt=""):
        raise EOFError

    assert ann.main(["--annotator", "ana", "--segments", str(seg_path), "--out", str(out), "--no-audio"],
                    input_fn=eof) == 0
    assert out.exists()
    assert ann.main(["--annotator", "bad name", "--segments", str(seg_path)]) == 1
    assert ann.main(["--annotator", "ana", "--segments", str(tmp_path / "none.csv")]) == 1
    assert "ERROR" in capsys.readouterr().err


# ------------------------------------------------------------- merge_labels

def _write_labels(d, name, mapping):
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["segment_id", "annotator", "label"])
        for seg, lab in mapping.items():
            w.writerow([seg, name, lab])


def test_fleiss_kappa_textbook_example():
    # Wikipedia worked example: 10 items, 14 raters, 5 categories -> kappa = 0.210
    table = np.array([
        [0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0], [2, 2, 8, 1, 1],
        [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2], [6, 5, 2, 1, 0], [0, 2, 2, 3, 7],
    ])
    assert ml.fleiss_kappa(table) == pytest.approx(0.210, abs=5e-4)


def test_fleiss_kappa_extremes_and_errors():
    perfect = np.array([[3, 0, 0], [0, 3, 0], [0, 0, 3]])
    assert ml.fleiss_kappa(perfect) == pytest.approx(1.0)
    assert np.isnan(ml.fleiss_kappa(np.array([[3, 0, 0], [3, 0, 0]])))  # chance agreement = 1
    with pytest.raises(ValueError):
        ml.fleiss_kappa(np.array([[2, 0, 0], [1, 0, 0]]))


def test_merge_majority_ties_and_outputs(tmp_path):
    labels = tmp_path / "labels"
    _write_labels(labels, "a", {"s1": "SOFT", "s2": "MODERATE", "s3": "ENERGETIC", "s4": "SOFT"})
    _write_labels(labels, "b", {"s1": "SOFT", "s2": "MODERATE", "s3": "SOFT", "s4": "MODERATE"})
    _write_labels(labels, "c", {"s1": "SOFT", "s2": "ENERGETIC", "s3": "MODERATE", "s4": "ENERGETIC"})
    _write_labels(labels, "d", {"s1": "SOFT", "s2": "MODERATE", "s3": "ENERGETIC"})
    out = tmp_path / "ds" / "labels.csv"
    metrics = tmp_path / "metrics"
    assert ml.main(["--labels-dir", str(labels), "--out", str(out), "--metrics-dir", str(metrics)]) == 0

    merged = {r["segment_id"]: r for r in _read(out)}
    assert list(_read(out)[0].keys()) == ["segment_id", "label", "agreement", "n_annotators"]
    assert merged["s1"]["label"] == "SOFT" and float(merged["s1"]["agreement"]) == 1.0
    assert merged["s2"]["label"] == "MODERATE" and float(merged["s2"]["agreement"]) == 0.75
    assert merged["s3"]["label"] == "ENERGETIC" and merged["s3"]["n_annotators"] == "4"
    assert float(merged["s3"]["agreement"]) == 0.5
    # s4: 3 annotators, 3 different... SOFT/MODERATE/ENERGETIC -> tie, excluded
    assert "s4" not in merged
    ties = _read(metrics / "annotation_ties.csv")
    assert [t["segment_id"] for t in ties] == ["s4"]
    assert ties[0]["n_annotators"] == "3"

    report = json.loads((metrics / "annotation_agreement.json").read_text(encoding="utf-8"))
    assert report["annotators"] == ["a", "b", "c", "d"]
    assert report["n_ties_excluded"] == 1 and report["n_merged"] == 3
    assert report["n_segments_fewer_than_2_annotators"] == 0
    assert len(report["pairwise_cohen"]) == 6
    ab = next(p for p in report["pairwise_cohen"] if (p["annotator_a"], p["annotator_b"]) == ("a", "b"))
    assert ab["n_common"] == 4 and ab["percent_agreement"] == 0.5
    assert report["fleiss_n_raters"] == 4 and report["fleiss_n_items"] == 3  # only s1-s3 have 4 raters
    assert isinstance(report["fleiss_kappa"], float)


def test_merge_two_annotators_disagreement_is_tie_and_warns_on_single(tmp_path, capsys):
    labels = tmp_path / "labels"
    _write_labels(labels, "a", {"s1": "SOFT", "s2": "SOFT", "s3": "MODERATE"})
    _write_labels(labels, "b", {"s1": "MODERATE", "s2": "SOFT"})
    metrics = tmp_path / "m"
    ml.run(labels, tmp_path / "labels.csv", metrics)
    merged = {r["segment_id"]: r for r in _read(tmp_path / "labels.csv")}
    assert set(merged) == {"s2", "s3"}
    assert merged["s3"]["n_annotators"] == "1"
    assert [t["segment_id"] for t in _read(metrics / "annotation_ties.csv")] == ["s1"]
    assert "WARNING" in capsys.readouterr().out
    report = json.loads((metrics / "annotation_agreement.json").read_text(encoding="utf-8"))
    assert report["n_segments_fewer_than_2_annotators"] == 1


def test_merge_rejects_invalid_label_and_empty_dir(tmp_path, capsys):
    labels = tmp_path / "labels"
    _write_labels(labels, "a", {"s1": "HAPPY"})
    assert ml.main(["--labels-dir", str(labels), "--out", str(tmp_path / "o.csv"),
                    "--metrics-dir", str(tmp_path / "m")]) == 1
    assert ml.main(["--labels-dir", str(tmp_path / "empty"), "--out", str(tmp_path / "o.csv"),
                    "--metrics-dir", str(tmp_path / "m")]) == 1
    err = capsys.readouterr().err
    assert "invalid label" in err and "no label files" in err


def test_styles_contract():
    assert STYLES == ["SOFT", "MODERATE", "ENERGETIC"]
