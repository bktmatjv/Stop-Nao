"""Build the segment-level feature dataset from the licensed songs.

Usage::

    python -m dance.audio.build_dataset --template --manifest data/songs/manifest.csv
    python -m dance.audio.build_dataset --songs data/songs \\
        --manifest data/songs/manifest.csv --out data/dataset/segments.csv

Provenance is mandatory: every song needs ``song_id, file, title, source,
license``. Nothing is ever downloaded by this tool.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

from dance.config import SAMPLE_RATE, SEGMENT_SECONDS

from .features import FEATURE_NAMES, extract_features, load_audio, segment_audio

MANIFEST_COLUMNS = ["song_id", "file", "title", "artist", "source", "license", "notes"]
REQUIRED_COLUMNS = ["song_id", "file", "title", "source", "license"]
SEGMENT_COLUMNS = ["segment_id", "song_id", "title", "start_s", "end_s"] + FEATURE_NAMES
_SONG_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

TEMPLATE = (
    ",".join(MANIFEST_COLUMNS)
    + "\n"
    "# Lines starting with '#' are ignored. One row per song (min. 10 songs for the project).\n"
    "# song_id: short unique id without spaces (letters, digits, _ or -), e.g. song01\n"
    "# file: audio file name inside data/songs/ (wav/mp3/flac/ogg)\n"
    "# title / artist: song metadata\n"
    "# source: where the audio came from (URL or 'owned by the team')\n"
    "# license: license or authorization (e.g. CC-BY 4.0, YouTube Audio Library, own work)\n"
    "# notes: optional\n"
    "# Use royalty-free / Creative Commons music (YouTube Audio Library, Free Music Archive)\n"
    "# or songs the team owns. Never use protected songs without authorization.\n"
    "# Example row (delete the leading '# ' to use it):\n"
    "# song01,song01.mp3,Example Title,Example Artist,https://freemusicarchive.org/...,CC-BY 4.0,\n"
)


class ManifestError(ValueError):
    """Raised when the manifest is invalid; ``problems`` lists every issue."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("Invalid manifest:\n  - " + "\n  - ".join(problems))


def write_template(path: Path, force: bool = False) -> None:
    path = Path(path)
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists (use --force to overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE, encoding="utf-8")


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Read the manifest rows, ignoring ``#`` comment lines."""
    path = Path(path)
    if not path.is_file():
        raise ManifestError([f"manifest not found: {path}"])
    with open(path, newline="", encoding="utf-8-sig") as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
    if missing:
        raise ManifestError([f"manifest is missing column(s): {', '.join(missing)}"])
    return [{k: (v or "").strip() for k, v in row.items() if k} for row in reader]


def validate_manifest(rows: list[dict[str, str]], songs_dir: Path) -> None:
    """Raise ``ManifestError`` listing every provenance/consistency problem."""
    problems: list[str] = []
    if not rows:
        problems.append("manifest has no rows")
    seen: set[str] = set()
    for i, row in enumerate(rows, start=1):
        tag = f"row {i} ({row.get('song_id') or '?'})"
        for col in REQUIRED_COLUMNS:
            if not row.get(col):
                problems.append(f"{tag}: '{col}' is empty (provenance is mandatory)")
        song_id = row.get("song_id", "")
        if song_id:
            if not _SONG_ID_RE.match(song_id):
                problems.append(f"{tag}: song_id must match [A-Za-z0-9_-]+")
            if song_id in seen:
                problems.append(f"{tag}: duplicated song_id")
            seen.add(song_id)
        file = row.get("file", "")
        if file and not (Path(songs_dir) / file).is_file():
            problems.append(f"{tag}: audio file not found: {Path(songs_dir) / file}")
    if problems:
        raise ManifestError(problems)


def build_dataset(
    songs_dir: Path,
    manifest: Path,
    out: Path,
    seconds: float = SEGMENT_SECONDS,
    sr: int = SAMPLE_RATE,
    log=print,
) -> list[dict]:
    """Segment every song, extract features and write the CSV. Returns the rows."""
    songs_dir = Path(songs_dir)
    rows = read_manifest(manifest)
    validate_manifest(rows, songs_dir)

    records: list[dict] = []
    summary: list[tuple[str, int, float, float]] = []
    for row in rows:
        y, sr_out = load_audio(songs_dir / row["file"], sr=sr)
        segments = segment_audio(y, sr_out, seconds)
        bpms = []
        for idx, (start, end, chunk) in enumerate(segments):
            vec = extract_features(chunk, sr_out)
            rec = {
                "segment_id": f"{row['song_id']}_{idx:03d}",
                "song_id": row["song_id"],
                "title": row["title"],
                "start_s": round(start, 3),
                "end_s": round(end, 3),
            }
            rec.update(dict(zip(FEATURE_NAMES, (float(v) for v in vec))))
            records.append(rec)
            bpms.append(rec["bpm"])
        if bpms:
            summary.append((row["song_id"], len(bpms), float(np.min(bpms)), float(np.max(bpms))))
            log(
                f"{row['song_id']}: {len(bpms)} segments, "
                f"BPM min/median/max = {np.min(bpms):.1f}/{np.median(bpms):.1f}/{np.max(bpms):.1f}"
            )
        else:
            log(f"{row['song_id']}: 0 segments (audio shorter than the minimum segment)")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SEGMENT_COLUMNS)
        writer.writeheader()
        writer.writerows(records)
    if records:
        all_bpm = [r["bpm"] for r in records]
        log(
            f"Wrote {len(records)} segments from {len(summary)} songs to {out} "
            f"(BPM overall min/mean/max = {min(all_bpm):.1f}/{np.mean(all_bpm):.1f}/{max(all_bpm):.1f})"
        )
    else:
        log(f"Wrote 0 segments to {out}")
    return records


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build the segment feature dataset (segments.csv).")
    p.add_argument("--songs", default="data/songs", help="directory with the audio files")
    p.add_argument("--manifest", default="data/songs/manifest.csv")
    p.add_argument("--out", default="data/dataset/segments.csv")
    p.add_argument("--seconds", type=float, default=SEGMENT_SECONDS, help="segment length in seconds")
    p.add_argument("--template", action="store_true", help="write an empty manifest template and exit")
    p.add_argument("--force", action="store_true", help="overwrite the manifest with --template")
    args = p.parse_args(argv)

    if args.template:
        try:
            write_template(Path(args.manifest), force=args.force)
        except FileExistsError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"Template written to {args.manifest}. Fill one row per song (with source and license).")
        return 0
    try:
        build_dataset(Path(args.songs), Path(args.manifest), Path(args.out), seconds=args.seconds)
    except ManifestError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
