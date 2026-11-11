"""Blind, resumable annotation of segments by listening.

Usage::

    python -m dance.audio.annotate --annotator ana

Each member labels every segment by listening to it and deciding which NAO
movement style fits. BPM and features are never shown (blind labeling), and
the segment order is shuffled per annotator (seeded, so resuming is stable).
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from pathlib import Path

from dance import STYLES
from dance.config import SAMPLE_RATE

CRITERIA = """\
Criterios de etiquetado (escucha el fragmento y decide que estilo de baile le queda mejor al robot):
  SOFT       = musica tranquila/lenta, movimientos amplios y lentos
  MODERATE   = ritmo marcado medio, movimientos regulares
  ENERGETIC  = musica intensa/rapida, movimientos rapidos y energicos
Etiqueta por tu impresion al escuchar; no uses reglas numericas.
"""

KEY_TO_LABEL = {"1": "SOFT", "2": "MODERATE", "3": "ENERGETIC"}
PROMPT = "[1=SOFT 2=MODERATE 3=ENERGETIC | r=repetir s=saltar q=guardar y salir] > "
LABEL_COLUMNS = ["segment_id", "annotator", "label"]


def read_segments(path: Path) -> list[dict[str, str]]:
    """Read only the identifying columns of segments.csv (features stay hidden)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"segments file not found: {path} (run build_dataset first)")
    with open(path, newline="", encoding="utf-8") as fh:
        return [
            {k: row[k] for k in ("segment_id", "song_id", "start_s", "end_s")}
            for row in csv.DictReader(fh)
        ]


def read_labels(path: Path) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["segment_id"]: row["label"] for row in csv.DictReader(fh)}


def write_labels(path: Path, annotator: str, labels: dict[str, str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(LABEL_COLUMNS)
        for seg_id, label in labels.items():
            writer.writerow([seg_id, annotator, label])


def shuffled_order(segments: list[dict[str, str]], annotator: str, seed: int, shuffle: bool = True):
    """Deterministic per-annotator order (a str seed is hashed deterministically)."""
    ordered = list(segments)
    if shuffle:
        random.Random(f"{seed}:{annotator}").shuffle(ordered)
    return ordered


def _default_play(y, sr) -> None:  # pragma: no cover - needs an audio device
    import sounddevice as sd

    sd.stop()
    sd.play(y, sr)
    try:
        sd.wait()
    except KeyboardInterrupt:
        sd.stop()


class SongAudioLoader:
    """Loads song slices on demand (one song cached at a time)."""

    def __init__(self, songs_dir: Path, manifest: Path, sr: int = SAMPLE_RATE):
        from .build_dataset import read_manifest

        self.songs_dir = Path(songs_dir)
        self.files = {r["song_id"]: r["file"] for r in read_manifest(manifest)}
        self.sr = sr
        self._cache: tuple[str, object] | None = None

    def get(self, song_id: str, start_s: float, end_s: float):
        from .features import load_audio

        if song_id not in self.files:
            raise KeyError(f"song_id '{song_id}' is not in the manifest")
        if self._cache is None or self._cache[0] != song_id:
            y, _ = load_audio(self.songs_dir / self.files[song_id], sr=self.sr)
            self._cache = (song_id, y)
        y = self._cache[1]
        return y[int(round(start_s * self.sr)) : int(round(end_s * self.sr))], self.sr


def annotate(
    segments: list[dict[str, str]],
    annotator: str,
    out: Path,
    loader: SongAudioLoader | None = None,
    play_fn=_default_play,
    input_fn=input,
    print_fn=print,
    seed: int = 42,
    shuffle: bool = True,
) -> dict[str, str]:
    """Run the interactive loop; returns the labels of this annotator (all sessions).

    ``loader`` is None for ``--no-audio`` mode. Labels are saved after every
    answer, so quitting (or a crash) never loses work, and already-labeled
    segments are skipped on the next run (resume).
    """
    labels = read_labels(out)
    pending = [s for s in shuffled_order(segments, annotator, seed, shuffle) if s["segment_id"] not in labels]
    print_fn(CRITERIA)
    print_fn(f"Anotador: {annotator} | ya etiquetados: {len(labels)} | pendientes: {len(pending)}")

    def play(seg) -> None:
        if loader is None:
            return
        y, sr = loader.get(seg["song_id"], float(seg["start_s"]), float(seg["end_s"]))
        play_fn(y, sr)

    for i, seg in enumerate(pending, start=1):
        print_fn(f"\nFragmento {i}/{len(pending)}  ({seg['segment_id']})")
        play(seg)
        while True:
            try:
                answer = input_fn(PROMPT).strip().lower()
            except EOFError:
                answer = "q"
            if answer in KEY_TO_LABEL:
                labels[seg["segment_id"]] = KEY_TO_LABEL[answer]
                write_labels(out, annotator, labels)
                break
            if answer == "r":
                play(seg)
            elif answer == "s":
                break
            elif answer == "q":
                write_labels(out, annotator, labels)
                print_fn(f"Guardado en {out} ({len(labels)} etiquetas). Hasta luego.")
                return labels
            else:
                print_fn("Entrada no valida. Usa 1, 2, 3, r, s o q.")
    write_labels(out, annotator, labels)
    remaining = len([s for s in segments if s["segment_id"] not in labels])
    print_fn(f"\nTerminado. {len(labels)} etiquetas guardadas en {out}. Sin etiquetar: {remaining}.")
    return labels


def main(argv: list[str] | None = None, input_fn=input, play_fn=_default_play) -> int:
    p = argparse.ArgumentParser(description="Label segments by listening (blind, resumable).")
    p.add_argument("--annotator", required=True, help="your name (used as file name)")
    p.add_argument("--segments", default="data/dataset/segments.csv")
    p.add_argument("--songs", default="data/songs")
    p.add_argument("--manifest", default="data/songs/manifest.csv")
    p.add_argument("--out", default=None, help="default: data/labels/<annotator>.csv")
    p.add_argument("--no-audio", action="store_true", help="do not play audio (testing only)")
    p.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    name = args.annotator.strip()
    if not re.match(r"^[A-Za-z0-9_-]+$", name):
        print("ERROR: --annotator must match [A-Za-z0-9_-]+ (it becomes the file name)", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else Path("data/labels") / f"{name}.csv"
    try:
        segments = read_segments(Path(args.segments))
        loader = None if args.no_audio else SongAudioLoader(Path(args.songs), Path(args.manifest))
    except Exception as exc:  # clear operator-facing error
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    annotate(
        segments, name, out, loader=loader, play_fn=play_fn, input_fn=input_fn,
        seed=args.seed, shuffle=args.shuffle,
    )
    return 0


__all__ = ["STYLES", "annotate", "main", "read_labels", "read_segments"]

if __name__ == "__main__":
    raise SystemExit(main())
