"""Merge the annotators' labels: majority vote + inter-annotator agreement.

Usage::

    python -m dance.audio.merge_labels --labels-dir data/labels \\
        --out data/dataset/labels.csv --metrics-dir report/metrics

* Majority vote per segment. Ties are excluded from ``labels.csv`` and listed
  in ``<metrics-dir>/annotation_ties.csv``.
* ``agreement`` = fraction of the segment's annotators that voted for the
  majority label.
* ``annotation_agreement.json`` holds Fleiss' kappa (all annotators, on the
  segments labeled by the maximum number of annotators) and per-pair Cohen's
  kappa (scikit-learn).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

from dance import STYLES


def fleiss_kappa(counts: np.ndarray) -> float:
    """Fleiss' kappa from an ``N x k`` matrix of per-item category counts.

    Every item must have been rated by the same number of raters. Returns NaN
    when chance agreement is 1 (kappa undefined).
    """
    counts = np.asarray(counts, dtype=float)
    n_items, _ = counts.shape
    n_raters = counts.sum(axis=1)
    if not np.all(n_raters == n_raters[0]):
        raise ValueError("Fleiss' kappa needs the same number of raters for every item")
    n = n_raters[0]
    if n < 2:
        raise ValueError("Fleiss' kappa needs at least 2 raters")
    p_cat = counts.sum(axis=0) / (n_items * n)
    p_item = ((counts**2).sum(axis=1) - n) / (n * (n - 1))
    p_bar = p_item.mean()
    p_exp = float((p_cat**2).sum())
    if np.isclose(p_exp, 1.0):
        return float("nan")
    return float((p_bar - p_exp) / (1 - p_exp))


def interpret_kappa(kappa: float | None) -> str:
    """Landis & Koch (1977) verbal scale."""
    if kappa is None:
        return "undefined"
    for upper, text in [(0.0, "poor"), (0.20, "slight"), (0.40, "fair"), (0.60, "moderate"), (0.80, "substantial")]:
        if kappa <= upper:
            return text
    return "almost perfect"


def load_label_files(labels_dir: Path) -> dict[str, dict[str, str]]:
    """Return ``{annotator: {segment_id: label}}`` from every ``*.csv`` in the dir."""
    labels_dir = Path(labels_dir)
    files = sorted(labels_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no label files (*.csv) found in {labels_dir}")
    result: dict[str, dict[str, str]] = {}
    for f in files:
        with open(f, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                label = row["label"].strip().upper()
                if label not in STYLES:
                    raise ValueError(f"{f.name}: invalid label '{row['label']}' for {row['segment_id']}")
                annotator = row.get("annotator") or f.stem
                result.setdefault(annotator, {})[row["segment_id"]] = label
    return result


def merge(by_annotator: dict[str, dict[str, str]]):
    """Majority vote. Returns ``(merged_rows, tie_rows, votes)``."""
    votes: dict[str, dict[str, str]] = {}
    for annotator, labs in by_annotator.items():
        for seg_id, label in labs.items():
            votes.setdefault(seg_id, {})[annotator] = label
    merged, ties = [], []
    for seg_id in sorted(votes):
        counter = Counter(votes[seg_id].values())
        n = sum(counter.values())
        top = counter.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            row = {"segment_id": seg_id, "n_annotators": n}
            row.update({s: counter.get(s, 0) for s in STYLES})
            row["votes"] = ";".join(f"{a}={l}" for a, l in sorted(votes[seg_id].items()))
            ties.append(row)
            continue
        label, count = top[0]
        merged.append(
            {"segment_id": seg_id, "label": label, "agreement": round(count / n, 4), "n_annotators": n}
        )
    return merged, ties, votes


def agreement_report(by_annotator: dict[str, dict[str, str]], votes: dict[str, dict[str, str]]) -> dict:
    annotators = sorted(by_annotator)
    max_n = max((len(v) for v in votes.values()), default=0)
    report: dict = {
        "annotators": annotators,
        "n_segments": len(votes),
        "n_segments_by_n_annotators": dict(sorted(Counter(len(v) for v in votes.values()).items())),
    }
    full = [v for v in votes.values() if len(v) == max_n]
    if max_n >= 2 and full:
        matrix = np.array([[list(v.values()).count(s) for s in STYLES] for v in full])
        kappa = fleiss_kappa(matrix)
        kappa_val = None if np.isnan(kappa) else round(kappa, 4)
        report.update(
            fleiss_kappa=kappa_val,
            fleiss_n_items=len(full),
            fleiss_n_raters=max_n,
            fleiss_interpretation=interpret_kappa(kappa_val),
        )
    else:
        report.update(fleiss_kappa=None, fleiss_n_items=0, fleiss_n_raters=max_n, fleiss_interpretation="undefined")
    pairs = []
    for a, b in itertools.combinations(annotators, 2):
        common = sorted(set(by_annotator[a]) & set(by_annotator[b]))
        entry = {"annotator_a": a, "annotator_b": b, "n_common": len(common), "cohen_kappa": None,
                 "percent_agreement": None}
        if common:
            ya = [by_annotator[a][s] for s in common]
            yb = [by_annotator[b][s] for s in common]
            k = cohen_kappa_score(ya, yb, labels=STYLES)
            entry["cohen_kappa"] = None if np.isnan(k) else round(float(k), 4)
            entry["percent_agreement"] = round(float(np.mean([x == y for x, y in zip(ya, yb)])), 4)
        pairs.append(entry)
    report["pairwise_cohen"] = pairs
    return report


def run(labels_dir: Path, out: Path, metrics_dir: Path, log=print) -> dict:
    by_annotator = load_label_files(labels_dir)
    merged, ties, votes = merge(by_annotator)
    few = [s for s, v in votes.items() if len(v) < 2]
    if few:
        log(f"WARNING: {len(few)} segment(s) have fewer than 2 annotators (e.g. {', '.join(sorted(few)[:5])})")

    out, metrics_dir = Path(out), Path(metrics_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["segment_id", "label", "agreement", "n_annotators"])
        writer.writeheader()
        writer.writerows(merged)
    with open(metrics_dir / "annotation_ties.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["segment_id", "n_annotators"] + STYLES + ["votes"])
        writer.writeheader()
        writer.writerows(ties)
    report = agreement_report(by_annotator, votes)
    report["n_merged"] = len(merged)
    report["n_ties_excluded"] = len(ties)
    report["n_segments_fewer_than_2_annotators"] = len(few)
    report["label_distribution"] = dict(Counter(r["label"] for r in merged))
    report["mean_agreement_with_majority"] = (
        round(float(np.mean([r["agreement"] for r in merged])), 4) if merged else None
    )
    (metrics_dir / "annotation_agreement.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(
        f"Merged {len(merged)} segments ({len(ties)} ties excluded) from {len(by_annotator)} annotators; "
        f"Fleiss kappa = {report['fleiss_kappa']} ({report['fleiss_interpretation']})"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Majority vote and inter-annotator agreement.")
    p.add_argument("--labels-dir", default="data/labels")
    p.add_argument("--out", default="data/dataset/labels.csv")
    p.add_argument("--metrics-dir", default="report/metrics")
    args = p.parse_args(argv)
    try:
        run(Path(args.labels_dir), Path(args.out), Path(args.metrics_dir))
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
