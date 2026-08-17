"""
Per-class instance counts for the field-landmark dataset.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "datasets" / "field_landmarks" / "field_yolo_v2"

SPLITS = ("train", "val")

LOW_SAMPLE_THRESHOLD = 50


def read_data_yaml(dataset_root: Path) -> tuple[int | None, dict[int, str]]:
    """
    Pull `nc` and `names` out of data.yaml without requiring PyYAML.
    """
    path = dataset_root / "data.yaml"
    if not path.is_file():
        return None, {}

    nc: int | None = None
    names: dict[int, str] = {}
    in_names = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indented = line[:1].isspace()

        if not indented:
            in_names = False
            key, _, value = line.partition(":")
            key = key.strip()
            if key == "nc":
                try:
                    nc = int(value.strip())
                except ValueError:
                    nc = None
            elif key == "names":
                in_names = True
            continue

        if in_names:
            key, _, value = line.strip().partition(":")
            try:
                names[int(key.strip())] = value.strip()
            except ValueError:
                continue

    return nc, names


def count_instances(dataset_root: Path) -> tuple[dict[str, Counter], dict[str, int], list[str]]:
    """
    Returns (per-split class counters, per-split frame counts, warnings).
    """
    per_split: dict[str, Counter] = {split: Counter() for split in SPLITS}
    frames: dict[str, int] = {split: 0 for split in SPLITS}
    warnings: list[str] = []

    for split in SPLITS:
        label_dir = dataset_root / "labels" / split
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Missing label directory: {label_dir}")

        empty = 0
        for label_path in sorted(label_dir.glob("*.txt")):
            frames[split] += 1
            rows = [ln for ln in label_path.read_text().splitlines() if ln.strip()]
            if not rows:
                empty += 1
                continue
            for row in rows:
                try:
                    per_split[split][int(float(row.split()[0]))] += 1
                except (ValueError, IndexError):
                    warnings.append(f"{label_path.name}: unparseable row {row!r}")

        if empty:
            warnings.append(
                f"{split}: {empty} label file(s) contain no annotations "
                "(these train as background frames)")

    return per_split, frames, warnings


def landmarks_per_frame_histogram(dataset_root: Path) -> dict[str, Counter]:
    """
    How many landmarks are visible per frame, per split.
    """
    hist: dict[str, Counter] = {split: Counter() for split in SPLITS}
    for split in SPLITS:
        for label_path in sorted((dataset_root / "labels" / split).glob("*.txt")):
            n = len([ln for ln in label_path.read_text().splitlines() if ln.strip()])
            hist[split][n] += 1
    return hist


def report(dataset_root: Path) -> int:
    nc, names = read_data_yaml(dataset_root)
    per_split, frames, warnings = count_instances(dataset_root)

    observed = set(per_split["train"]) | set(per_split["val"])
    class_ids = sorted(range(nc)) if nc is not None else sorted(observed)

    print("=" * 78)
    print(f"FIELD-LANDMARK CLASS BALANCE -- {dataset_root}")
    print("=" * 78)
    print(f"  frames:    train={frames['train']}  val={frames['val']}  "
          f"total={frames['train'] + frames['val']}")
    print(f"  instances: train={sum(per_split['train'].values())}  "
          f"val={sum(per_split['val'].values())}  "
          f"total={sum(per_split['train'].values()) + sum(per_split['val'].values())}")
    print(f"  classes:   nc={nc if nc is not None else '(no data.yaml)'}  "
          f"observed in labels={len(observed)}")
    print(f"  LOW_SAMPLE threshold: < {LOW_SAMPLE_THRESHOLD} total instances")
    print()

    header = (f"  {'id':>3}  {'name':<14}  {'total':>7}  {'train':>7}  {'val':>7}  "
              f"{'val%':>6}  flags")
    print(header)
    print("  " + "-" * (len(header) - 2))

    low_sample: list[int] = []
    val_zero: list[int] = []
    train_zero: list[int] = []
    absent: list[int] = []

    for class_id in class_ids:
        n_train = per_split["train"][class_id]
        n_val = per_split["val"][class_id]
        total = n_train + n_val

        flags: list[str] = []
        if total == 0:
            flags.append("ABSENT")
            absent.append(class_id)
        else:
            if total < LOW_SAMPLE_THRESHOLD:
                flags.append("LOW_SAMPLE")
                low_sample.append(class_id)
            if n_val == 0:
                flags.append("VAL_ZERO")
                val_zero.append(class_id)
            if n_train == 0:
                flags.append("TRAIN_ZERO")
                train_zero.append(class_id)

        val_pct = f"{n_val / total:.0%}" if total else "-"
        name = names.get(class_id, f"class_{class_id}")
        print(f"  {class_id:>3}  {name:<14}  {total:>7}  {n_train:>7}  {n_val:>7}  "
              f"{val_pct:>6}  {' '.join(flags)}")

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    counts = {cid: per_split['train'][cid] + per_split['val'][cid] for cid in class_ids}
    non_zero = {cid: n for cid, n in counts.items() if n > 0}
    if non_zero:
        richest = max(non_zero, key=lambda c: non_zero[c])
        poorest = min(non_zero, key=lambda c: non_zero[c])
        print(f"  most instances:  class {richest} ({non_zero[richest]})")
        print(f"  least instances: class {poorest} ({non_zero[poorest]})")
        print(f"  imbalance ratio: {non_zero[richest] / non_zero[poorest]:.1f}x "
              "(most / least)")

    def fmt(ids: list[int]) -> str:
        return ", ".join(str(i) for i in ids) if ids else "none"

    print(f"  LOW_SAMPLE (< {LOW_SAMPLE_THRESHOLD} total): {fmt(low_sample)}")
    print(f"  VAL_ZERO   (unmeasurable):    {fmt(val_zero)}")
    print(f"  TRAIN_ZERO (unlearnable):     {fmt(train_zero)}")
    print(f"  ABSENT     (no instances):    {fmt(absent)}")

    print()
    print("  landmarks visible per frame (a homography needs >= 4):")
    hist = landmarks_per_frame_histogram(dataset_root)
    for split in SPLITS:
        under = sum(n for k, n in hist[split].items() if k < 4)
        total_frames = sum(hist[split].values())
        counts_seen = sorted(hist[split])
        span = f"{counts_seen[0]}-{counts_seen[-1]}" if counts_seen else "-"
        print(f"    {split:>5}: range {span} per frame, "
              f"{under}/{total_frames} frames have < 4 (uncalibratable alone)")

    if warnings:
        print()
        print("  warnings:")
        for warning in warnings:
            print(f"    - {warning}")

    print()
    print("  This report does not change the dataset. Deciding whether the "
          "flagged classes\n  need targeted re-labelling is a data-collection "
          "call, not a code change.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help=f"dataset root containing labels/ (default: {DEFAULT_DATASET})")
    args = parser.parse_args(argv)

    if not args.dataset.is_dir():
        print(f"ERROR: dataset not found at {args.dataset}", file=sys.stderr)
        return 1
    return report(args.dataset)


if __name__ == "__main__":
    sys.exit(main())
