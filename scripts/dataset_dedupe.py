"""
Deduplicating split manifest builder.

WHY THIS EXISTS
    Several registered datasets are assembled from multiple independently
    exported Roboflow projects (ball has 6 source dirs, player has 2).
    Each export did its OWN train/valid/test split, and the same source
    image appears in more than one of them -- so after merging, a file that
    one export put in `train` is byte-identical to a file another export
    put in `test`.

    Measured on the ball dataset: 1,912 files, 1,404 unique (26.6%
    redundancy), and 114 unique images present in more than one split.
    Training on that and reporting the resulting mAP would be reporting
    test-set memorisation. This is not a hypothetical -- see
    docs/dataset_audit/ball.md.

WHAT IT DOES
    Hashes every image (sha1 of file bytes), assigns each UNIQUE image to
    exactly one split, and writes train/val/test manifest .txt files
    listing absolute image paths. Ultralytics accepts a .txt manifest
    anywhere it accepts a directory, so this fixes both leakage and
    redundancy with no copying and NO MUTATION of the source datasets.

SPLIT PRIORITY
    test > valid > train. If an image appears in both train and test it is
    kept in TEST and dropped from train. Evaluation integrity is the thing
    being protected, so the eval split always wins; the training set gives
    up the sample. The alternative (keep it in train) would leave the
    evaluation split contaminated, which is the exact failure being fixed.

LONG PATHS
    Paths over 260 chars are written in extended-length form on Windows so
    the loader can open them (see scripts/dataset_qa.py::MAX_PATH -- the
    goalpost set contains a 298-char filename that a naive open() cannot
    read at all).

Usage
    python -m scripts.dataset_dedupe ball
    python -m scripts.dataset_dedupe            # every dataset in the registry
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

from configs import registry as R  # noqa: E402

SPLIT_PRIORITY = {"test": 0, "valid": 1, "train": 2}
MAX_PATH = 260


def _readable(p: Path) -> str:
    """Path form the loader can actually open (extended-length on Windows
    when over MAX_PATH)."""
    s = str(p)
    if sys.platform == "win32" and len(s) > MAX_PATH and not s.startswith("\\\\?\\"):
        return "\\\\?\\" + s
    return s


def build_manifests(name: str, out_dir: Path | None = None) -> dict:
    out_dir = out_dir or (R.REPO_ROOT / "runs" / "_data" / name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # digest -> (best_split, path)
    chosen: dict[str, tuple[str, Path]] = {}
    counts_raw: dict[str, int] = defaultdict(int)
    dropped_dup = 0
    dropped_leak = 0
    unreadable: list[str] = []

    for src in R.dataset_source_dirs(name):
        for split in ("train", "valid", "test"):
            img_dir = src / split / "images"
            if not img_dir.is_dir():
                continue
            for p in sorted(img_dir.iterdir()):
                if p.suffix.lower() not in R.IMAGE_SUFFIXES:
                    continue
                counts_raw[split] += 1
                try:
                    digest = hashlib.sha1(Path(_readable(p)).read_bytes()).hexdigest()
                except OSError as exc:
                    unreadable.append(f"{p} ({exc})")
                    continue
                prev = chosen.get(digest)
                if prev is None:
                    chosen[digest] = (split, p)
                    continue
                # Duplicate. Keep whichever split has higher priority.
                if SPLIT_PRIORITY[split] < SPLIT_PRIORITY[prev[0]]:
                    chosen[digest] = (split, p)
                if split != prev[0]:
                    dropped_leak += 1
                else:
                    dropped_dup += 1

    by_split: dict[str, list[Path]] = defaultdict(list)
    for split, path in chosen.values():
        by_split[split].append(path)

    written = {}
    # ultralytics maps the data.yaml key `val` to the on-disk split `valid`
    for split, key in (("train", "train"), ("valid", "val"), ("test", "test")):
        paths = sorted(by_split.get(split, []))
        manifest = out_dir / f"{key}.txt"
        manifest.write_text(
            "\n".join(_readable(p) for p in paths) + "\n", encoding="utf-8")
        written[key] = (manifest, len(paths))

    return {
        "name": name,
        "raw_counts": dict(counts_raw),
        "unique": len(chosen),
        "dropped_within_split": dropped_dup,
        "dropped_cross_split": dropped_leak,
        "unreadable": unreadable,
        "manifests": written,
        "out_dir": out_dir,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("datasets", nargs="*")
    args = ap.parse_args()
    names = args.datasets or list(R.datasets_config()["datasets"].keys())

    for name in names:
        r = build_manifests(name)
        raw = sum(r["raw_counts"].values())
        print(f"\n[{name}] {raw:,} files -> {r['unique']:,} unique")
        print(f"  dropped {r['dropped_within_split']:,} within-split duplicate(s)")
        print(f"  dropped {r['dropped_cross_split']:,} CROSS-SPLIT duplicate(s) (leakage)")
        for key, (path, n) in r["manifests"].items():
            print(f"  {key:5s}: {n:6,d} -> {path}")
        for u in r["unreadable"]:
            print(f"  UNREADABLE: {u}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
