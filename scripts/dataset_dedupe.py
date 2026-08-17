"""
Deduplicating split manifest builder.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
