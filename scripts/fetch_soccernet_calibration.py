"""
Fetch the SoccerNet calibration-2023 dataset into datasets/external/.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs import registry as R  # noqa: E402

SOURCE_NAME = "soccernet_calibration_2023"

DEFAULT_SPLITS = ("train", "valid", "test")


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> int:
    """Extract `zf` under `dest`, refusing any member that would escape it."""
    dest_resolved = dest.resolve()
    for member in zf.namelist():
        target = (dest / member).resolve()
        if dest_resolved not in target.parents and target != dest_resolved:
            raise RuntimeError(
                f"Refusing to extract '{member}': it resolves outside {dest}"
            )
    zf.extractall(dest)
    return len(zf.namelist())


def _extract_zip(zip_path: Path, out_root: Path, force: bool) -> Path:
    """Extract one split zip. Returns the directory holding its contents."""
    with zipfile.ZipFile(zip_path) as zf:
        tops = {Path(n).parts[0] for n in zf.namelist() if n.strip()}
        if len(tops) == 1:
            dest, content_dir = out_root, out_root / next(iter(tops))
        else:
            content_dir = dest = out_root / zip_path.stem

        if content_dir.exists() and not force:
            print(f"    {zip_path.name}: already extracted -> {content_dir} "
                  f"(use --force to redo)")
            return content_dir

        dest.mkdir(parents=True, exist_ok=True)
        n = _safe_extract(zf, dest)
        print(f"    {zip_path.name}: extracted {n:,} members -> {content_dir}")
    return content_dir


def _inventory(root: Path) -> dict[str, tuple[int, int]]:
    """Per-split (image count, annotation count), counted recursively."""
    out: dict[str, tuple[int, int]] = {}
    if not root.is_dir():
        return out
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        images = sum(1 for p in split_dir.rglob("*")
                     if p.suffix.lower() in R.IMAGE_SUFFIXES)
        annos = sum(1 for _ in split_dir.rglob("*.json"))
        if images or annos:
            out[split_dir.name] = (images, annos)
    return out


def _print_inventory(root: Path) -> int:
    counts = _inventory(root)
    if not counts:
        print(f"  nothing found under {root}")
        return 0
    total_img = total_ann = 0
    for split, (img, ann) in counts.items():
        print(f"  {split:12s} images={img:6,d}  annotations={ann:6,d}")
        total_img += img
        total_ann += ann
    print(f"  {'TOTAL':12s} images={total_img:6,d}  annotations={total_ann:6,d}")
    if total_ann and total_img != total_ann:
        print("  NOTE: image and annotation counts differ -- expected only "
              "for the 'challenge' split.")
    return total_img


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS),
                    help=f"splits to download (default: {' '.join(DEFAULT_SPLITS)}; "
                         f"'challenge' has no public annotations)")
    ap.add_argument("--dest", default=None,
                    help="download root (default: the path registered in "
                         "configs/datasets.yaml::external_sources)")
    ap.add_argument("--no-extract", action="store_true",
                    help="download the zips but leave them packed")
    ap.add_argument("--force", action="store_true",
                    help="re-extract splits that are already unpacked")
    ap.add_argument("--inventory", action="store_true",
                    help="only count what is already on disk; download nothing")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved plan and exit")
    args = ap.parse_args()

    spec = R.external_source(SOURCE_NAME)
    dest = Path(args.dest).expanduser() if args.dest else R.external_source_dir(SOURCE_NAME)
    task = spec["task"]

    known = set(spec.get("splits") or DEFAULT_SPLITS)
    unknown = [s for s in args.splits if s not in known]
    if unknown:
        print(f"ERROR: unknown split(s) {unknown}. "
              f"configs/datasets.yaml declares: {sorted(known)}")
        return 2

    print(f"source   : {SOURCE_NAME}")
    print(f"task     : {task}")
    print(f"splits   : {', '.join(args.splits)}")
    print(f"dest     : {dest}\n")

    if args.inventory:
        return 0 if _print_inventory(dest) else 1

    if args.dry_run:
        print("dry run -- would call:")
        print(f"    SoccerNetDownloader(LocalDirectory={str(dest)!r})"
              f".downloadDataTask(task={task!r}, split={args.splits!r})")
        print("then extract every downloaded .zip and inventory the result.")
        return 0

    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError:
        print("ERROR: the 'SoccerNet' package is not installed -- it is the "
              "only published way to fetch this dataset.\n"
              "    pip install SoccerNet\n"
              "Nothing was downloaded.")
        return 2

    dest.mkdir(parents=True, exist_ok=True)
    downloader = SoccerNetDownloader(LocalDirectory=str(dest))
    print(f"[fetch] downloading {task} ({', '.join(args.splits)}) -- this is "
          f"several GB and takes a while ...", flush=True)
    try:
        downloader.downloadDataTask(task=task, split=list(args.splits))
    except Exception as exc:  # noqa: BLE001 -- surface the mirror's own error
        print(f"\nERROR: SoccerNet download failed: {type(exc).__name__}: {exc}\n"
              f"If this mentions a password, the mirror has moved this task "
              f"behind the NDA gate -- request credentials at "
              f"{spec.get('homepage', 'https://www.soccer-net.org/data')} and "
              f"set downloader.password before downloadDataTask.")
        return 1

    zips = sorted(dest.rglob("*.zip"))
    print(f"\n[fetch] downloaded, {len(zips)} zip archive(s) present")

    if args.no_extract:
        for z in zips:
            print(f"    {z}  ({z.stat().st_size / 1e9:.2f} GB)")
        print("\n--no-extract given; stopping before extraction.")
        return 0

    print("[extract]")
    extracted_root = dest
    for z in zips:
        content = _extract_zip(z, z.parent, args.force)
        extracted_root = content.parent

    print("\n[inventory]")
    n = _print_inventory(extracted_root)

    print(f"\nRaw SoccerNet data is at: {extracted_root}")
    print("NOT trainable yet: annotations are pitch LINE polylines, while "
          "configs/datasets.yaml::datasets.calibration expects 32 indexed "
          "YOLO pose keypoints. A line->keypoint converter is the next step "
          "(see this file's module docstring).")
    return 0 if n else 1


if __name__ == "__main__":
    raise SystemExit(main())
