"""
Fetch the SoccerNet calibration-2023 dataset into datasets/external/.

WHY THIS SCRIPT EXISTS
    SoccerNet is not a file you can wget. The frames and their line
    annotations are only published through the `SoccerNet` pip package,
    which resolves the current mirror at runtime:

        pip install SoccerNet
        from SoccerNet.Downloader import SoccerNetDownloader as SNdl
        SNdl(LocalDirectory=...).downloadDataTask(
            task="calibration-2023", split=["train", "valid", "test"])

    This wraps that call so the destination comes from the registry
    (configs/datasets.yaml::external_sources) instead of being retyped, and
    so the download is followed by extraction and an actual inventory of
    what landed on disk.

    The NDA + password gate on soccer-net.org applies to the raw full-match
    VIDEO archives. The calibration task ships still frames plus
    annotations and downloads without one; if that ever changes, the
    package raises and this script surfaces the error rather than leaving a
    half-empty directory behind.

WHAT YOU GET -- AND WHY IT IS NOT YET TRAINABLE
    Each split is <frame>.jpg plus <frame>.json, where the JSON maps pitch
    LINE names ("Big rect. left bottom", "Circle central", ...) to a
    polyline of normalised points. That is not the format anything in this
    repo trains on:

        configs/datasets.yaml::datasets.calibration  expects YOLO pose
        labels -- one row per image, 32 indexed keypoints (kpt_shape
        [32, 3]) -- and the existing manual flow uses a third scheme again
        (17 NAMED landmarks, constants.CALIBRATION_POINT_ORDER).

    Converting lines -> keypoints means intersecting the annotated
    polylines and deciding which intersection is which landmark. That
    converter does not exist yet, so this download is registered under
    `external_sources` (raw, documented) and NOT under `datasets` (verified
    and trainable). Registering it as trainable would make dataset_qa.py
    and every trainer fail on a directory that was never meant to satisfy
    the images/ + labels/ contract.

USAGE
    python -m scripts.fetch_soccernet_calibration                # train/valid/test
    python -m scripts.fetch_soccernet_calibration --dry-run      # show plan only
    python -m scripts.fetch_soccernet_calibration --splits train
    python -m scripts.fetch_soccernet_calibration --inventory    # no download
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from configs import registry as R  # noqa: E402

SOURCE_NAME = "soccernet_calibration_2023"

# The challenge split is deliberately NOT in the default set: its
# annotations are withheld for the SoccerNet competition, so it downloads
# images with nothing to learn from. Ask for it explicitly if you intend to
# submit predictions.
DEFAULT_SPLITS = ("train", "valid", "test")


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> int:
    """Extract `zf` under `dest`, refusing any member that would escape it.

    Zip archives can carry absolute paths or ../ components; Python's
    extractall sanitises those, but silently. We reject instead, so a
    tampered mirror is a visible failure rather than a quiet no-op.
    """
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
    """Extract one split zip. Returns the directory holding its contents.

    If every member already shares one top-level directory the archive is
    unpacked as-is; otherwise a directory named after the zip is created,
    so a flat archive does not spray thousands of frames into out_root.
    """
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
        # Expected for the challenge split (annotations withheld); a
        # mismatch anywhere else means a truncated download.
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
