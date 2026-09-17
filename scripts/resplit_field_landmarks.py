"""
Re-split the field-landmark dataset by CLIP, not by frame.

WHY THIS EXISTS
    The raw dataset (datasets/field_landmarks/field_yolo_raw) ships a
    train/val split that mixes frames from all three source clips into both
    sides:

        train: clip0=90, clip1=92, clip2=80
        val:   clip0=29, clip1=19, clip2=32

    Every frame is named `frame_{clip_id}_{n}.jpg` and there are only THREE
    source clips. Frames from the same clip share a camera angle, a
    background, and in many cases near-identical pitch geometry, so a
    val frame from clip0 is a near-duplicate of some train frame from
    clip0. Validation metrics on that split measure memorisation of three
    camera poses, not generalisation to an unseen one -- and a landmark
    detector's entire job is to work on a camera it has never seen.

    This script regroups at the clip level: a whole clip goes to train or
    to val, never both.

WHAT IT DOES NOT DO
    It does not invent, augment, or drop data. Every raw image/label pair
    lands in exactly one split of the output, and the raw tree is copied
    from, never moved or modified.

HONEST LIMITATION OF A 3-CLIP DATASET
    With three clips of roughly equal size (119 / 111 / 112 frames), the
    smallest possible held-out clip is ~32% of the data, not the ~20-25%
    a frame-level split would give you. That is not a bug in this script;
    it is the cost of having only three camera angles. The script prints
    the achieved fraction so the gap is visible rather than assumed away.

    It also means val is a SINGLE camera angle. A good val score says "the
    model generalised to this one unseen angle", not "the model
    generalises". Two clips of training data is thin. Treat the resulting
    numbers as directional until more clips exist.

Run:
    python -m scripts.resplit_field_landmarks
    python -m scripts.resplit_field_landmarks --dry-run
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = REPO_ROOT / "datasets" / "field_landmarks" / "field_yolo_raw"
DEFAULT_OUT = REPO_ROOT / "datasets" / "field_landmarks" / "field_yolo_v2"

SPLITS = ("train", "val")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

# `frame_{clip_id}_{n}` -- clip_id is the group key, n is the frame index
# within that clip. Anchored so a stray file that merely contains "frame_"
# is rejected loudly rather than silently grouped under the wrong clip.
FRAME_RE = re.compile(r"^frame_(\d+)_(\d+)$")

#: Fraction of total frames we would LIKE the val split to be. With three
#: near-equal clips this is unreachable (see module docstring); the chosen
#: clip is the one landing closest to it.
TARGET_VAL_FRACTION = 0.225


class ResplitError(RuntimeError):
    """Raised for any condition that would produce a silently wrong split."""


def find_dataset_root(raw_root: Path) -> Path:
    """
    Locate the directory that directly contains `images/` and `labels/`.

    The raw drop nests the payload one level deeper than the handoff
    described (`field_yolo_raw/field_yolo/images/...`), so this descends at
    most one level rather than hard-coding either shape. It never guesses
    further than that -- an unrecognised layout raises.
    """
    if (raw_root / "images").is_dir() and (raw_root / "labels").is_dir():
        return raw_root

    children = [p for p in sorted(raw_root.iterdir()) if p.is_dir()] if raw_root.is_dir() else []
    for child in children:
        if (child / "images").is_dir() and (child / "labels").is_dir():
            return child

    raise ResplitError(
        f"No images/ + labels/ pair found under {raw_root} (or one level below). "
        f"Found: {[p.name for p in children] or 'nothing'}. "
        "Point --raw at the directory containing the YOLO tree."
    )


def clip_id_of(stem: str) -> int:
    match = FRAME_RE.match(stem)
    if match is None:
        raise ResplitError(
            f"Filename {stem!r} does not match the expected 'frame_{{clip}}_{{n}}' "
            "pattern. Clip grouping is the whole point of this script, so an "
            "ungroupable file is a hard error, not something to skip."
        )
    return int(match.group(1))


def count_label_instances(label_path: Path) -> int:
    """Number of labelled boxes in a YOLO label file (0 for missing/empty)."""
    if not label_path.is_file():
        return 0
    return sum(1 for line in label_path.read_text().splitlines() if line.strip())


def collect_frames(dataset_root: Path) -> dict[int, list[dict]]:
    """
    Walk both raw splits and group every image by clip id.

    Returns {clip_id: [{image, label, n_instances, source_split}, ...]}.
    """
    by_clip: dict[int, list[dict]] = defaultdict(list)
    seen_stems: dict[str, Path] = {}

    for split in SPLITS:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        if not image_dir.is_dir():
            raise ResplitError(f"Missing raw image directory: {image_dir}")

        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            stem = image_path.stem

            # The same stem appearing in both raw splits would mean the raw
            # data already double-counts a frame; copying both would leak it
            # across the new split too.
            if stem in seen_stems:
                raise ResplitError(
                    f"Duplicate frame stem {stem!r} in both {seen_stems[stem]} "
                    f"and {image_path}. Refusing to guess which one is real."
                )
            seen_stems[stem] = image_path

            label_path = label_dir / f"{stem}.txt"
            if not label_path.is_file():
                raise ResplitError(
                    f"Image {image_path} has no matching label at {label_path}. "
                    "An unlabelled image silently becomes a background frame "
                    "during training, so this is rejected rather than copied."
                )

            by_clip[clip_id_of(stem)].append({
                "image": image_path,
                "label": label_path,
                "n_instances": count_label_instances(label_path),
                "source_split": split,
            })

    if not by_clip:
        raise ResplitError(f"No images found under {dataset_root}/images/{{train,val}}")
    return dict(by_clip)


def choose_val_clip(by_clip: dict[int, list[dict]]) -> int:
    """
    Pick the single held-out clip: the one whose share of the dataset is
    closest to TARGET_VAL_FRACTION. Ties break toward the lower clip id so
    the split is reproducible across runs.
    """
    total = sum(len(frames) for frames in by_clip.values())
    return min(
        by_clip,
        key=lambda cid: (abs(len(by_clip[cid]) / total - TARGET_VAL_FRACTION), cid),
    )


def assign_splits(by_clip: dict[int, list[dict]]) -> dict[int, str]:
    """Map every clip id to exactly one split."""
    if len(by_clip) < 2:
        raise ResplitError(
            f"Need at least 2 clips to split by clip, found {len(by_clip)}: "
            f"{sorted(by_clip)}. A single-clip dataset cannot have a "
            "held-out camera angle at all."
        )
    val_clip = choose_val_clip(by_clip)
    return {cid: ("val" if cid == val_clip else "train") for cid in sorted(by_clip)}


def verify_assignment(by_clip: dict[int, list[dict]], assignment: dict[int, str]) -> None:
    """
    The loud-failure gate. Checks the two conditions that would make the
    output worthless while still looking like a valid dataset.
    """
    problems: list[str] = []

    for cid, _frames in by_clip.items():
        split = assignment.get(cid)
        if split is None:
            problems.append(f"clip {cid} was never assigned to a split")
            continue
        # Condition 2: a clip straddling both splits is the exact leak this
        # script exists to remove.
        if split not in SPLITS:
            problems.append(f"clip {cid} assigned to unknown split {split!r}")

    # Condition 1: a split with no labelled frames at all.
    for split in SPLITS:
        clips = [cid for cid, s in assignment.items() if s == split]
        if not clips:
            problems.append(f"split {split!r} received no clips at all")
            continue
        labelled = sum(
            1 for cid in clips for f in by_clip[cid] if f["n_instances"] > 0
        )
        if labelled == 0:
            problems.append(
                f"split {split!r} (clips {clips}) has zero frames with any "
                "labelled landmark"
            )

    # And per clip: a clip that contributes nothing but empty labels is a
    # data problem worth surfacing before training, not after.
    for cid, frames in sorted(by_clip.items()):
        if all(f["n_instances"] == 0 for f in frames):
            problems.append(f"clip {cid} has zero labelled instances across all {len(frames)} frames")

    if problems:
        raise ResplitError(
            "Refusing to write the re-split dataset:\n  - " + "\n  - ".join(problems)
        )


def copy_split(
    by_clip: dict[int, list[dict]], assignment: dict[int, str], out_root: Path
) -> dict[str, int]:
    """Copy (never move) each frame into out_root/{images,labels}/{split}."""
    for split in SPLITS:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    written = dict.fromkeys(SPLITS, 0)
    for cid, frames in sorted(by_clip.items()):
        split = assignment[cid]
        for frame in frames:
            image_dst = out_root / "images" / split / frame["image"].name
            label_dst = out_root / "labels" / split / frame["label"].name
            shutil.copy2(frame["image"], image_dst)
            shutil.copy2(frame["label"], label_dst)
            written[split] += 1
    return written


def verify_on_disk(out_root: Path, assignment: dict[int, str]) -> None:
    """
    Re-read what was actually written and confirm no clip appears in both
    splits. Checking the plan is not the same as checking the result.
    """
    on_disk: dict[int, set[str]] = defaultdict(set)
    for split in SPLITS:
        for image_path in sorted((out_root / "images" / split).iterdir()):
            if image_path.suffix.lower() in IMAGE_SUFFIXES:
                on_disk[clip_id_of(image_path.stem)].add(split)

    straddling = {cid: sorted(splits) for cid, splits in on_disk.items() if len(splits) > 1}
    if straddling:
        raise ResplitError(
            f"POST-WRITE LEAK CHECK FAILED -- clips present in both splits: {straddling}"
        )
    for cid, splits in on_disk.items():
        expected = assignment.get(cid)
        actual = next(iter(splits))
        if actual != expected:
            raise ResplitError(
                f"clip {cid} landed in {actual!r} but was assigned {expected!r}"
            )


def report(by_clip: dict[int, list[dict]], assignment: dict[int, str]) -> None:
    total = sum(len(f) for f in by_clip.values())

    print("=" * 72)
    print("RAW DATASET -- frames per clip, per ORIGINAL split (the leak)")
    print("=" * 72)
    print(f"  {'clip':>5}  {'train':>7}  {'val':>7}  {'total':>7}   leaked?")
    for cid, frames in sorted(by_clip.items()):
        n_train = sum(1 for f in frames if f["source_split"] == "train")
        n_val = sum(1 for f in frames if f["source_split"] == "val")
        leaked = "YES" if n_train > 0 and n_val > 0 else "no"
        print(f"  {cid:>5}  {n_train:>7}  {n_val:>7}  {len(frames):>7}   {leaked}")
    print(f"  {'ALL':>5}  "
          f"{sum(1 for fs in by_clip.values() for f in fs if f['source_split'] == 'train'):>7}  "
          f"{sum(1 for fs in by_clip.values() for f in fs if f['source_split'] == 'val'):>7}  "
          f"{total:>7}")

    print()
    print("=" * 72)
    print("NEW SPLIT -- whole clips held out")
    print("=" * 72)
    print(f"  {'clip':>5}  {'split':>7}  {'frames':>7}  {'labelled':>9}  {'instances':>10}")
    for cid, frames in sorted(by_clip.items()):
        labelled = sum(1 for f in frames if f["n_instances"] > 0)
        instances = sum(f["n_instances"] for f in frames)
        print(f"  {cid:>5}  {assignment[cid]:>7}  {len(frames):>7}  "
              f"{labelled:>9}  {instances:>10}")

    print()
    for split in SPLITS:
        clips = sorted(cid for cid, s in assignment.items() if s == split)
        n = sum(len(by_clip[cid]) for cid in clips)
        print(f"  {split:>5}: clips {clips} -> {n} frames ({n / total:.1%} of {total})")

    val_frac = sum(len(by_clip[cid]) for cid, s in assignment.items() if s == "val") / total
    if not (0.20 <= val_frac <= 0.25):
        print()
        print(f"  NOTE: val is {val_frac:.1%}, outside the ~20-25% target. With only "
              f"{len(by_clip)} clips of\n        near-equal size this is the closest "
              "achievable clip-level split. Frame-level\n        splitting could hit the "
              "target exactly but would reintroduce the leak.")
    print()
    print("  NOTE: val is a SINGLE camera angle. A good val number here means "
          "'generalised to\n        this one unseen clip', not 'generalises'. More "
          "clips are the only fix.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW,
                        help=f"raw dataset root (default: {DEFAULT_RAW})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"output dataset root (default: {DEFAULT_OUT})")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the split without copying any files")
    args = parser.parse_args(argv)

    if not args.raw.exists():
        raise ResplitError(
            f"Raw dataset not found at {args.raw}. Nothing was written. "
            "Place the field_yolo tree there (or pass --raw) and re-run."
        )

    dataset_root = find_dataset_root(args.raw)
    print(f"raw dataset root: {dataset_root}")
    print(f"output root:      {args.out}")
    print()

    by_clip = collect_frames(dataset_root)
    assignment = assign_splits(by_clip)
    verify_assignment(by_clip, assignment)

    report(by_clip, assignment)

    if args.dry_run:
        print()
        print("--dry-run: no files copied.")
        return 0

    print()
    written = copy_split(by_clip, assignment, args.out)
    verify_on_disk(args.out, assignment)
    print(f"copied: train={written['train']} val={written['val']} "
          f"(raw dataset untouched)")
    print("post-write leak check: PASS (no clip appears in both splits)")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ResplitError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
