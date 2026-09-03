"""
Convert external pitch-keypoint datasets into OUR 32-index YOLO-pose scheme.

Currently one source: the SoccerNet field-keypoint mirror
``nreHieW/SoccerNet_Field_Keypoints`` on HuggingFace (parquet: an image plus
57 pixel keypoints per frame, 16k/3.1k/3.1k train/val/test).

Those 57 slots are the "No Bells Just Whistles" / PnLCalib scheme, whose
world coordinates are published on a 105 m x 68 m pitch -- the same pitch
model and the same origin convention we use. So the index mapping is not
guessed: SOCCERNET_TO_OURS is DERIVED at import time by nearest-coordinate
matching against PITCH_KEYPOINTS_32, and the derivation refuses to produce a
mapping if any of our 32 points lacks a SoccerNet slot within
MATCH_TOLERANCE_M. A wrong order would silently yield plausible-looking
garbage homographies, so this stays a computation, never a hand-typed list.

    python -m scripts.convert_pitch_kpts --out-name soccernet_kpts_32
"""

from __future__ import annotations

import argparse
import io
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.computer_vision.tactical_analysis.pitch_keypoints import (  # noqa: E402
    PITCH_KEYPOINTS_32,
)
from configs import registry as R  # noqa: E402

HF_DATASET = "nreHieW/SoccerNet_Field_Keypoints"

#: The mirror's 57 slots, as pitch-metre coordinates IN OUR FRAME.
#:
#: Provenance: the mirror is produced by https://github.com/nreHieW/Eagle ::
#: eagle/utils/pitch.py, whose INTERSECTION_TO_PITCH_POINTS names the slots
#: and GROUND_TRUTH_POINTS gives them coordinates on a 105 m x 68 m pitch
#: (fetched 2026-08-30). Do NOT substitute the "No Bells Just Whistles"
#: keypoint_world_coords_2D list -- it is a DIFFERENT 57-slot ordering, and
#: fitting a homography through it leaves a ~26 m residual on this data.
#:
#: One axis change is applied: Eagle's y grows toward the BOTTOM touchline
#: (its "TL_PITCH_CORNER" is y=68), ours grows toward the NEAR touchline
#: (our index 0, left_corner_far, is y=0). Measured on both datasets, the
#: point at Eagle y=0 sits LOWER in the image while our y=0 point sits
#: HIGHER, so the two conventions are mirrored and every y becomes 68 - y.
#: x agrees (x=0 is the left of frame in both). Without this flip the two
#: training sources would label visually identical frames inconsistently.
#:
#: Slots 0, 1, 24 and 25 are crossbar-height goal posts (z = -2.44), not on
#: the pitch plane; NOT_ON_PLANE_SLOTS keeps them out of the matching.
SOCCERNET_57_WORLD_M: list[tuple[float, float]] = [
    (0.0, 37.66),  #  0 L_GOAL_TL_POST   # NOT on the pitch plane (z=-2.44)
    (0.0, 30.34),  #  1 L_GOAL_TR_POST   # NOT on the pitch plane (z=-2.44)
    (0.0, 37.66),  #  2 L_GOAL_BL_POST
    (0.0, 30.34),  #  3 L_GOAL_BR_POST
    (5.5, 43.16),  #  4 L_GOAL_AREA_BR_CORNER
    (5.5, 24.84),  #  5 L_GOAL_AREA_TR_CORNER
    (0.0, 43.16),  #  6 L_GOAL_AREA_BL_CORNER
    (0.0, 24.84),  #  7 L_GOAL_AREA_TL_CORNER
    (16.5, 54.16),  #  8 L_PENALTY_AREA_BR_CORNER
    (16.5, 13.84),  #  9 L_PENALTY_AREA_TR_CORNER
    (0.0, 54.16),  # 10 L_PENALTY_AREA_BL_CORNER
    (0.0, 13.84),  # 11 L_PENALTY_AREA_TL_CORNER
    (0.0, 68.0),  # 12 BL_PITCH_CORNER
    (0.0, 0.0),  # 13 TL_PITCH_CORNER
    (52.5, 68.0),  # 14 B_TOUCH_AND_HALFWAY_LINES_INTERSECTION
    (52.5, 0.0),  # 15 T_TOUCH_AND_HALFWAY_LINES_INTERSECTION
    (88.5, 54.16),  # 16 R_PENALTY_AREA_BL_CORNER
    (88.5, 13.84),  # 17 R_PENALTY_AREA_TL_CORNER
    (105.0, 54.16),  # 18 R_PENALTY_AREA_BR_CORNER
    (105.0, 13.84),  # 19 R_PENALTY_AREA_TR_CORNER
    (99.5, 43.16),  # 20 R_GOAL_AREA_BL_CORNER
    (99.5, 24.84),  # 21 R_GOAL_AREA_TL_CORNER
    (105.0, 43.16),  # 22 R_GOAL_AREA_BR_CORNER
    (105.0, 24.84),  # 23 R_GOAL_AREA_TR_CORNER
    (105.0, 30.34),  # 24 R_GOAL_TL_POST   # NOT on the pitch plane (z=-2.44)
    (105.0, 37.66),  # 25 R_GOAL_TR_POST   # NOT on the pitch plane (z=-2.44)
    (105.0, 30.34),  # 26 R_GOAL_BL_POST
    (105.0, 37.66),  # 27 R_GOAL_BR_POST
    (105.0, 68.0),  # 28 BR_PITCH_CORNER
    (105.0, 0.0),  # 29 TR_PITCH_CORNER
    (61.312432, 31.537574),  # 30 CENTER_CIRCLE_TANGENT_TR
    (43.687568, 31.537574),  # 31 CENTER_CIRCLE_TANGENT_TL
    (61.312432, 36.462426),  # 32 CENTER_CIRCLE_TANGENT_BR
    (43.687568, 36.462426),  # 33 CENTER_CIRCLE_TANGENT_BL
    (58.970027, 27.529973),  # 34 CENTER_CIRCLE_TR
    (46.029973, 27.529973),  # 35 CENTER_CIRCLE_TL
    (58.970027, 40.470027),  # 36 CENTER_CIRCLE_BR
    (46.029973, 40.470027),  # 37 CENTER_CIRCLE_BL
    (61.65, 34.0),  # 38 CENTER_CIRCLE_R
    (43.35, 34.0),  # 39 CENTER_CIRCLE_L
    (52.5, 24.85),  # 40 T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION
    (52.5, 43.15),  # 41 B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION
    (52.5, 34.0),  # 42 CENTER_MARK
    (20.15, 34.0),  # 43 LEFT_CIRCLE_R
    (16.5, 41.312489),  # 44 BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION
    (16.5, 26.687511),  # 45 TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION
    (19.990673, 32.299911),  # 46 LEFT_CIRCLE_TANGENT_T
    (19.990673, 35.700089),  # 47 LEFT_CIRCLE_TANGENT_B
    (11.0, 34.0),  # 48 L_PENALTY_MARK
    (16.5, 34.0),  # 49 L_MIDDLE_PENALTY
    (84.85, 34.0),  # 50 RIGHT_CIRCLE_L
    (88.5, 41.312489),  # 51 BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION
    (88.5, 26.687511),  # 52 TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION
    (85.009327, 32.299911),  # 53 RIGHT_CIRCLE_TANGENT_T
    (85.009327, 35.700089),  # 54 RIGHT_CIRCLE_TANGENT_B
    (94.0, 34.0),  # 55 R_PENALTY_MARK
    (88.5, 34.0),  # 56 R_MIDDLE_PENALTY
]

#: Goal-post slots at crossbar height -- excluded from landmark matching.
NOT_ON_PLANE_SLOTS = {0, 1, 24, 25}

#: How far a SoccerNet slot may sit from our point and still be called the
#: same landmark. Every one of our 32 currently matches to within 0.01 m,
#: so this is a guard against upstream drift, not a fudge factor.
MATCH_TOLERANCE_M = 0.5

#: A frame with fewer landmarks than this cannot produce a homography
#: (MIN_CALIBRATION_POINTS is 4) and only adds label noise.
MIN_VISIBLE = 6


def derive_mapping(
    world: list[tuple[float, float]] = SOCCERNET_57_WORLD_M,
    tolerance_m: float = MATCH_TOLERANCE_M,
) -> dict[int, int]:
    """
    our index -> SoccerNet slot, by nearest world coordinate.

    Raises ValueError if any of our 32 points has no SoccerNet slot within
    ``tolerance_m``, or if two of ours would claim the same slot.
    """
    mapping: dict[int, int] = {}
    for ours, (ox, oy) in sorted(PITCH_KEYPOINTS_32.items()):
        best, best_d = None, float("inf")
        for slot, (sx, sy) in enumerate(world):
            if slot in NOT_ON_PLANE_SLOTS:
                continue
            d = math.hypot(ox - sx, oy - sy)
            if d < best_d:
                best, best_d = slot, d
        if best is None or best_d > tolerance_m:
            raise ValueError(
                f"our keypoint {ours} at ({ox:.2f}, {oy:.2f}) has no SoccerNet "
                f"slot within {tolerance_m} m (nearest was slot {best} at "
                f"{best_d:.3f} m)")
        mapping[ours] = best
    if len(set(mapping.values())) != len(mapping):
        raise ValueError(f"mapping is not injective: {mapping}")
    return mapping


#: our index -> SoccerNet 57-scheme slot.
SOCCERNET_TO_OURS: dict[int, int] = derive_mapping()


def convert_row(
    keypoints: list, width: int, height: int,
    mapping: dict[int, int] | None = None,
    margin: float = 0.02,
) -> tuple[str, int] | None:
    """
    One SoccerNet row -> one YOLO-pose label line, in OUR index order.

    ``keypoints`` is the raw 57-long list; entries are ``[x_px, y_px]`` or
    None. Returns (label_line, n_visible), or None when too few landmarks
    survive to be worth training on. A point is marked visible (v=2) only
    when it actually lands inside the frame -- upstream extrapolates some
    landmarks well outside the image, and those would teach the model to
    hallucinate off-screen points.
    """
    mapping = SOCCERNET_TO_OURS if mapping is None else mapping
    xs: list[float] = []
    ys: list[float] = []
    flat: list[str] = []
    n_visible = 0
    for ours in range(32):
        slot = mapping[ours]
        kp = keypoints[slot] if slot < len(keypoints) else None
        if kp is None or len(kp) < 2:
            flat.append("0 0 0")
            continue
        x, y = float(kp[0]) / width, float(kp[1]) / height
        if not (-margin <= x <= 1 + margin and -margin <= y <= 1 + margin):
            flat.append("0 0 0")
            continue
        x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
        xs.append(x)
        ys.append(y)
        flat.append(f"{x:.6f} {y:.6f} 2")
        n_visible += 1

    if n_visible < MIN_VISIBLE:
        return None

    # Same box convention as the existing field_datasets/2 labels: the
    # extent of the visible landmarks, not the whole frame.
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " + " ".join(flat), n_visible


def _iter_parquet(path: Path):
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
        yield from batch.to_pylist()


def convert_split(shards: list[Path], out_dir: Path, limit: int | None,
                  stride: int, square: int | None = None) -> dict:
    """
    ``square`` stretches every image to square x square. The existing
    calibration source (field_datasets/2) is a 960x960 square-stretched
    Roboflow export, and inference preprocesses frames the same way
    (models.yaml :: calibration.inference.preprocess = stretch_square), so
    SoccerNet's native 16:9 frames must be put in the same convention or the
    model sees two different pitch aspect ratios. Normalised keypoints are
    invariant under the stretch, so only the pixels change.
    """
    from PIL import Image

    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    kept = skipped_bad = skipped_sparse = 0
    seen = 0
    visible_total = 0
    for shard in shards:
        for row in _iter_parquet(shard):
            seen += 1
            if stride > 1 and seen % stride:
                continue
            if row.get("is_bad"):
                skipped_bad += 1
                continue
            img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            out = convert_row(row["keypoints"], img.width, img.height)
            if out is None:
                skipped_sparse += 1
                continue
            line, n_vis = out
            stem = f"sn_{shard.stem.replace('-of-', '_')}_{row['id']:07d}"
            if square:
                img = img.resize((square, square), Image.BILINEAR)
            img.save(out_dir / "images" / f"{stem}.jpg", quality=92)
            (out_dir / "labels" / f"{stem}.txt").write_text(line + "\n",
                                                            encoding="utf-8")
            kept += 1
            visible_total += n_vis
            if limit and kept >= limit:
                break
        if limit and kept >= limit:
            break
    return {"kept": kept, "skipped_is_bad": skipped_bad,
            "skipped_too_few_keypoints": skipped_sparse,
            "mean_visible_keypoints": round(visible_total / max(kept, 1), 2)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-name", default="soccernet_kpts_32",
                    help="directory name under the processed dataset root")
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-valid", type=int, default=None)
    ap.add_argument("--limit-test", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth train frame (SoccerNet frames are dense)")
    ap.add_argument("--square", type=int, default=960,
                    help="stretch images to this square size; 0 keeps 16:9")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    plan = {
        "train": [f"data/train-0000{i}-of-00005.parquet" for i in range(5)],
        "valid": ["data/val-00000-of-00001.parquet"],
        "test": ["data/test-00000-of-00001.parquet"],
    }
    cache = {split: [Path(hf_hub_download(HF_DATASET, f, repo_type="dataset"))
                     for f in files]
             for split, files in plan.items()}

    root = R.dataset_root() / args.out_name
    print(f"mapping (our index -> soccernet slot): {SOCCERNET_TO_OURS}")
    for split, limit in (("train", args.limit_train),
                         ("valid", args.limit_valid),
                         ("test", args.limit_test)):
        stats = convert_split(cache[split], root / split, limit,
                              args.stride if split == "train" else 1,
                              square=args.square or None)
        print(f"  {split:6s} {stats}", flush=True)
    print(f"\nwritten to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
