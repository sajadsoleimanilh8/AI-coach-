"""
Single source of truth for dataset and model checkpoint locations.

WHAT THIS REPLACES
    Before this module, dataset and model paths were scattered as absolute
    Windows literals across the training/eval scripts --
    "ai/computer_vision/player detection/"'s merge.py (lines 6-9, 13),
    remap.py (line 9), test.py (lines 14-15), test_yolo.py (lines 35, 39)
    and extract_clip.py (line 24) each baked in D:\\SportsStrategyCoachAI\\...
    -- while inference read a single $YOLO_MODEL_PATH env var that pointed
    at one combined 4-class checkpoint. None of it was portable and none of
    it agreed with anything else.

    Everything now resolves through configs/datasets.yaml + configs/models.yaml
    via this module.

DESIGN RULES (do not weaken these)
    1. Resolution failure is LOUD. A missing dataset directory, a missing
       split, or a missing checkpoint raises with the exact resolved path
       that was tried. There is no silent fallback -- in particular a
       missing trained checkpoint must never degrade to a stock COCO
       checkpoint, which is what docker-compose.gpu.yml's
       `yolov8s.pt` placeholder did: it produced confident, meaningless
       detections that looked like the pipeline was working.
    2. verify_dataset() runs as a pre-flight in every trainer, BEFORE
       ultralytics is even imported, so a bad dataset fails in seconds with
       a readable message instead of thirty seconds into epoch 1.
    3. Resolved data.yaml files are GENERATED, never hand-edited. Every
       data.yaml shipped inside the dataset directories declares its splits
       as "../train/images", which resolves one level too high and does not
       exist (verified for all 11 dataset dirs). We never point ultralytics
       at those files; build_data_yaml() writes a corrected one with
       absolute paths.
    4. Multi-source datasets (ball has 6 source dirs, player has 2) are
       combined by listing every split directory in the generated
       data.yaml, which ultralytics accepts natively. Files are NOT copied
       -- the old merge.py duplicated every image on disk to achieve this.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# configs/registry.py -> configs/ -> repo root
CONFIG_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONFIG_DIR.parent

DATASETS_YAML = CONFIG_DIR / "datasets.yaml"
MODELS_YAML = CONFIG_DIR / "models.yaml"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class RegistryError(Exception):
    """Base for every registry failure. Callers (trainers, the pipeline
    runner) surface the message as-is -- each one is written to say what
    was looked for and where."""


class DatasetNotFoundError(RegistryError):
    pass


class DatasetStructureError(RegistryError):
    pass


class CheckpointNotFoundError(RegistryError):
    pass


# ----------------------------------------------------------------------
# Raw config loading
# ----------------------------------------------------------------------

_cache: dict[str, Any] = {}


def _load(path: Path) -> dict:
    key = str(path)
    if key not in _cache:
        if not path.exists():
            raise RegistryError(f"Registry config missing: {path}")
        with open(path, encoding="utf-8") as fh:
            _cache[key] = yaml.safe_load(fh) or {}
    return _cache[key]


def datasets_config() -> dict:
    return _load(DATASETS_YAML)


def models_config() -> dict:
    return _load(MODELS_YAML)


# ----------------------------------------------------------------------
# Root resolution
# ----------------------------------------------------------------------

def dataset_root() -> Path:
    """
    Resolves the dataset root. Order (first hit wins):
        1. $SSC_DATASET_ROOT
        2. $DATASET_ROOT
        3. datasets.yaml::root_default_windows   (on Windows)
        4. datasets.yaml::root_default_posix     (elsewhere)

    Does NOT check existence -- verify_dataset() does that, so the error
    can name the specific dataset that failed rather than the root alone.
    """
    cfg = datasets_config()
    for env in ("SSC_DATASET_ROOT", "DATASET_ROOT"):
        val = os.getenv(env)
        if val:
            return Path(val).expanduser()
    key = "root_default_windows" if os.name == "nt" else "root_default_posix"
    default = cfg.get(key)
    if not default:
        raise RegistryError(
            f"datasets.yaml has no '{key}' and neither $SSC_DATASET_ROOT nor "
            f"$DATASET_ROOT is set -- cannot resolve the dataset root."
        )
    return Path(default).expanduser()


def external_root() -> Path:
    """Root for raw third-party downloads (SoccerNet et al).

    $SSC_EXTERNAL_ROOT overrides; the default is the `external` sibling of
    the processed/ root, which is where the existing Roboflow exports were
    already unpacked.
    """
    val = os.getenv("SSC_EXTERNAL_ROOT")
    if val:
        return Path(val).expanduser()
    return dataset_root().parent / "external"


def external_sources_config() -> dict:
    return datasets_config().get("external_sources", {}) or {}


def external_source(name: str) -> dict:
    """Spec for one raw external download.

    Kept separate from dataset_spec() on purpose: these have not been
    converted to the images/ + labels/ contract, so they must not appear to
    anything that iterates the trainable datasets.
    """
    sources = external_sources_config()
    if name not in sources:
        raise RegistryError(
            f"Unknown external source '{name}'. Declared in datasets.yaml "
            f"under external_sources: {sorted(sources)}"
        )
    return sources[name]


def external_source_dir(name: str) -> Path:
    """Absolute download directory for one external source. Existence is
    NOT checked -- the fetch script creates it."""
    spec = external_source(name)
    return external_root() / spec.get("dir", name)


def model_root() -> Path:
    """Root for trained checkpoints. $SSC_MODEL_ROOT overrides; default is
    <repo>/models (matches the existing models/yolo/ layout and the
    Docker image's /app/models mount)."""
    val = os.getenv("SSC_MODEL_ROOT")
    if val:
        return Path(val).expanduser()
    return REPO_ROOT / models_config().get("model_root_default", "models")


def runs_root() -> Path:
    val = os.getenv("SSC_RUNS_ROOT")
    if val:
        return Path(val).expanduser()
    return REPO_ROOT / models_config().get("runs_root_default", "runs/train")


# ----------------------------------------------------------------------
# Dataset verification
# ----------------------------------------------------------------------

@dataclass
class SplitReport:
    name: str
    images_dir: Path
    labels_dir: Path
    n_images: int
    n_labels: int
    problems: list[str] = field(default_factory=list)


@dataclass
class DatasetReport:
    name: str
    task: str
    root: Path
    sources: list[Path]
    splits: list[SplitReport]
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems and all(not s.problems for s in self.splits)

    @property
    def n_images(self) -> int:
        return sum(s.n_images for s in self.splits)


def dataset_spec(name: str) -> dict:
    cfg = datasets_config()
    specs = cfg.get("datasets", {})
    if name not in specs:
        raise RegistryError(
            f"Unknown dataset '{name}'. Declared in datasets.yaml: "
            f"{sorted(specs)}"
        )
    return specs[name]


def dataset_source_dirs(name: str) -> list[Path]:
    """Absolute paths of every source directory backing this dataset."""
    root = dataset_root()
    return [root / src for src in dataset_spec(name)["sources"]]


def verify_dataset(name: str, *, strict: bool = True) -> DatasetReport:
    """
    Pre-flight structural check for one registered dataset.

    Verifies, per source directory, that every required split exists, that
    it contains images/ and labels/ subdirectories, and that neither is
    empty. Prints/records the EXACT resolved absolute path being used, so a
    wrong $SSC_DATASET_ROOT is obvious rather than mysterious.

    Deliberately does NOT inspect label CONTENT -- that is the dataset QA
    pass (scripts/dataset_qa.py), which is slower and produces a full
    report. This is the fast gate every trainer runs first.

    Args:
        name: dataset key from datasets.yaml.
        strict: raise DatasetStructureError on any problem. Set False to
            get the report back for inspection (used by the QA script,
            which wants to report all datasets, not die on the first).
    """
    spec = dataset_spec(name)
    root = dataset_root()
    sources = dataset_source_dirs(name)
    required = datasets_config().get("required_splits", ["train", "valid", "test"])

    report = DatasetReport(
        name=name, task=spec.get("task", "detect"), root=root,
        sources=sources, splits=[],
    )

    if not root.exists():
        report.problems.append(
            f"Dataset root does not exist: {root}  "
            f"(set $SSC_DATASET_ROOT to override; see configs/datasets.yaml)"
        )

    for src in sources:
        if not src.exists():
            report.problems.append(f"Source directory missing: {src}")
            continue
        for split in required:
            images_dir = src / split / "images"
            labels_dir = src / split / "labels"
            sr = SplitReport(
                name=f"{src.name}/{split}", images_dir=images_dir,
                labels_dir=labels_dir, n_images=0, n_labels=0,
            )
            if not images_dir.is_dir():
                sr.problems.append(f"missing images dir: {images_dir}")
            else:
                sr.n_images = sum(
                    1 for p in images_dir.iterdir()
                    if p.suffix.lower() in IMAGE_SUFFIXES
                )
                if sr.n_images == 0:
                    sr.problems.append(f"no images in {images_dir}")
            if not labels_dir.is_dir():
                sr.problems.append(f"missing labels dir: {labels_dir}")
            else:
                sr.n_labels = sum(1 for p in labels_dir.glob("*.txt"))
                if sr.n_labels == 0:
                    sr.problems.append(f"no label files in {labels_dir}")
            report.splits.append(sr)

    if strict and not report.ok:
        raise DatasetStructureError(format_dataset_report(report))
    return report


def format_dataset_report(report: DatasetReport) -> str:
    lines = [
        f"Dataset '{report.name}' (task={report.task}) failed verification.",
        f"  resolved root: {report.root}",
    ]
    for src in report.sources:
        lines.append(f"  source: {src}")
    for p in report.problems:
        lines.append(f"  ERROR: {p}")
    for s in report.splits:
        if s.problems:
            for p in s.problems:
                lines.append(f"  ERROR [{s.name}]: {p}")
        else:
            lines.append(f"  ok [{s.name}]: {s.n_images} images / {s.n_labels} labels")
    lines.append(
        "  Fix the dataset layout or point $SSC_DATASET_ROOT at the correct "
        "root. Training was NOT started."
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Generated data.yaml
# ----------------------------------------------------------------------

def build_data_yaml(name: str, out_dir: Path | None = None,
                    dedupe: bool = True, refresh: bool = False) -> Path:
    """
    Writes a resolved, absolute-path data.yaml for `name` and returns it.

    This exists because every data.yaml shipped in the dataset directories
    declares its splits relative as "../train/images", which resolves one
    directory too high and does not exist. Rather than editing 11 vendored
    files in place (they get overwritten on any dataset re-export), we
    generate a correct one at train time.

    For multi-source datasets the split value is a LIST of absolute
    directories -- ultralytics resolves each independently, so the 6 ball
    sets and 2 player sets train together with no file copying.

    Pose datasets additionally carry kpt_shape and flip_idx. flip_idx is
    load-bearing: without it, fliplr augmentation mirrors the image but not
    the left/right-symmetric landmark identities, which silently trains the
    model to emit mirrored keypoints and corrupts every homography derived
    from them.
    """
    spec = dataset_spec(name)
    sources = dataset_source_dirs(name)
    out_dir = out_dir or (REPO_ROOT / "runs" / "_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    def split_dirs(split: str) -> Any:
        paths = [str((s / split / "images").resolve()) for s in sources]
        return paths[0] if len(paths) == 1 else paths

    if dedupe:
        # Deduplicated manifests. Required for correctness on any dataset
        # assembled from multiple exports: the 6 ball source dirs each did
        # their own train/valid/test split, putting byte-identical images
        # in train AND test (114 such images -- see
        # docs/dataset_audit/ball.md). Training on the raw directories and
        # reporting the resulting mAP would report memorisation.
        from scripts.dataset_dedupe import build_manifests

        manifest_dir = out_dir / name
        needed = [manifest_dir / f"{k}.txt" for k in ("train", "val", "test")]
        # A manifest holds ABSOLUTE image paths, so existing is not the same
        # as usable: after the dataset root moves, a cached manifest points
        # at paths that are all gone and ultralytics trains on an empty set
        # instead of failing. Rebuild whenever the cached paths no longer
        # live under the current root.
        stale = refresh or not all(p.exists() for p in needed)
        if not stale:
            root = str(dataset_root().resolve()).lower()
            for m in needed:
                first = next(
                    (ln.strip() for ln in m.read_text(encoding="utf-8").splitlines() if ln.strip()),
                    None,
                )
                if first is not None and not first.lower().startswith(root):
                    stale = True
                    break
        if stale:
            build_manifests(name, manifest_dir)
        split_value = {k: str(manifest_dir / f"{k}.txt") for k in ("train", "val", "test")}
    else:
        split_value = {
            "train": split_dirs("train"),
            "val": split_dirs("valid"),
            "test": split_dirs("test"),
        }

    doc: dict[str, Any] = {
        "train": split_value["train"],
        "val": split_value["val"],
        "test": split_value["test"],
        "nc": spec["nc"],
        "names": {int(k): v for k, v in spec["names"].items()},
    }
    if "kpt_shape" in spec:
        doc["kpt_shape"] = list(spec["kpt_shape"])
    if "flip_idx" in spec:
        doc["flip_idx"] = list(spec["flip_idx"])

    out_path = out_dir / f"{name}.data.yaml"
    header = (
        f"# GENERATED by configs/registry.py::build_data_yaml('{name}') --\n"
        f"# do not edit by hand, it is rewritten on every training run.\n"
        f"# Source dirs are listed absolutely because the vendored\n"
        f"# data.yaml files use '../train/images', which does not resolve.\n"
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(header)
        yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True)
    return out_path


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    task: str
    dataset: str
    base_weights: str
    checkpoint: Path
    metrics_file: Path
    run_name: str
    classes: dict
    train: dict
    inference: dict
    raw: dict

    def require_checkpoint(self) -> Path:
        """Returns the checkpoint path, raising if it is not on disk.

        Inference call sites use this instead of testing os.path.exists
        themselves so the error message is consistent and names the
        trainer that produces the file."""
        if not self.checkpoint.exists():
            raise CheckpointNotFoundError(
                f"Trained '{self.name}' checkpoint not found at:\n"
                f"    {self.checkpoint}\n"
                f"Train it first:  python -m training.train_{self.name}\n"
                f"Refusing to fall back to a stock/untrained checkpoint -- "
                f"that produces confident but meaningless detections."
            )
        return self.checkpoint


def model_names() -> list[str]:
    return list(models_config().get("models", {}).keys())


def get_model(name: str) -> ModelSpec:
    cfg = models_config()
    models = cfg.get("models", {})
    if name not in models:
        raise RegistryError(
            f"Unknown model '{name}'. Registered in models.yaml: {sorted(models)}"
        )
    m = models[name]
    defaults = dict(cfg.get("defaults", {}))

    train = {**defaults, **(m.get("train") or {})}
    # Environment overrides for the two knobs that legitimately change per
    # machine. Everything else belongs in models.yaml, not in an env var.
    if os.getenv("SSC_DEVICE"):
        train["device"] = os.getenv("SSC_DEVICE")
    if os.getenv("SSC_BATCH"):
        train["batch"] = int(os.environ["SSC_BATCH"])
    if os.getenv("SSC_EPOCHS"):
        train["epochs"] = int(os.environ["SSC_EPOCHS"])

    root = model_root()
    # $SSC_CKPT_<MODEL> points one model at another checkpoint file, for
    # evaluating a candidate without publishing it. Diagnostics only.
    override = os.getenv(f"SSC_CKPT_{name.upper()}")
    checkpoint = Path(override) if override else root / m["checkpoint"]
    return ModelSpec(
        name=name,
        task=m.get("task", "detect"),
        dataset=m["dataset"],
        base_weights=m["base_weights"],
        checkpoint=checkpoint,
        metrics_file=root / m["metrics_file"],
        run_name=m.get("run_name", name),
        classes={int(k): v for k, v in (m.get("classes") or {}).items()},
        train=train,
        inference=dict(m.get("inference") or {}),
        raw=m,
    )


def checkpoint_path(name: str) -> Path:
    """Convenience for inference call sites: resolved path, existence checked."""
    return get_model(name).require_checkpoint()


def env_overrides() -> dict[str, str]:
    """
    The per-model checkpoint paths as environment variables, for Docker and
    .env generation: SSC_MODEL_PLAYER, SSC_MODEL_BALL, ...

    Used by scripts/print_model_env.py so docker-compose and .env.example
    stay in sync with this registry instead of drifting from it (the old
    setup hardcoded a single YOLO_MODEL_PATH pointing at a stock
    checkpoint).
    """
    return {
        f"SSC_MODEL_{n.upper()}": str(get_model(n).checkpoint)
        for n in model_names()
    }
