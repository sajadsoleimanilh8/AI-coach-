"""
Single source of truth for dataset and model checkpoint locations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

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



_cache: dict[str, Any] = {}


def _load(path: Path) -> dict:
    key = str(path)
    if key not in _cache:
        if not path.exists():
            raise RegistryError(f"Registry config missing: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            _cache[key] = yaml.safe_load(fh) or {}
    return _cache[key]


def datasets_config() -> dict:
    return _load(DATASETS_YAML)


def models_config() -> dict:
    return _load(MODELS_YAML)



def dataset_root() -> Path:
    """
    Resolves the dataset root. Order (first hit wins):
        1. $SSC_DATASET_ROOT
        2. $DATASET_ROOT
        3. datasets.yaml::root_default_windows   (on Windows)
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
    """Root for raw third-party downloads (SoccerNet et al)."""
    val = os.getenv("SSC_EXTERNAL_ROOT")
    if val:
        return Path(val).expanduser()
    return dataset_root().parent / "external"


def external_sources_config() -> dict:
    return datasets_config().get("external_sources", {}) or {}


def external_source(name: str) -> dict:
    """Spec for one raw external download."""
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



def build_data_yaml(name: str, out_dir: Path | None = None,
                    dedupe: bool = True, refresh: bool = False) -> Path:
    """
    Writes a resolved, absolute-path data.yaml for `name` and returns it.
    """
    spec = dataset_spec(name)
    sources = dataset_source_dirs(name)
    out_dir = out_dir or (REPO_ROOT / "runs" / "_data")
    out_dir.mkdir(parents=True, exist_ok=True)

    def split_dirs(split: str) -> Any:
        paths = [str((s / split / "images").resolve()) for s in sources]
        return paths[0] if len(paths) == 1 else paths

    if dedupe:
        from scripts.dataset_dedupe import build_manifests

        manifest_dir = out_dir / name
        needed = [manifest_dir / f"{k}.txt" for k in ("train", "val", "test")]
        if refresh or not all(p.exists() for p in needed):
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
        """Returns the checkpoint path, raising if it is not on disk."""
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
    """
    return {
        f"SSC_MODEL_{n.upper()}": str(get_model(n).checkpoint)
        for n in model_names()
    }
