"""
Shared training/eval core for all five specialised CV models.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from configs import registry as R


@dataclass
class TrainOutcome:
    model: str
    task: str
    dataset: str
    data_yaml: str
    weights: str
    metrics_file: str
    metrics: dict
    started_at: str
    finished_at: str
    train_args: dict


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=R.REPO_ROOT, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:                                        # noqa: BLE001
        return None


def _environment() -> dict:
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_sha": _git_sha(),
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["vram_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    except Exception:                                        # noqa: BLE001
        pass
    try:
        import ultralytics
        env["ultralytics"] = ultralytics.__version__
    except Exception:                                        # noqa: BLE001
        pass
    return env


def preflight(model_name: str) -> tuple[R.ModelSpec, Path]:
    """
    Verifies the dataset and resolves the generated data.yaml.
    """
    spec = R.get_model(model_name)
    print(f"[preflight] model={spec.name} task={spec.task} dataset={spec.dataset}")
    report = R.verify_dataset(spec.dataset, strict=False)
    print(f"[preflight] dataset root: {report.root}")
    for src in report.sources:
        print(f"[preflight]   source: {src}")
    if not report.ok:
        raise R.DatasetStructureError(R.format_dataset_report(report))
    for s in report.splits:
        print(f"[preflight]   ok {s.name}: {s.n_images:,} images / {s.n_labels:,} labels")

    data_yaml = R.build_data_yaml(spec.dataset)
    print(f"[preflight] generated data.yaml: {data_yaml}")
    return spec, data_yaml


def train_model(model_name: str, *, resume: bool = False,
                overrides: dict | None = None) -> TrainOutcome:
    """Trains one registered model end to end and writes its metrics."""
    spec, data_yaml = preflight(model_name)
    started = datetime.now(timezone.utc).isoformat()

    from ultralytics import YOLO

    args: dict[str, Any] = dict(spec.train)
    args.update(overrides or {})
    args.update({
        "data": str(data_yaml),
        "project": str(R.runs_root()),
        "name": spec.run_name,
        "resume": resume,
    })
    print(f"[train] {spec.name}: {json.dumps({k: v for k, v in args.items() if k != 'data'}, default=str)}")

    model = YOLO(spec.base_weights)
    if model.task != spec.task:
        raise R.RegistryError(
            f"Registry declares task '{spec.task}' for model '{spec.name}', but "
            f"base_weights '{spec.base_weights}' loads as task '{model.task}'. "
            f"Fix configs/models.yaml -- training would silently produce the "
            f"wrong kind of model."
        )
    model.train(**args)

    metrics = evaluate_model(spec, model=model, data_yaml=data_yaml)

    weights_src = R.runs_root() / spec.run_name / "weights" / "best.pt"
    spec.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if weights_src.exists():
        shutil.copy2(weights_src, spec.checkpoint)
        print(f"[train] checkpoint -> {spec.checkpoint}")
    else:
        print(f"[train] WARNING: expected weights not found at {weights_src}")

    outcome = TrainOutcome(
        model=spec.name, task=spec.task, dataset=spec.dataset,
        data_yaml=str(data_yaml), weights=str(spec.checkpoint),
        metrics_file=str(spec.metrics_file), metrics=metrics,
        started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
        train_args={k: str(v) for k, v in args.items()},
    )
    spec.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(outcome) | {"environment": _environment()}
    spec.metrics_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[train] metrics -> {spec.metrics_file}")
    return outcome


def evaluate_model(spec: R.ModelSpec, *, model=None, data_yaml: Path | None = None,
                   split: str = "test") -> dict:
    """
    Runs ultralytics' validator and extracts the real metric values.
    """
    from ultralytics import YOLO

    if model is None:
        model = YOLO(str(spec.require_checkpoint()))
    if data_yaml is None:
        data_yaml = R.build_data_yaml(spec.dataset)

    res = model.val(
        data=str(data_yaml), split=split,
        imgsz=spec.inference.get("imgsz", spec.train.get("imgsz", 640)),
        conf=0.001,
        iou=0.6,
        plots=True,
        project=str(R.runs_root()), name=f"{spec.run_name}_val_{split}",
    )

    out: dict[str, Any] = {"split": split, "raw": {}}
    try:
        out["raw"] = {k: float(v) for k, v in res.results_dict.items()
                      if isinstance(v, (int, float))}
    except Exception:                                        # noqa: BLE001
        pass

    box = getattr(res, "box", None)
    if box is not None:
        out["box"] = {
            "precision": float(box.mp), "recall": float(box.mr),
            "map50": float(box.map50), "map50_95": float(box.map),
        }
        try:
            out["per_class"] = {
                spec.classes.get(int(c), str(c)): {
                    "precision": float(box.p[i]), "recall": float(box.r[i]),
                    "map50": float(box.ap50[i]), "map50_95": float(box.ap[i]),
                }
                for i, c in enumerate(res.ap_class_index)
            }
        except Exception:                                    # noqa: BLE001
            pass
    for attr, key in (("seg", "mask"), ("pose", "pose")):
        m = getattr(res, attr, None)
        if m is not None:
            out[key] = {
                "precision": float(m.mp), "recall": float(m.mr),
                "map50": float(m.map50), "map50_95": float(m.map),
            }
    out["confusion_matrix_png"] = str(
        R.runs_root() / f"{spec.run_name}_val_{split}" / "confusion_matrix.png")
    out["save_dir"] = str(getattr(res, "save_dir", ""))
    return out
