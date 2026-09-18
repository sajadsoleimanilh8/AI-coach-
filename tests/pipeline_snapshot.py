"""Run the real pipeline on a clip and return a normalized snapshot of its output.

Shared by tests/test_pipeline_e2e.py and usable by hand when refactoring:

    python tests/pipeline_snapshot.py samples/sample_15s.mp4 out.json

The pipeline writes into a throwaway SQLite file; the caller is responsible
for pointing DATABASE_URL at a temp location before backend.* is imported (the
root conftest.py already does this for pytest). Every output row is dumped with
surrogate ids, timestamps and wall-clock timings removed and floats rounded, so
two runs of unchanged code on the same machine produce identical snapshots.
"""

from __future__ import annotations

import enum
import hashlib
import json
import shutil
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any

#: Fields that legitimately differ run to run and say nothing about behaviour.
VOLATILE = frozenset({
    "match_id", "metric_id", "event_id", "id", "job_id", "video_id", "created_at", "updated_at",
    "computed_at", "timestamp", "detection_id", "tracking_id", "status_id", "calibration_id",
    "frame_row_id", "storage_path", "video_path", "output_path", "path", "duration_ms",
    "elapsed_ms", "wall_seconds", "seconds", "started_at", "finished_at",
})


def _norm(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return None
    if isinstance(value, dict):
        return {k: _norm(v) for k, v in sorted(value.items()) if k not in VOLATILE}
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    return value


def run_snapshot(clip: Path) -> dict[str, Any]:
    """Run run_pipeline() on `clip` in an isolated SQLite DB; return the snapshot."""
    from sqlalchemy import create_engine, inspect
    from sqlalchemy.orm import sessionmaker

    from backend.database import models as M
    from backend.database.session import Base
    from backend.pipeline.latency import PipelineTimer
    from backend.pipeline.runner import run_pipeline

    work = Path(tempfile.mkdtemp(prefix="ssc_pipeline_snapshot_"))
    local_clip = work / clip.name
    shutil.copy2(clip, local_clip)

    engine = create_engine(f"sqlite:///{work / 'snapshot.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        match = M.Match(home_team="Home", away_team="Away", video_path=str(local_clip), duration=0.0)
        db.add(match)
        db.flush()
        video = M.Video(id="snapshot-video", original_filename=clip.name, stored_filename=clip.name,
                        content_type="video/mp4", file_size=local_clip.stat().st_size,
                        storage_path=str(local_clip), match_id=match.match_id)
        job = M.ProcessingJob(video_id=video.id, status=M.ProcessingStatus.queued, progress=0,
                              message="snapshot")
        db.add_all([video, job])
        db.commit()
        db.refresh(job)

        progress: list = []
        result = run_pipeline(db, job, PipelineTimer(job_id=job.id, match_id=match.match_id),
                              lambda pct, msg: progress.append([pct, msg]))

        tables: dict[str, list] = {}
        for mapper in Base.registry.mappers:
            cls = mapper.class_
            if not hasattr(cls, "__tablename__"):
                continue
            cols = [c.key for c in inspect(cls).column_attrs if c.key not in VOLATILE]
            rows = [_norm({c: getattr(r, c) for c in cols}) for r in db.query(cls).all()]
            rows.sort(key=lambda r: json.dumps(r, sort_keys=True, default=str))
            tables[cls.__tablename__] = rows

        return {
            "result": {k: _norm(v) for k, v in vars(result).items() if k not in VOLATILE},
            "progress": progress,
            "tables": tables,
        }
    finally:
        db.close()
        engine.dispose()
        shutil.rmtree(work, ignore_errors=True)


def fingerprint(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Small, committable summary: row counts plus a hash of everything."""
    blob = json.dumps(snapshot, sort_keys=True, default=str).encode("utf-8")
    return {
        "sha256": hashlib.sha256(blob).hexdigest(),
        "row_counts": {t: len(rows) for t, rows in sorted(snapshot["tables"].items()) if rows},
        "frames_processed": snapshot["result"].get("frames_processed"),
    }


if __name__ == "__main__":
    import os
    import sys

    os.environ.setdefault("DATABASE_URL", f"sqlite:///{Path(tempfile.mkdtemp()) / 'unused.db'}")
    snap = run_snapshot(Path(sys.argv[1]).resolve())
    Path(sys.argv[2]).write_text(json.dumps(snap, indent=1, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps(fingerprint(snap), indent=1))
