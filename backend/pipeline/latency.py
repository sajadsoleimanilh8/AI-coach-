"""
Real pipeline-stage timing.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class StageTiming:
    stage: str
    seconds: float
    detail: str | None = None


@dataclass
class PipelineTimer:
    """Wrap each pipeline stage in `with timer.stage("name"):`. Every
    number this produces is a real time.perf_counter() delta around code
    that actually ran -- there is no path to a fabricated number here."""

    job_id: str
    match_id: str | None = None
    stages: list[StageTiming] = field(default_factory=list)

    @contextmanager
    def stage(self, name: str, detail: str | None = None):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.stages.append(StageTiming(stage=name, seconds=round(elapsed, 4), detail=detail))

    @property
    def total_seconds(self) -> float:
        return round(sum(s.seconds for s in self.stages), 4)

    def to_report_dict(self) -> dict:
        """Shape matches backend/api/schemas.py::PipelineLatencyReport
        exactly, so this can be written straight into
        AnalysisResult.result_json and served back without reshaping."""
        return {
            "job_id": self.job_id,
            "match_id": self.match_id,
            "total_seconds": self.total_seconds,
            "stages": [
                {"stage": s.stage, "seconds": s.seconds, "detail": s.detail}
                for s in self.stages
            ],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "measured",
        }
