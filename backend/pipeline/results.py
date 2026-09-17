"""Pipeline result and error types.

Moved verbatim out of backend/pipeline/runner.py; run_pipeline() there is
the orchestrator that calls these in order.
"""


from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class PipelineAssetError(Exception):
    """A required asset (model weights, video file, calibration) was
    missing or unusable. Distinct from an unexpected bug -- tasks.py
    surfaces this as job.error with the message as-is, since it's already
    written to be actionable (what's missing, where the pipeline expected
    to find it)."""


@dataclass
class PipelineResult:
    match_id: str
    frames_processed: int
    players_tracked: int
    events_detected: int
    player_metrics_written: int
    team_metrics_written: int
    homography_confidence: float
    # Fraction of frames whose calibration passed BOTH the confidence gate
    # and the geometric cross-check. Reported alongside the scalar
    # confidence above because the two answer different questions: a run
    # can have a high confidence on the frames it solved while having
    # solved almost none of them.
    calibration_valid_fraction: float = 0.0
    calibration_episodes: int = 0
    detectors_available: tuple = ()
    # OverlayRenderResult.as_dict(), or None when the stage did not run.
    # Carried through to the job's AnalysisResult so a missing processed
    # video is always accompanied by the reason it is missing.
    overlay_render: dict | None = None
    reid_merge: dict | None = None
    pitch_coord_coverage: dict | None = None
