"""End-to-end regression test: the real pipeline on a real clip.

Unit tests cover the scoring and geometry functions, but the pipeline stages
that wire them together (backend/pipeline/*) had ~12-27% line coverage and no
test of run_pipeline() at all. This test runs the real thing on
samples/sample_15s.mp4 with the five trained checkpoints and checks:

  * structural invariants on everything it writes (valid enums, honest None
    handling, rows present for every stage), and
  * an exact fingerprint of the normalized output, stored in
    tests/golden/pipeline_sample_15s.json -- any change to tracking points,
    events, metrics or calibration rows fails it.

The helper is imported as a sibling module (`pipeline_snapshot`), not as
`tests.pipeline_snapshot`: ultralytics installs its own top-level `tests`
package into site-packages, which shadows this directory's name.

It skips when the clip or any checkpoint is missing (both are gitignored, so it
skips in CI). The fingerprint is machine-specific: GPU and CPU inference can
differ in the last digits. When a change to pipeline output is INTENDED, re-run
with SSC_UPDATE_PIPELINE_GOLDEN=1 to record the new fingerprint, and commit it
with the change that caused it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIP = REPO_ROOT / "samples" / "sample_15s.mp4"
GOLDEN = Path(__file__).parent / "golden" / "pipeline_sample_15s.json"
MODELS = ("player", "ball", "field", "calibration", "goalpost")

ALLOWED_CONFIDENCE = {"normal", "low_sample", "low_upstream_confidence"}
ALLOWED_METHOD = {"ml_trained", "deterministic", "heuristic_proxy"}


def _missing_assets() -> list[str]:
    missing = [] if CLIP.exists() else [str(CLIP.relative_to(REPO_ROOT))]
    try:
        from configs import registry
        for name in MODELS:
            if not Path(registry.checkpoint_path(name)).exists():
                missing.append(f"{name} checkpoint")
    except Exception as exc:  # noqa: BLE001 - any registry problem means "cannot run here"
        missing.append(f"model registry ({exc})")
    return missing


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(bool(_missing_assets()), reason=f"needs local assets: {_missing_assets()}"),
]


@pytest.fixture(scope="module")
def snapshot():
    from pipeline_snapshot import run_snapshot
    return run_snapshot(CLIP)


def test_every_stage_wrote_rows(snapshot):
    tables = snapshot["tables"]
    for table in ("frames", "player_detections", "player_tracking", "events",
                  "player_metrics", "team_metrics", "calibration_status"):
        assert tables.get(table), f"pipeline wrote no {table} rows"
    assert snapshot["result"]["frames_processed"] > 0


def test_metric_rows_use_only_contract_values(snapshot):
    for table in ("player_metrics", "team_metrics"):
        for row in snapshot["tables"][table]:
            assert row["confidence"] in ALLOWED_CONFIDENCE, row
            assert row["method"] in ALLOWED_METHOD, row


def test_low_sample_metrics_are_not_given_a_fake_value(snapshot):
    """The honesty rule: a metric that could not be measured is stored as
    value=None with low_sample, never as a plausible default."""
    for row in snapshot["tables"]["player_metrics"]:
        if row["value"] is None:
            assert row["confidence"] != "normal", f"None value reported as normal confidence: {row}"


def test_progress_is_monotonic_and_finishes(snapshot):
    pct = [p for p, _ in snapshot["progress"]]
    assert pct == sorted(pct), "progress went backwards"
    assert pct and pct[-1] >= 95


def test_output_matches_recorded_fingerprint(snapshot):
    from pipeline_snapshot import fingerprint

    current = fingerprint(snapshot)
    if os.getenv("SSC_UPDATE_PIPELINE_GOLDEN") == "1" or not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if os.getenv("SSC_UPDATE_PIPELINE_GOLDEN") != "1":
            pytest.skip(f"recorded a new fingerprint at {GOLDEN}; commit it")
        return

    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert current["row_counts"] == expected["row_counts"], (
        "pipeline output row counts changed:\n"
        f"  expected {expected['row_counts']}\n  got      {current['row_counts']}"
    )
    assert current["sha256"] == expected["sha256"], (
        "pipeline output changed (same row counts, different values). If intended, "
        "re-run with SSC_UPDATE_PIPELINE_GOLDEN=1 and commit the new fingerprint; "
        "`python tests/pipeline_snapshot.py samples/sample_15s.mp4 out.json` dumps the full output."
    )
