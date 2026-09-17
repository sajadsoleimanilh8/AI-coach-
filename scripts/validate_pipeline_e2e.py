"""End-to-end pipeline validation on real broadcast clips.

Runs `run_pipeline()` -- the real production entry point, not a
reimplementation -- over N real clips from storage/uploads and reports what
each stage actually produced.

WHY IT TRIMS THE CLIPS
    The installed torch in this environment is CPU-only (2.13.0+cpu). A
    full clip is several thousand frames through five models; the point of
    this script is to exercise every stage on real footage, and a trimmed
    prefix does that identically. Frame counts are reported so nothing is
    implied about what was not run. `--frames 0` runs the whole clip.

WHAT IT DOES NOT DO
    It does not lower `HOMOGRAPHY_CONFIDENCE_MIN`, touch
    `compute_homography()`'s all-point reprojection scoring, or otherwise
    make calibration look better than it is. On real broadcast footage
    `calibration_valid_fraction` is expected to be 0.0, and the purpose
    here is to confirm every downstream stage DEGRADES CORRECTLY when it
    is, not to make it non-zero.

Usage
    python -m scripts.validate_pipeline_e2e --clips 3 --frames 140
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

UPLOAD_DIR = REPO_ROOT / "storage" / "uploads"


def trim_clip(src: Path, dst: Path, n_frames: int,
              start_frame: int = 0) -> tuple[int, float, tuple[int, int]]:
    """Copies `n_frames` frames of `src` starting at `start_frame`.

    `start_frame` exists because these uploads open with a dark intro/fade:
    the player model returns zero detections on frames 0-11 of
    0c9bd797 (mean pixel value ~49), and 2-3 detections at frame 200
    (mean ~90). Trimming from frame 0 therefore measured the title card,
    not the football. Returns (frames_written, fps, (w, h)).
    """
    import cv2

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    written = 0
    while n_frames <= 0 or written < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
    cap.release()
    writer.release()
    return written, float(fps), (w, h)


def _trajectories_from_db(db, M, match_id: str) -> dict:
    """Rebuilds {player_id: [TrackingPoint]} from the rows the pipeline just
    wrote, so the simulation runs on persisted output rather than on an
    in-memory structure a caller could have doctored."""
    from ai.computer_vision.player_tracking.trajectory import TrackingPoint

    traj: dict[int, list] = {}
    rows = (db.query(M.PlayerTracking)
            .filter_by(match_id=match_id)
            .order_by(M.PlayerTracking.player_id, M.PlayerTracking.frame_id)
            .all())
    for r in rows:
        traj.setdefault(r.player_id, []).append(TrackingPoint(
            match_id=match_id, player_id=r.player_id, frame_id=r.frame_id,
            team_id=r.team_id, pixel_x=r.pixel_x, pixel_y=r.pixel_y,
            pitch_x_m=r.pitch_x_m, pitch_y_m=r.pitch_y_m,
            homography_confidence=r.homography_confidence,
            speed=r.speed, distance=r.distance, acceleration=r.acceleration,
        ))
    return traj


def run_one(clip_path: Path, frames: int, workdir: Path, start_frame: int = 0) -> dict:
    """Runs the real pipeline over one clip in an isolated SQLite DB."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    db_path = workdir / f"{clip_path.stem}.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"

    from backend.database import models as M
    from backend.database.session import Base
    from backend.pipeline.latency import PipelineTimer
    from backend.pipeline.runner import run_pipeline

    engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    M.Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    trimmed = workdir / f"trim_{clip_path.stem}.mp4"
    n_written, fps, (w, h) = trim_clip(clip_path, trimmed, frames, start_frame)

    match = M.Match(home_team="Home", away_team="Away",
                    video_path=str(trimmed), duration=0.0)
    db.add(match)
    db.flush()
    video = M.Video(id=clip_path.stem, original_filename=clip_path.name,
                    stored_filename=trimmed.name, content_type="video/mp4",
                    file_size=trimmed.stat().st_size, storage_path=str(trimmed),
                    match_id=match.match_id)
    job = M.ProcessingJob(video_id=video.id, status=M.ProcessingStatus.queued,
                          progress=0, message="e2e")
    db.add(video)
    db.add(job)
    db.commit()
    db.refresh(job)

    out: dict = {
        "clip": clip_path.name, "frames_in_trim": n_written, "start_frame": start_frame,
        "fps": round(fps, 2), "resolution": f"{w}x{h}",
    }

    timer = PipelineTimer(job_id=job.id, match_id=match.match_id)
    try:
        result = run_pipeline(db, job, timer)
    except Exception as exc:  # noqa: BLE001 -- this script reports failures
        out["ERROR"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()[-2000:]
        db.close()
        engine.dispose()
        return out

    out.update({
        "frames_processed": result.frames_processed,
        "players_tracked": result.players_tracked,
        "events_detected": result.events_detected,
        "player_metrics_written": result.player_metrics_written,
        "team_metrics_written": result.team_metrics_written,
        "homography_confidence": round(result.homography_confidence, 4),
        "calibration_valid_fraction": result.calibration_valid_fraction,
        "calibration_episodes": result.calibration_episodes,
        "detectors_available": list(result.detectors_available),
        "reid_merge": result.reid_merge,
    })

    # ---- what actually landed in the database -------------------------
    mid = result.match_id
    out["db"] = {
        "player_tracking_rows": db.query(M.PlayerTracking).filter_by(match_id=mid).count(),
        "frames_rows": db.query(M.Frame).filter_by(match_id=mid).count(),
        "events": db.query(M.Event).filter_by(match_id=mid).count(),
        "player_metrics": db.query(M.PlayerMetric).filter_by(match_id=mid).count(),
        "team_metrics": db.query(M.TeamMetric).filter_by(match_id=mid).count(),
        "calibration_status_rows": db.query(M.CalibrationStatus).filter_by(match_id=mid).count(),
    }

    ev_types: dict[str, int] = {}
    for e in db.query(M.Event).filter_by(match_id=mid).all():
        ev_types[e.event_type] = ev_types.get(e.event_type, 0) + 1
    out["event_types"] = ev_types

    # Per-team metric rows -- the Phase 3 fix. team_id must not be a single
    # pooled "unassigned" bucket if team assignment worked.
    tm_rows = db.query(M.TeamMetric).filter_by(match_id=mid).all()
    by_team: dict[str, dict] = {}
    for t in tm_rows:
        by_team.setdefault(t.team_id, {})[t.metric_name] = {
            "value": t.value_numeric if t.value_numeric is not None else t.value_label,
            "confidence": t.confidence.value if hasattr(t.confidence, "value") else t.confidence,
            "method": t.method.value if hasattr(t.method, "value") else t.method,
        }
    out["team_metrics_by_team"] = by_team

    cs = db.query(M.CalibrationStatus).filter_by(match_id=mid).all()
    out["calibration_invalid_reasons"] = sorted({c.invalid_reason for c in cs if c.invalid_reason})
    out["calibration_valid_episodes"] = sum(1 for c in cs if c.valid)

    # Every metric must carry a real MetricMethod/MetricConfidence.
    bad = []
    for row in db.query(M.PlayerMetric).filter_by(match_id=mid).all():
        m = row.method.value if hasattr(row.method, "value") else row.method
        c = row.confidence.value if hasattr(row.confidence, "value") else row.confidence
        if m not in ("ml_trained", "deterministic", "heuristic_proxy") or \
           c not in ("normal", "low_sample", "low_upstream_confidence"):
            bad.append(f"player_metric {row.metric_name}: method={m} confidence={c}")
    for t in tm_rows:
        m = t.method.value if hasattr(t.method, "value") else t.method
        c = t.confidence.value if hasattr(t.confidence, "value") else t.confidence
        if m not in ("ml_trained", "deterministic", "heuristic_proxy") or \
           c not in ("normal", "low_sample", "low_upstream_confidence"):
            bad.append(f"team_metric {t.metric_name}: method={m} confidence={c}")
    out["honesty_contract_violations"] = bad

    out["stage_timings_s"] = {s.stage: s.seconds for s in timer.stages}
    out["total_pipeline_seconds"] = timer.total_seconds

    # ---- the what-if engine, on THIS clip's real tracked players -------
    # On real footage calibration is invalid, so there are no pitch
    # coordinates and the engine must report `unavailable` rather than
    # producing numbers. That refusal is the thing being validated here.
    try:
        from ai.computer_vision.tactical_analysis.attacking_direction import (
            infer_attacking_directions,
        )
        from ai.simulation_ai.what_if_analysis.engine import Intervention, simulate
        from backend.pipeline.runner import _split_by_team

        by_team = _split_by_team(_trajectories_from_db(db, M, mid))
        teams = sorted(by_team)
        sim_out: dict = {"teams_available": teams}
        if teams:
            res = simulate(
                by_team,
                [Intervention("compactness", team_id=teams[0], pct=10.0)],
                team_assignment_confidence=0.9,
                directions=infer_attacking_directions(
                    {p: pts for tr in by_team.values() for p, pts in tr.items()}),
            )
            d = res.as_dict()
            sim_out.update({
                "interventions": d["interventions"],
                "is_reinforcement_learning": d["is_reinforcement_learning"],
                "n_metrics": len(d["metrics"]),
                "metrics": [
                    {k: m[k] for k in ("metric_name", "team_id", "baseline_value",
                                       "simulated_value", "delta", "confidence",
                                       "method", "recomputed_by")}
                    for m in d["metrics"]
                ],
                "unavailable": d["unavailable"][:4],
            })
        else:
            sim_out["note"] = "no team-split trajectories to simulate from"
        out["simulation"] = sim_out
    except Exception as exc:  # noqa: BLE001
        out["simulation"] = {"ERROR": f"{type(exc).__name__}: {exc}"}
    db.close()
    # Windows keeps the .db file locked until the pool is disposed, which
    # makes TemporaryDirectory cleanup raise.
    engine.dispose()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clips", type=int, default=3)
    ap.add_argument("--start-frame", type=int, default=200,
                    help="skip the intro/fade these clips open with")
    ap.add_argument("--frames", type=int, default=140,
                    help="frames per clip; 0 = whole clip")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "docs" / "dataset_audit" / "phase3_e2e_results.json")
    args = ap.parse_args()

    # Distinct file sizes => distinct source footage rather than three
    # copies of the same upload (storage/uploads has many duplicates).
    seen_sizes: set[int] = set()
    clips: list[Path] = []
    for p in sorted(UPLOAD_DIR.glob("*.mp4")):
        if p.stat().st_size in seen_sizes:
            continue
        seen_sizes.add(p.stat().st_size)
        clips.append(p)
        if len(clips) >= args.clips:
            break

    if not clips:
        print(f"no clips found in {UPLOAD_DIR}", file=sys.stderr)
        return 1

    results = []
    with tempfile.TemporaryDirectory(prefix="ssc_e2e_", ignore_cleanup_errors=True) as tmp:
        for clip in clips:
            print(f"\n=== {clip.name} ({clip.stat().st_size/1e6:.1f} MB) ===", flush=True)
            r = run_one(clip, args.frames, Path(tmp), args.start_frame)
            results.append(r)
            print(json.dumps(r, indent=2, default=str), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
