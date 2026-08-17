"""
Re-create `matches` rows for analysis output that outlived its match row.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from sqlalchemy import func

from backend.database.models import (
    AnalysisResult,
    CalibrationStatus,
    Frame,
    Match,
    PlayerTracking,
    TeamMetric,
)
from backend.database.session import SessionLocal

PLACEHOLDER_TEAM = "(recovered)"


def orphaned_match_ids(db) -> set[str]:
    """Every match_id referenced by surviving output but absent from `matches`."""
    referenced: set[str] = set()
    for model in (PlayerTracking, Frame, CalibrationStatus, TeamMetric):
        referenced.update(
            row[0] for row in db.query(model.match_id).distinct().all() if row[0]
        )
    for (payload,) in db.query(AnalysisResult.result_json).all():
        if isinstance(payload, dict) and payload.get("match_id"):
            referenced.add(payload["match_id"])
        elif isinstance(payload, str):
            try:
                value = json.loads(payload).get("match_id")
            except (ValueError, AttributeError):
                value = None
            if value:
                referenced.add(value)

    existing = {row[0] for row in db.query(Match.match_id).all()}
    return referenced - existing


def measured_duration(db, match_id: str) -> float:
    """Seconds, from this match's own persisted Frame rows."""
    row = (
        db.query(func.max(Frame.frame_number), func.max(Frame.fps))
        .filter(Frame.match_id == match_id)
        .first()
    )
    if not row or row[0] is None or not row[1]:
        return 0.0
    return float(row[0]) / float(row[1])


def earliest_timestamp(db, match_id: str) -> datetime | None:
    """Earliest surviving computed_at for this match, for a real created_at."""
    stamps = []
    for model in (TeamMetric, CalibrationStatus):
        value = (
            db.query(func.min(model.computed_at))
            .filter(model.match_id == match_id)
            .scalar()
        )
        if value:
            stamps.append(value)
    return min(stamps) if stamps else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true",
                       help="list what would be restored, change nothing")
    group.add_argument("--apply", action="store_true",
                       help="insert the missing match rows")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        missing = sorted(orphaned_match_ids(db))
        if not missing:
            print("No orphaned matches. Nothing to restore.")
            return 0

        print(f"{len(missing)} match_id(s) referenced by surviving output but "
              f"absent from `matches`:\n")
        print(f"  {'match_id':38s} {'tracking':>9s} {'frames':>7s} "
              f"{'team_metrics':>13s} {'duration_s':>11s}")
        for match_id in missing:
            n_track = db.query(func.count()).select_from(PlayerTracking).filter(
                PlayerTracking.match_id == match_id).scalar()
            n_frames = db.query(func.count()).select_from(Frame).filter(
                Frame.match_id == match_id).scalar()
            n_team = db.query(func.count()).select_from(TeamMetric).filter(
                TeamMetric.match_id == match_id).scalar()
            print(f"  {match_id:38s} {n_track:>9d} {n_frames:>7d} "
                  f"{n_team:>13d} {measured_duration(db, match_id):>11.2f}")

        if args.dry_run:
            print("\n--dry-run: nothing written. Re-run with --apply to restore.")
            return 0

        for match_id in missing:
            db.add(Match(
                match_id=match_id,
                home_team=PLACEHOLDER_TEAM,
                away_team=PLACEHOLDER_TEAM,
                video_path="",
                duration=measured_duration(db, match_id),
                created_at=earliest_timestamp(db, match_id) or datetime.utcnow(),
            ))
        db.commit()

        print(f"\nRestored {len(missing)} match row(s).")
        print("Team names show as '(recovered)' and these matches have no video: "
              "both were destroyed with the rows and are not recoverable. Their "
              "tracking, calibration, team-metric and event data is intact and "
              "now reachable again.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
