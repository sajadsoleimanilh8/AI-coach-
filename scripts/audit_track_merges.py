"""
Audits reid_merge.py's hardest promise: it never merges across teams.

    venv/Scripts/python.exe scripts/audit_track_merges.py --video videos/test2.mp4 --frames 1500

WHY THIS EXISTS
The unit tests in ai/computer_vision/player_tracking/tests/test_reid_merge.py
check the team gate mechanically -- given two labels that differ, refuse to
merge -- which is true by construction and says nothing about whether the
labels themselves are right. reid_merge resolves teams at
TEAM_SEPARATION_CONFIDENCE_MIN, which is 0.0: every track gets a label, however
weak the clustering evidence. So the promise is only ever as good as that
labelling, and nothing in the test suite can see that.

This script closes the gap on real footage. It labels the UNMERGED tracks a
second time at the reporting-grade bar (TEAM_ASSIGNMENT_CONFIDENCE_MIN, the one
the pipeline uses to write a team onto a detection) and checks every merge
reid_merge actually made against it. Three outcomes per merge:

    agree     -- both tracks confidently labelled, same team. Verified good.
    unknown   -- at least one track too weakly labelled to judge. The merge is
                 NOT known to be wrong, but it is not verified either; it
                 rested on colour-histogram correlation and geometry alone.
    DISAGREE  -- both confidently labelled, opposite teams. A real violation.

A non-zero DISAGREE count is a bug, and this script exits 1 on one.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai.computer_vision.player_tracking import reid_merge
from ai.computer_vision.player_tracking.tracker import track_video
from ai.computer_vision.tactical_analysis.team_assignment import assign_teams_with_stats
from configs import registry


def audit(video_path: str, n_frames: int, device: str = "0") -> dict:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    frames = []
    for detections in track_video(str(registry.checkpoint_path("player")),
                                  video_path, device=device):
        frames.append(detections)
        if len(frames) >= n_frames:
            break

    # Reporting-grade labels, computed on a copy so the merge below still sees
    # the ids the tracker produced.
    unmerged = copy.deepcopy(frames)
    strict = {pid: stats.team_id
              for pid, stats in assign_teams_with_stats(video_path, unmerged).tracks.items()}

    result = reid_merge.merge_reidentified_tracks(video_path, frames, fps=fps)

    agree = unknown = 0
    conflicts: list[tuple[int, int, str, str]] = []
    for dead, born in result.merged_pairs:
        team_dead, team_born = strict.get(dead), strict.get(born)
        if team_dead is None or team_born is None:
            unknown += 1
        elif team_dead != team_born:
            conflicts.append((dead, born, team_dead, team_born))
        else:
            agree += 1

    labelled = sum(1 for v in strict.values() if v is not None)
    return {
        "video": video_path,
        "frames": len(frames),
        "tracks": len(strict),
        "confidently_labelled": labelled,
        "merges": result.merges,
        "verified_same_team": agree,
        "unverifiable": unknown,
        "cross_team_merges": len(conflicts),
        "conflicts": conflicts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default="videos/test1.mp4")
    parser.add_argument("--frames", type=int, default=1500)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    report = audit(args.video, args.frames, args.device)

    print(f"\n--- merge audit: {report['video']} ({report['frames']} frames) ---")
    print(f"  tracks                          {report['tracks']}")
    print(f"  confidently labelled by team    {report['confidently_labelled']}")
    print(f"  merges made                     {report['merges']}")
    print(f"  verified same-team              {report['verified_same_team']}")
    print(f"  unverifiable (weak label)       {report['unverifiable']}")
    print(f"  CROSS-TEAM MERGES               {report['cross_team_merges']}")
    for dead, born, team_dead, team_born in report["conflicts"]:
        print(f"      {dead} -> {born}: {team_dead} vs {team_born}")

    if report["cross_team_merges"]:
        print("\nFAIL: reid_merge joined tracks the team assignment separates.")
        return 1
    print("\nOK: no merge joined two confidently-labelled opposing tracks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
