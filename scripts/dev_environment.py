"""
Prepare and verify a usable dev environment for every dashboard tab.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "backend" / "database" / "sports_strategy.db"

BACKEND = os.getenv("SSC_BACKEND_URL", "http://localhost:8000")
NEXUS = os.getenv("SSC_NEXUS_URL", "http://localhost:8100")
FRONTEND = os.getenv("SSC_FRONTEND_URL", "http://localhost:3000")

DEMO_PLAYER = "demo-player-1"

OK, WARN, BAD = "  OK  ", " WARN ", " FAIL "



def _get(url: str, timeout: float = 5.0, expect_json: bool = True):
    """Returns (status, parsed_body_or_None)."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            raw = response.read()
            if not expect_json:
                return response.status, None
            try:
                return response.status, json.loads(raw or b"null")
            except ValueError:
                return response.status, None
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.loads(error.read() or b"null")
        except Exception:
            return error.code, None
    except Exception:
        return 0, None


def _post(url: str, payload: dict, timeout: float = 30.0):
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as error:
        try:
            return error.code, json.loads(error.read() or b"null")
        except Exception:
            return error.code, None
    except Exception:
        return 0, None



def best_match() -> dict | None:
    """The match with the most computed metrics AND a video still on disk."""
    if not DB_PATH.exists():
        return None

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT m.match_id, m.video_path,
               (SELECT COUNT(*) FROM player_metrics  p WHERE p.match_id = m.match_id),
               (SELECT COUNT(*) FROM team_metrics    t WHERE t.match_id = m.match_id),
               (SELECT COUNT(*) FROM player_tracking r WHERE r.match_id = m.match_id)
        FROM matches m
        """
    )
    rows = cursor.fetchall()
    connection.close()

    candidates = [
        {
            "match_id": match_id,
            "video_exists": bool(video_path) and Path(video_path).exists(),
            "player_metrics": pm,
            "team_metrics": tm,
            "tracking_rows": tr,
        }
        for match_id, video_path, pm, tm, tr in rows
    ]
    playable = [c for c in candidates if c["video_exists"]] or candidates
    if not playable:
        return None
    return max(playable, key=lambda c: (c["player_metrics"] + c["team_metrics"], c["tracking_rows"]))



HEALTH_PROFILES = [
    dict(sleep_duration_hours=8.5, sleep_quality=9, night_awakenings=0, trained_last_24h=False,
         trained_last_48h=True, training_duration_minutes=60, training_intensity=4,
         high_intensity_activity=False, hours_since_last_training=36, fatigue=2,
         muscle_soreness=2, pain_level=1, perceived_readiness=9, hydration_liters=3.0,
         nutrition_quality=9, hours_since_last_meal=2.5,
         caffeine_or_supplement_notes="Well rested, light session two days ago."),
    dict(sleep_duration_hours=6.0, sleep_quality=5, night_awakenings=2, trained_last_24h=True,
         trained_last_48h=True, training_duration_minutes=95, training_intensity=8,
         high_intensity_activity=True, hours_since_last_training=14, fatigue=6,
         muscle_soreness=6, pain_level=3, perceived_readiness=5, hydration_liters=1.6,
         nutrition_quality=6, hours_since_last_meal=5.0,
         caffeine_or_supplement_notes="Double session yesterday, legs heavy."),
    dict(sleep_duration_hours=4.5, sleep_quality=3, night_awakenings=4, trained_last_24h=True,
         trained_last_48h=True, training_duration_minutes=120, training_intensity=9,
         high_intensity_activity=True, hours_since_last_training=8, fatigue=9,
         muscle_soreness=8, pain_level=6, perceived_readiness=3, hydration_liters=0.9,
         nutrition_quality=3, hours_since_last_meal=7.0,
         caffeine_or_supplement_notes="Poor sleep, travelled overnight."),
]

PSYCH_PROFILES = [
    dict(concentration_level=9, focus_maintenance=9, mental_clarity_raw=9, pre_match_stress=2,
         importance_pressure=4, nervousness=2, performance_confidence=9,
         tactical_confidence_raw=9, match_motivation=9, competitive_motivation_raw=9,
         mistake_recovery_speed=8, pressure_performance_effect="improves", post_error_calm=8),
    dict(concentration_level=6, focus_maintenance=5, mental_clarity_raw=6, pre_match_stress=6,
         importance_pressure=7, nervousness=6, performance_confidence=6,
         tactical_confidence_raw=6, match_motivation=7, competitive_motivation_raw=7,
         mistake_recovery_speed=5, pressure_performance_effect="no_change", post_error_calm=5),
    dict(concentration_level=3, focus_maintenance=3, mental_clarity_raw=4, pre_match_stress=9,
         importance_pressure=9, nervousness=8, performance_confidence=3,
         tactical_confidence_raw=4, match_motivation=6, competitive_motivation_raw=5,
         mistake_recovery_speed=2, pressure_performance_effect="reduces", post_error_calm=3),
]


def seed(player_id: str) -> None:
    print(f"\nSeeding questionnaires for player_id={player_id!r} via the real scoring API")

    for label, profiles, path in (
        ("health", HEALTH_PROFILES, f"/api/prematch_health/{player_id}/submit"),
        ("psychology", PSYCH_PROFILES, f"/api/psychology/{player_id}/submit"),
    ):
        for index, profile in enumerate(profiles, start=1):
            status, body = _post(f"{BACKEND}{path}", dict(profile))
            if status == 201:
                if label == "health":
                    score = f"readiness={body['physical_readiness']:.1f} risk={body['performance_risk']}"
                else:
                    score = f"mental={body['mental_readiness']} stress={body['stress']} risk={body['pressure_risk']}"
                print(f"  {OK} {label} #{index}: {score}")
            else:
                detail = (body or {}).get("detail") if isinstance(body, dict) else body
                print(f"  {BAD} {label} #{index}: HTTP {status} {detail}")



def redis_alive(host: str = "127.0.0.1", port: int = 6379) -> bool:
    """Raw PING over a socket — no redis-py needed, and it proves the server
    actually answers rather than merely that a port is open."""
    import socket

    try:
        with socket.create_connection((host, port), timeout=1.5) as sock:
            sock.sendall(b"PING\r\n")
            return b"PONG" in sock.recv(64)
    except Exception:
        return False


def celery_worker_alive() -> bool | None:
    """True/False when celery is importable, None when it is not."""
    try:
        from backend.celery_app import celery_app
    except Exception:
        return None
    try:
        return bool(celery_app.control.ping(timeout=1.5))
    except Exception:
        return False


def report(match: dict | None) -> None:
    print("=" * 74)
    print("SERVICES")
    print("=" * 74)

    has_redis = redis_alive()
    print(f"  {OK if has_redis else WARN} redis          127.0.0.1:6379   (broker + metrics cache)")
    if not has_redis:
        print("        start:  docker run -d --name ssc-redis -p 6379:6379 redis:7-alpine")
        print("        without it, uploads save the file but the job is marked failed")
        print("        ('Processing queue unavailable'). Re-run one by hand with:")
        print("        venv/Scripts/python -m scripts.run_job_local --latest")

    worker = celery_worker_alive()
    worker_mark = OK if worker else (WARN if worker is None else WARN)
    worker_note = (
        "responding" if worker
        else ("could not check (celery not importable here)" if worker is None else "no worker responding")
    )
    print(f"  {worker_mark} celery worker  {worker_note}")
    if worker is False:
        print("        start:  venv/Scripts/python -m celery -A backend.celery_app worker "
              "--loglevel=info --pool=solo")
        print("        NOTE :  --pool=solo is required on Windows; the default prefork")
        print("                pool does not work there.")


    backend_status, _ = _get(f"{BACKEND}/health")
    print(f"  {OK if backend_status == 200 else BAD} core backend   {BACKEND}")
    if backend_status != 200:
        print("        start:  venv/Scripts/python -m uvicorn backend.api.main:app --port 8000")
        print("        NOTE :  use venv/, not .venv/ -- only venv/ has ultralytics+torch,")
        print("                which Calibration needs.")

    nexus_status, nexus_body = _get(f"{NEXUS}/api/health")
    print(f"  {OK if nexus_status == 200 else WARN} NEXUS          {NEXUS}   (optional; Coach Chat only)")
    if nexus_status == 200 and isinstance(nexus_body, dict):
        models = nexus_body.get("local_models") or []
        chat_models = [m for m in models if "embed" not in m]
        print(f"        providers: {nexus_body.get('providers')}")
        print(f"        local chat models: {chat_models or 'NONE -- run: ollama pull mistral'}")
    elif nexus_status != 200:
        print("        start:  venv/Scripts/python -m uvicorn nexus.api.main:app --port 8100")

    frontend_status, _ = _get(FRONTEND, expect_json=False)
    print(f"  {OK if frontend_status == 200 else BAD} frontend       {FRONTEND}")
    if frontend_status != 200:
        print("        start:  cd frontend/web && npm run dev")

    print()
    print("=" * 74)
    print("TABS")
    print("=" * 74)

    if not match:
        print(f"  {BAD} no matches in the database — every match-scoped tab will be empty.")
        return

    mid = match["match_id"]
    _, summary = _get(f"{BACKEND}/api/matches/{mid}")
    summary = summary if isinstance(summary, dict) else {}

    def line(num, name, ready, note):
        print(f"  {OK if ready else WARN} {num} {name:22s} {note}")

    line("01", "Match Analysis", bool(summary.get("video_file_exists")),
         f"video={'playable' if summary.get('video_file_exists') else 'missing'} "
         f"job={summary.get('job_status') or 'none'}")
    line("02", "Player Intelligence", match["player_metrics"] > 0,
         f"{match['player_metrics']} player metrics")
    line("03", "Team Intelligence", match["team_metrics"] > 0,
         f"{match['team_metrics']} team metrics")
    line("04", "Calibration", bool(summary.get("video_file_exists")),
         "decodes frames from the stored video (needs ultralytics)")
    line("05", "Simulation", match["tracking_rows"] > 0,
         f"{match['tracking_rows']} tracking rows to perturb")

    health_status, _ = _get(f"{BACKEND}/api/prematch_health/{DEMO_PLAYER}/latest")
    line("06", "Pre-Match Health", health_status == 200,
         f"demo player {DEMO_PLAYER!r} "
         f"{'has history' if health_status == 200 else '-- run with --seed'}")

    psych_status, _ = _get(f"{BACKEND}/api/psychology/{DEMO_PLAYER}/latest")
    line("07", "Pre-Match Psychology", psych_status == 200,
         f"demo player {DEMO_PLAYER!r} "
         f"{'has history' if psych_status == 200 else '-- run with --seed'}")

    line("08", "Coach Chat", nexus_status == 200,
         "NEXUS reachable" if nexus_status == 200 else "NEXUS down on :8100")

    print()
    print("=" * 74)
    print("USE THIS MATCH")
    print("=" * 74)
    print(f"  {mid}")
    print("  Paste it into Match Analysis -> 'Attach an Already-Processed Match'.")
    print(f"  Questionnaire tabs: use player id  {DEMO_PLAYER}")
    print()
    print("  Honest expectations for this data:")
    print("    - heatmaps / pitch minimap stay EMPTY: no tracking row has pitch")
    print("      coordinates (calibration never validated on this footage).")
    print("      The video overlay still works -- that is pixel space.")
    print("    - formation shows 'Not Detected': the row exists, its value is")
    print("      genuinely null (reason: insufficient_outfield_players).")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", action="store_true", help="seed the questionnaire tabs")
    parser.add_argument("--player", default=DEMO_PLAYER, help="player id to seed")
    args = parser.parse_args()

    backend_status, _ = _get(f"{BACKEND}/health")
    if args.seed:
        if backend_status != 200:
            print(f"{BAD} backend is not running at {BACKEND} -- cannot seed through the API.")
            return 1
        seed(args.player)
        print()

    report(best_match())
    return 0


if __name__ == "__main__":
    sys.exit(main())
