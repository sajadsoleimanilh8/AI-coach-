Football Analysis Database Schema
Purpose of This Revision

This document merges and reconciles four previously separate docs:

docs/database_schema.md
docs/tracking_system_design.md
docs/event_detection.md
docs/data_analysis.md (Analysis Logic Design v3)

Those four were written independently and disagreed with each other on field names, missing fields, and types. This doc is now the single source of truth — if another doc conflicts with this one, this one wins. Five concrete conflicts were found and resolved; each is called out inline below with a FIXED: note so the team can see what changed and why.

Entity Relationship Overview
Match
 ├─→ Frame
 │    ├─→ PlayerDetection
 │    └─→ BallDetection
 ├─→ PlayerTracking   (derived from PlayerDetection + ByteTrack + Homography)
 ├─→ Event            (derived from PlayerTracking + BallDetection)
 ├─→ PlayerMetric     (derived from Event, per Analysis Logic Design v3)
 └─→ TeamMetric       (derived from Event + PlayerMetric aggregates)
Match

General match information. Unchanged from v1.

Field	Type	Notes
match_id	UUID (PK)	
home_team	String	
away_team	String	
video_path	String	
duration	Float	seconds
created_at	DateTime	
Frame

Processed video frames.

Field	Type	Notes
frame_id	Integer (PK)	
match_id	UUID (FK → Match)	
frame_number	Integer	
timestamp	Float	seconds from match start
fps	Float	added — was in tracking_system_design.md's frame input format but missing from the schema table; needed to convert frame_number ↔ timestamp reliably
PlayerDetection

Raw YOLO detection output, one row per player per frame.

Field	Type	Notes
detection_id	UUID (PK)	
frame_id	Integer (FK → Frame)	
player_id	Integer	ByteTrack-assigned tracking ID, not a stable player identity across matches
team_id	String, nullable	FIXED (gap #2): was completely absent from PlayerDetection/PlayerTracking in v1, so there was no way to know which team a detected player belonged to at the detection level. Null until team assignment runs.
team_assignment_confidence	Float, nullable	added. 0–1, from jersey-cluster separation. Per Analysis Logic Design v3 §0.2: treat team_id as unassigned if this is below TEAM_ASSIGNMENT_CONFIDENCE_MIN (0.5), even if team_id itself is non-null.
x, y, width, height	Float	pixel-space bounding box
confidence	Float	YOLO detection confidence, 0–1
BallDetection

Ball detection output, one row per frame (or per candidate, if multiple).

Field	Type	Notes
detection_id	UUID (PK)	added — v1 had no primary key on this table
frame_id	Integer (FK → Frame)	
ball_x, ball_y	Float	pixel-space
confidence	Float	
PlayerTracking

Continuous player trajectories, one row per player per frame, in both pixel and pitch coordinates.

Field	Type	Notes
tracking_id	UUID (PK)	added — v1 had no primary key
match_id	UUID (FK → Match)	added, denormalized for query performance (avoids joining through Frame for every match-level aggregate)
player_id	Integer	
frame_id	Integer (FK → Frame)	
team_id	String, nullable	FIXED (gap #2) — same reasoning as PlayerDetection
pixel_x, pixel_y	Float	raw tracked position, kept for debugging/re-calibration
pitch_x_m, pitch_y_m	Float, nullable	renamed from pitch_x_meter/pitch_y_meter (v1 naming) to match the _m suffix convention used throughout Analysis Logic Design v3. Nullable because homography may fail for a given frame.
homography_confidence	Float, nullable	FIXED (gap #2), critical: this field was entirely missing from every prior doc despite Analysis Logic Design v3 §0.2 explicitly requiring it. Per v3: if this is below HOMOGRAPHY_CONFIDENCE_MIN (0.6), pitch_x_m/pitch_y_m must be treated as unusable by any downstream consumer.
speed	Float	m/s, computed from consecutive pitch_x_m/pitch_y_m — null if homography unavailable for either endpoint
distance	Float	meters, cumulative or per-frame-delta (pick one convention — recommend per-frame-delta, sum in queries)
acceleration	Float	added — tracking_system_design.md §7 defines this formula and lists it as used for sprint load / fatigue, but it was missing from that same doc's own schema table (§9)
Event

Detected match events (pass, shot, touch, turnover, press, etc.).

FIXED (gap #1): database_schema.md and event_detection.md each defined this table differently — one had metadata: JSON with no match_id, the other had extra_data with match_id. Unified below; match_id is kept (needed for match-scoped queries without joining through Frame every time), and the JSON field is standardized as metadata (matches the naming used in Analysis Logic Design v3's output contract for consistency with PlayerMetric.sub_scores style).

Field	Type	Notes
event_id	UUID (PK)	
match_id	UUID (FK → Match)	unified — present in both source docs' intent, now explicit
frame_id	Integer (FK → Frame), nullable	anchors the event to a specific frame for replay/debugging
event_type	String	pass, shot, first_touch, turnover, press, etc.
player_id	Integer, nullable	primary actor (e.g. shooter, passer, presser)
related_player_id	Integer, nullable	added — e.g. pass target (player_to), turnover winner (won_by). v1's per-event-type JSON examples (player_from/player_to, lost_by/won_by) had no consistent column for this; without it, every event type needs a different query pattern.
team_id	String, nullable	team of player_id at time of event
pitch_x_m, pitch_y_m	Float, nullable	event location, pitch coordinates
homography_confidence	Float, nullable	added, same reasoning as PlayerTracking — an event's location is only as trustworthy as the homography that produced it
timestamp	Float	seconds from match start
metadata	String name, JSON type	unified field name (was metadata in one doc, extra_data in the other — standardized on metadata)
PlayerMetric

Analysis engine output, one row per player per metric per match. This table's shape is fixed exactly to the standard output contract defined in Analysis Logic Design v3 §0.3 — do not add ad hoc fields per metric type; everything metric-specific goes in sub_scores.

FIXED (gap #3): v1's PlayerMetric was missing sample_size, computed_at, and schema_version, all three of which v3 §0.3 requires. Without sample_size, there is no way to distinguish a low_sample score from a fully-trusted one in the database — the frontend would have nothing to display for that distinction even though Analysis Logic Design v3 treats it as central to judge-facing credibility.

Field	Type	Notes
metric_id	UUID (PK)	added — v1 had no primary key
match_id	UUID (FK → Match)	added — v1's PlayerMetric had no match scoping, which breaks multi-match player history (needed for ACWR in Injury Risk Phase 2)
player_id	Integer	
metric_name	String	e.g. first_touch_score, press_resistance_score, injury_risk_score
value	Float, nullable	important: per v3 §0.6/§0.7, a score can legitimately be null (e.g. zero pressured possessions) — this must not default to 0, which would misrepresent "not measured" as "measured and bad"
method	String	ml_trained | deterministic | heuristic_proxy
confidence	String	normal | low_sample | low_upstream_confidence
sample_size	Integer	added (fixes gap #3)
sub_scores	JSON	component breakdown, e.g. {"control": 76.0, "retention": 100.0, ...}
computed_at	DateTime	added (fixes gap #3)
schema_version	String	added (fixes gap #3) — e.g. "v3", so formula changes don't silently corrupt historical comparisons
TeamMetric

Team-level analysis output, one row per team per metric per match.

FIXED (gap #4 and #5): v1 had confidence: Float here but confidence: String on PlayerMetric — an inconsistent type for conceptually the same field. v1's TeamMetric was also missing method and sub_scores entirely, even though Analysis Logic Design v3 explicitly requires every score — including Team Rating and xG — to declare its method for the judge-facing transparency split (ML-trained vs. deterministic vs. heuristic). TeamMetric is now structurally identical to PlayerMetric except player_id → team_id, which also makes the DB layer and API serialization code shareable between the two instead of duplicated.

Field	Type	Notes
metric_id	UUID (PK)	added
match_id	UUID (FK → Match)	added
team_id	String	
metric_name	String	e.g. formation, possession, team_rating
value	Float, nullable	
method	String	added (fixes gap #5) — ml_trained | deterministic | heuristic_proxy
confidence	String	FIXED (gap #4): was Float in v1, now String enum, matching PlayerMetric
confidence_score	Float, nullable	added — if a raw 0–1 confidence number is also needed (e.g. Formation Detection's exp(-distance_score/...) value from Analysis Logic Design v3 §3), it lives here separately from the categorical confidence enum, so neither use case forces the other into the wrong type
sample_size	Integer	added, for consistency with PlayerMetric
sub_scores	JSON	added (fixes gap #5)
computed_at	DateTime	
schema_version	String	
Relationships (updated)
Match
 ├─→ Frame
 │     ├─→ PlayerDetection  (team_id, team_assignment_confidence)
 │     └─→ BallDetection
 ├─→ PlayerTracking  (team_id, pitch_x_m/y_m, homography_confidence)
 ├─→ Event  (team_id, pitch_x_m/y_m, homography_confidence, related_player_id)
 ├─→ PlayerMetric  (method, confidence, sample_size, sub_scores, schema_version)
 └─→ TeamMetric    (method, confidence, confidence_score, sample_size, sub_scores, schema_version)

team_id and homography_confidence now flow through every table from PlayerDetection onward, so any downstream consumer (Analysis Logic Design v3's scoring functions) can resolve confidence without a separate lookup.

Summary of Fixes (traceability)
#	Conflict	Resolution
1	Event.metadata vs Event.extra_data, match_id present in only one version	Unified: match_id kept, JSON field named metadata
2	No team_id or homography_confidence anywhere in detection/tracking tables	Added to PlayerDetection, PlayerTracking, and Event
3	PlayerMetric missing sample_size, computed_at, schema_version	Added, matching Analysis Logic Design v3 §0.3 output contract exactly
4	TeamMetric.confidence: Float vs PlayerMetric.confidence: String	TeamMetric.confidence changed to String enum; raw float moved to new confidence_score field
5	TeamMetric missing method and sub_scores	Added, so Team Rating/xG can declare method for judge transparency same as player-level metrics

Pre-Match Health Intelligence

Two tables backing the self-reported pre-match questionnaire → deterministic readiness scoring flow (ai/performance_ai/match_readiness_predictor/, served by backend/api/prematch_health.py). These produce a performance-readiness estimate, not a medical assessment.

They are split on the same principle as ProcessingJob → AnalysisResult: what the player submitted is stored separately from what was computed from it, so every score stays auditable against the untouched answers it came from, and a formula change (schema_version) can be re-run over historical submissions.

IMPORTANT: player_id on both tables is a String, and is NOT the Integer player_id used by PlayerDetection/PlayerTracking/Event/PlayerMetric. Those are ByteTrack tracking IDs scoped to a single processed video and carry no cross-match identity. A pre-match questionnaire is submitted before any tracking exists, so it uses a plain external identifier supplied by the caller. The two ID spaces are unrelated and must never be joined.

PreMatchQuestionnaire

The raw self-report, exactly as submitted and validated.

Field	Type	Notes
id	UUID (PK)	
player_id	String, indexed	external identifier — see the note above on why this is not the tracking-ID player_id
match_id	UUID (FK → Match), nullable, indexed	nullable by design, not as a workaround: Match rows are only created at video-upload time, so a genuine pre-match questionnaire routinely has no match yet. "Not linked yet" is a normal state here
questionnaire_json	JSON	all ~19 validated answers as one blob — same justification as AnalysisResult.result_json and Video.metadata_json: write-once, always read back whole, nothing queries an individual answer. The fields that ARE queried are real indexed columns
submitted_at	DateTime, indexed	indexed because "latest submission for this player" is the module's most common read

PreMatchHealthAssessment

The computed output for exactly one questionnaire (1:1, unique FK + cascade delete, exactly like AnalysisResult).

Field	Type	Notes
id	UUID (PK)	
questionnaire_id	UUID (FK → PreMatchQuestionnaire), unique, indexed	unique so the 1:1 is enforced by the database, not only by uselist=False
player_id	String, indexed	denormalized from the questionnaire so "latest for this player" is a single-table lookup, same denormalization PlayerTracking.match_id already uses
match_id	UUID (FK → Match), nullable, indexed	denormalized, same reason
submission_index	Integer, indexed	1-based per player, assigned at write time. This is what /latest and the history ordering sort on — NOT computed_at. datetime.utcnow() inherits the OS clock granularity (~15ms on Windows), so two submissions milliseconds apart get byte-identical timestamps and "most recent" becomes whatever the database returns first — which a player resubmitting to fix a typo would hit immediately. An explicit counter makes the order exact regardless of clock resolution, and doubles as a human-meaningful "submission #N"
features	JSON	the full normalized 0–100 feature vector (inputs + the three headline scores), stored so a future ML/DL model trains on exactly what the heuristic saw
physical_readiness	Float	0–100 headline score
fatigue_score	Float	0–100, higher = more fatigued
recovery_score	Float	0–100
performance_risk	String enum	low | moderate | high — banded on physical_readiness
workload_risk	String enum	low | moderate | high — banded on recent training load, independent of how ready the player feels
factors	JSON	per-dimension positive/negative/neutral classification with the sub-score behind each. This is the explainability record: the reasoning is stored alongside the headline number rather than discarded once the score is computed
method	String	reuses MetricMethod — heuristic_proxy today, ml_trained if a real model replaces HeuristicReadinessScorer behind the same interface
schema_version	String	e.g. "v1" — assessments computed under different formula constants are not comparable, same role as PlayerMetric.schema_version
computed_at	DateTime, indexed	when the assessment was computed. Answers "when", not "in what order" — see submission_index

NOTE: there is deliberately no confidence column here, unlike PlayerMetric/TeamMetric. MetricConfidence's three states are all about upstream measurement quality — low_sample (too few tracked events) and low_upstream_confidence (poor homography/tracking) — and neither can occur for a single complete, validated self-report. Storing "normal" on every row would be a constant dressed up as a measurement. What genuinely varies (which scoring tier ran) is already carried by method; the inherent limitation of self-reported data is stated in the API response and the UI copy instead.

Relationships
Match
 ├─→ (existing children, all cascade delete-orphan)
 ├─→ PreMatchQuestionnaire  (nullable link, NOT cascaded)
 │     └─→ PreMatchHealthAssessment  (1:1, cascade delete-orphan)
 └─→ PreMatchHealthAssessment  (nullable link, NOT cascaded)

The two pre-match links are intentionally not cascade="all, delete-orphan" while every other Match child is. The others are derived from the match's video and are meaningless without it; a questionnaire is the player's own self-report, valid on its own and often submitted before any match exists. Deleting a Match nullifies the FK instead, leaving the questionnaire and its assessment intact but unlinked.

<<<<<<< HEAD
This document supersedes docs/database_schema.md, docs/tracking_system_design.md's §9 schema section, and docs/event_detection.md's "Database Events Table" section. Those docs' non-schema content (algorithm choice, pipeline stages, module structure) is still valid and unaffected.
=======
This document supersedes docs/database_schema.md, docs/tracking_system_design.md's §9 schema section, and docs/event_detection.md's "Database Events Table" section. Those docs' non-schema content (algorithm choice, pipeline stages, module structure) is still valid and unaffected.
>>>>>>> d5cb07546c131118fecba674c2802f3eeb586b11
