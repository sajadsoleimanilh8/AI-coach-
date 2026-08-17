
---

## 6. Off-Ball Movement Score (§6)
**Reads:** `PlayerTracking` (this player's trajectory) + all opponents' trajectories for the same frames, restricted to frames where this player does NOT have the ball.
**Writes:** `PlayerMetric(metric_name="off_ball_movement_score", method="heuristic_proxy", ...)`.

### Formulas
```python
# Space creation (50%)
space_creation_score = 100 * clip(safe_ratio(avg_distance_gained_from_marker, MAX_USEFUL_SEPARATION_GAIN_M, default=0.0), 0, 1)

# Attacking-third presence (30%)
attacking_presence_score = 100 * safe_ratio(off_ball_frames_in_attacking_third, total_off_ball_frames_while_team_in_possession, default=None)

# Movement efficiency (20%)
efficiency_score = 100 * safe_ratio(net_displacement_m, total_path_length_m, default=None)

value = space_creation_score * 0.50 + attacking_presence_score * 0.30 + efficiency_score * 0.20
```

---

## 7. Defensive Positioning Score (§7)
**Reads:** `PlayerTracking` for this player + teammates with `team_id == this player's team_id` (gated on `team_assignment_confidence >= 0.5`).
**Writes:** `PlayerMetric(metric_name="defensive_positioning_score", method="heuristic_proxy", ...)`.

### Formulas
```python
# Line discipline (40%)
line_discipline_score = 100 * (1 - clip(avg_deviation_from_defensive_line_m / MAX_ACCEPTABLE_LINE_DEVIATION_M, 0, 1))

# Cover / Balance Spacing (35%)
spacing_score = 100 * gaussian_score(nearest_teammate_distance_m, ideal=IDEAL_DEFENSIVE_SPACING_M, tolerance=SPACING_TOLERANCE_M)

# Reaction to Threat (25%)
reaction_score = 100 * clip(safe_ratio(closing_speed_toward_threat_ms, REACTION_SPEED_REFERENCE_MS, default=None), 0, 1)

value = line_discipline_score * 0.40 + spacing_score * 0.35 + reaction_score * 0.25
```

---

## 8. Passing Vision Score (§8)
**Reads:** `Event` rows of type `pass`/`turnover` for this player (gated on `team_assignment_confidence >= 0.5` -- pass/turnover classification itself depends on real `team_id`: `detect_passes()` checks `team1 == team2`, which is trivially true when `team_id` is unset, so these counts are only meaningful once team assignment has run).
**Writes:** `PlayerMetric(metric_name="passing_vision_score", method="heuristic_proxy", ...)`.

### Formulas
```python
# Completion rate (40%)
completion_rate_score = 100 * safe_ratio(completed_passes, completed_passes + turnovers_lost)

# Angle diversity (30%) -- Shannon entropy over PASS_DIRECTION_SECTORS compass buckets, a proxy for
# "seeing multiple passing options" rather than always playing the same safe ball
angle_diversity_score = 100 * shannon_entropy(pass_sector_counts) / log(PASS_DIRECTION_SECTORS)

# Forward pass ratio (30%) -- fraction of passes progressing toward the opponent's goal,
# using the same attacking_direction assumption shot_heuristics.py already relies on
forward_pass_ratio_score = 100 * safe_ratio(forward_passes, completed_passes)

value = completion_rate_score * 0.40 + angle_diversity_score * 0.30 + forward_pass_ratio_score * 0.30
```

---

## 9. Decision Making Score (§9)
**Reads:** `Event` rows of type `pass`/`shot`/`turnover` for this player, plus per-touch decision time derived from the chronological possession-segment sequence built in `_detect_events()` (gated on `team_assignment_confidence >= 0.5`, since successful/lost action counts depend on real `team_id` the same way passing_vision_score's do -- `avg_decision_time` itself is team-agnostic, but the score is gated on the weaker input).
**Writes:** `PlayerMetric(metric_name="decision_making_score", method="heuristic_proxy", ...)`.

### Formulas
```python
# Retention (50%) -- successful_actions = this player's passes + shots; lost_actions = this player's turnovers
retention_score = 100 * safe_ratio(successful_actions, successful_actions + lost_actions)

# Decision speed (50%) -- same scale press_resistance_score already uses for its own decision_speed sub-score
decision_speed_score = 100 * clip((DECISION_TIME_MAX_S - avg_decision_time) / (DECISION_TIME_MAX_S - DECISION_TIME_MIN_S), 0, 1)

value = retention_score * 0.50 + decision_speed_score * 0.50
```

---

## 10. Finishing Efficiency Score (§10)
**Reads:** `Event` rows of type `shot` for this player (gated on `homography_confidence >= 0.6`, NOT `team_assignment_confidence` -- shot location/attribution doesn't depend on team_id). **This is location-based shot quality, NOT true conversion rate** -- no goal/miss/on-target outcome is observable anywhere in this pipeline's data; `projected_y_at_goal` is only ever used by `shot_heuristics.py` to decide whether a fast ball movement qualifies as a "shot" at all, never as a persisted result.
**Writes:** `PlayerMetric(metric_name="finishing_efficiency_score", method="heuristic_proxy", ...)`.

### Formulas
```python
# Distance to goal (35%) -- closer shots score higher, a standard xG-model input
distance_score = gaussian_score(distance_to_goal_m, ideal=0.0, tolerance=SHOT_DISTANCE_TOLERANCE_M)

# Shooting angle (35%) -- angle subtended by the goal mouth as seen from the shot location;
# wider angle = more of the goal is visible/reachable
angle_score = 100 * clip(angle_deg / MAX_REALISTIC_SHOOTING_ANGLE_DEG, 0, 1)

# Placement (30%) -- peaks at goal-mouth center, tapers toward the posts
placement_score = gaussian_score(projected_y_at_goal, ideal=goal_center_y, tolerance=GOAL_WIDTH_M / 2)

value = distance_score * 0.35 + angle_score * 0.35 + placement_score * 0.30
```
