# `ai/` — what is implemented, and what is a blueprint slot

This tree follows the platform blueprint, but only part of it is built.
Directories for unbuilt modules used to exist as empty folders holding a
`.gitkeep`: 58 of them, which made roughly two thirds of this tree look like
code that did not exist. They were removed, and this file is the record of
what the blueprint still reserves.

Nothing below is a promise of a delivery date. It is a map of the gap
between the blueprint and the code.

| Package | Implemented | Blueprint only (no code) |
|---|---|---|
| `analytics` | — | `dashboards`, `match_analytics`, `player_analytics`, `team_analytics` |
| `common` | `tests` | — |
| `computer_vision` | `pass_detection`, `player_tracking`, `pose_estimation`, `shot_detection`, `tactical_analysis` | `ball_tracking`, `body_orientation`, `heatmap_generation`, `press_detection`, `referee_detection` |
| `health_ai` | — | `bmi_engine`, `bmr_engine`, `calorie_estimation`, `food_detection`, `health_risk_analysis`, `nutrition_ai`, `sleep_analysis`, `wearable_sync` |
| `llm_coach` | — | `chat_assistant`, `match_explainer`, `report_generator`, `strategy_consultant`, `voice_coach` |
| `opponent_intelligence` | `counter_strategy_generator`, `weakness_map` | `key_player_detector`, `opponent_analyzer`, `set_piece_analysis` |
| `performance_ai` | `match_readiness_predictor`, `tests` | `burnout_detection`, `career_growth_predictor`, `fatigue_prediction`, `hydration_score`, `injury_prediction`, `performance_forecasting`, `readiness_score`, `recovery_analysis`, `recovery_recommendation`, `talent_prediction`, `transfer_value_prediction`, `travel_fatigue_analysis` |
| `player_intelligence` | `body_orientation_score`, `decision_making_score`, `defensive_positioning`, `finishing_efficiency_score`, `first_touch_score`, `off_ball_movement`, `passing_vision_score`, `press_resistance_score`, `scanning_behavior`, `tests` | — |
| `psychology_ai` | `confidence_score`, `focus_score`, `mental_readiness`, `motivation_score`, `pressure_index`, `stress_analysis`, `tests` | `emotional_state_analysis` |
| `recommendation_ai` | — | `injury_prevention_coach`, `lineup_generator`, `nutrition_recommender`, `real_time_tactical_assistant`, `recovery_optimizer`, `strategy_optimizer`, `training_recommender` |
| `simulation_ai` | `tests`, `what_if_analysis` | `digital_twin_player`, `lineup_simulator`, `match_outcome_simulator`, `tactical_simulator` |
| `sports_management_ai` | — | `contract_risk_analysis`, `salary_optimizer`, `squad_builder`, `transfer_advisor`, `youth_development` |
| `team_intelligence` | `formation_stability`, `pressing_structure_analysis`, `weak_zone_detection` | `communication_analysis`, `possession_analysis`, `team_chemistry_score`, `transition_analysis` |

To start one of the blueprint modules, create the package under the relevant
parent, follow the layout of an implemented sibling (for example
`ai/player_intelligence/decision_making_score/`), and return the standard
metric envelope from `ai/common/metrics.py`.
