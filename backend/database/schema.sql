CREATE TABLE IF NOT EXISTS videos (
    id VARCHAR(36) PRIMARY KEY,
    original_filename VARCHAR(255) NOT NULL,
    stored_filename VARCHAR(255) NOT NULL UNIQUE,
    content_type VARCHAR(120),
    file_size INTEGER NOT NULL,
    storage_path TEXT NOT NULL,
    metadata_json JSON,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS processing_jobs (
    id VARCHAR(36) PRIMARY KEY,
    video_id VARCHAR(36) NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    status VARCHAR(32) NOT NULL DEFAULT 'queued',
    progress INTEGER NOT NULL DEFAULT 0,
    message TEXT,
    error TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_processing_jobs_video_id
    ON processing_jobs(video_id);

CREATE TABLE IF NOT EXISTS analysis_results (
    id VARCHAR(36) PRIMARY KEY,
    job_id VARCHAR(36) NOT NULL UNIQUE REFERENCES processing_jobs(id) ON DELETE CASCADE,
    result_json JSON NOT NULL,
    summary TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_analysis_results_job_id
    ON analysis_results(job_id);

CREATE TABLE IF NOT EXISTS prematch_questionnaires (
    id VARCHAR(36) PRIMARY KEY,
    player_id VARCHAR(64) NOT NULL,
    match_id VARCHAR(36) REFERENCES matches(match_id),
    questionnaire_json JSON NOT NULL,
    submitted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_prematch_questionnaires_player_id
    ON prematch_questionnaires(player_id);

CREATE INDEX IF NOT EXISTS idx_prematch_questionnaires_match_id
    ON prematch_questionnaires(match_id);

CREATE INDEX IF NOT EXISTS idx_prematch_questionnaires_submitted_at
    ON prematch_questionnaires(submitted_at);

CREATE TABLE IF NOT EXISTS prematch_health_assessments (
    id VARCHAR(36) PRIMARY KEY,
    questionnaire_id VARCHAR(36) NOT NULL UNIQUE
        REFERENCES prematch_questionnaires(id) ON DELETE CASCADE,
    player_id VARCHAR(64) NOT NULL,
    match_id VARCHAR(36) REFERENCES matches(match_id),
    submission_index INTEGER NOT NULL DEFAULT 1,
    features JSON NOT NULL,
    physical_readiness FLOAT NOT NULL,
    fatigue_score FLOAT NOT NULL,
    recovery_score FLOAT NOT NULL,
    performance_risk VARCHAR(16) NOT NULL,
    workload_risk VARCHAR(16) NOT NULL,
    factors JSON NOT NULL,
    method VARCHAR(32) NOT NULL,
    schema_version VARCHAR(16) NOT NULL,
    computed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_prematch_health_assessments_player_id
    ON prematch_health_assessments(player_id);

CREATE INDEX IF NOT EXISTS idx_prematch_health_assessments_match_id
    ON prematch_health_assessments(match_id);

CREATE INDEX IF NOT EXISTS idx_prematch_health_assessments_computed_at
    ON prematch_health_assessments(computed_at);

CREATE INDEX IF NOT EXISTS idx_prematch_health_assessments_submission_index
    ON prematch_health_assessments(submission_index);

CREATE TABLE IF NOT EXISTS psychology_assessments (
    assessment_id VARCHAR(36) PRIMARY KEY,
    player_id VARCHAR(64) NOT NULL,
    cv_player_id INTEGER,
    match_id VARCHAR(36) REFERENCES matches(match_id),
    responses JSON NOT NULL,
    features JSON NOT NULL,
    mental_readiness FLOAT NOT NULL,
    focus FLOAT NOT NULL,
    confidence FLOAT NOT NULL,
    stress FLOAT NOT NULL,
    pressure_risk VARCHAR(16) NOT NULL,
    mental_performance_risk VARCHAR(16) NOT NULL,
    factors JSON NOT NULL,
    sub_scores JSON NOT NULL,
    method VARCHAR(32) NOT NULL,
    confidence_level VARCHAR(32) NOT NULL,
    schema_version VARCHAR(16) NOT NULL,
    submission_index INTEGER NOT NULL DEFAULT 1,
    submitted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    computed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_psychology_assessments_player_id
    ON psychology_assessments(player_id);

CREATE INDEX IF NOT EXISTS idx_psychology_assessments_match_id
    ON psychology_assessments(match_id);

CREATE INDEX IF NOT EXISTS idx_psychology_assessments_submitted_at
    ON psychology_assessments(submitted_at);

CREATE INDEX IF NOT EXISTS idx_psychology_assessments_computed_at
    ON psychology_assessments(computed_at);

CREATE INDEX IF NOT EXISTS idx_psychology_assessments_submission_index
    ON psychology_assessments(submission_index);

CREATE TABLE IF NOT EXISTS calibration_status (
    calibration_id VARCHAR(36) PRIMARY KEY,
    match_id VARCHAR(36) NOT NULL REFERENCES matches(match_id),
    frame_start INTEGER NOT NULL,
    frame_end INTEGER NOT NULL,
    valid BOOLEAN NOT NULL,
    invalid_reason TEXT,
    confidence FLOAT NOT NULL,
    reprojection_error_m FLOAT,
    n_points INTEGER NOT NULL DEFAULT 0,
    source VARCHAR(16) NOT NULL,
    solved_on_frame INTEGER,
    camera_motion VARCHAR(16),
    camera_shift_px FLOAT,
    homography_matrix JSON,
    method VARCHAR(32) NOT NULL,
    computed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_calibration_status_match_id
    ON calibration_status(match_id);

CREATE INDEX IF NOT EXISTS idx_calibration_status_match_frames
    ON calibration_status(match_id, frame_start, frame_end);

CREATE INDEX IF NOT EXISTS idx_calibration_status_valid
    ON calibration_status(valid);
