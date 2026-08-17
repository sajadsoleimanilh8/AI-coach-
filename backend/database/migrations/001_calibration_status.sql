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
