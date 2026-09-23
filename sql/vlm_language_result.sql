CREATE TABLE IF NOT EXISTS vlm_language_result (
    event_id BIGINT UNSIGNED PRIMARY KEY,
    english_answer TEXT NULL,
    korean_translation TEXT NULL,
    translation_status VARCHAR(24) NOT NULL,
    request_history JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_vlm_language_event FOREIGN KEY(event_id)
        REFERENCES fire_event(event_id) ON DELETE CASCADE
);
GRANT SELECT, INSERT, UPDATE ON its.vlm_language_result TO 'its_web'@'localhost';
