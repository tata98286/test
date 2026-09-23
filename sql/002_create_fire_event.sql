USE `its`;

CREATE TABLE IF NOT EXISTS `fire_event` (
    `event_id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `detected_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `status` ENUM('UNRECEIVED', 'RECEIVED') NOT NULL DEFAULT 'UNRECEIVED',
    `accepted_at` TIMESTAMP NULL DEFAULT NULL,
    `accepted_by` BIGINT UNSIGNED NULL,
    `source_filename` VARCHAR(255) NULL,
    `yolo_confidence` DECIMAL(5, 2) NULL,
    `vlm_result` VARCHAR(30) NULL,
    `vlm_answer` VARCHAR(500) NULL,
    `vlm_prompt_version` VARCHAR(64) NULL,
    `vlm_prompt` MEDIUMTEXT NULL,
    `event_hashes` VARCHAR(255) NULL,
    `evidence_names` JSON NULL,
    PRIMARY KEY (`event_id`),
    KEY `idx_fire_event_detected_at` (`detected_at`),
    KEY `idx_fire_event_status` (`status`),
    CONSTRAINT `fk_fire_event_accepted_by`
        FOREIGN KEY (`accepted_by`) REFERENCES `user` (`user_id`)
        ON DELETE SET NULL
) ENGINE=InnoDB
  DEFAULT CHARACTER SET utf8mb4
  COLLATE utf8mb4_0900_ai_ci;
