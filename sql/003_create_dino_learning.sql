USE `its`;

CREATE TABLE IF NOT EXISTS `dino_training_sample` (
    `sample_id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `event_id` BIGINT UNSIGNED NOT NULL,
    `source_filename` VARCHAR(255) NOT NULL,
    `video_second` DECIMAL(10, 3) NOT NULL DEFAULT 0,
    `crop_path` VARCHAR(1024) NOT NULL,
    `yolo_label` VARCHAR(20) NULL,
    `yolo_conf` DECIMAL(6, 5) NULL,
    `dino_scores` JSON NOT NULL,
    `label` JSON NULL,
    `label_source` ENUM('vlm', 'human') NULL,
    `status` ENUM('PENDING', 'LABELED', 'USED') NOT NULL DEFAULT 'PENDING',
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    `labeled_at` TIMESTAMP NULL DEFAULT NULL,
    PRIMARY KEY (`sample_id`),
    KEY `idx_dino_training_event` (`event_id`),
    KEY `idx_dino_training_status` (`status`),
    CONSTRAINT `fk_dino_training_event`
        FOREIGN KEY (`event_id`) REFERENCES `fire_event` (`event_id`)
        ON DELETE CASCADE
) ENGINE=InnoDB
  DEFAULT CHARACTER SET utf8mb4
  COLLATE utf8mb4_0900_ai_ci;

CREATE TABLE IF NOT EXISTS `dino_model_version` (
    `version_id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `checkpoint_path` VARCHAR(512) NOT NULL,
    `num_mined` INT UNSIGNED NOT NULL,
    `num_anchor` INT UNSIGNED NOT NULL,
    `metrics` JSON NOT NULL,
    `deployed` BOOLEAN NOT NULL DEFAULT FALSE,
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`version_id`),
    UNIQUE KEY `uq_dino_model_checkpoint` (`checkpoint_path`),
    KEY `idx_dino_model_deployed` (`deployed`, `version_id`)
) ENGINE=InnoDB
  DEFAULT CHARACTER SET utf8mb4
  COLLATE utf8mb4_0900_ai_ci;

GRANT SELECT, INSERT, UPDATE, DELETE ON `its`.`dino_training_sample`
    TO 'its_web'@'localhost';
GRANT SELECT, INSERT, UPDATE, DELETE ON `its`.`dino_model_version`
    TO 'its_web'@'localhost';
