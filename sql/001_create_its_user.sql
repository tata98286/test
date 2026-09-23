CREATE DATABASE IF NOT EXISTS `its`
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_0900_ai_ci;

USE `its`;

CREATE TABLE IF NOT EXISTS `user` (
    `user_id` BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
    `username` VARCHAR(100) NOT NULL COMMENT 'Unique login ID',
    `password_hash` VARCHAR(255) NOT NULL COMMENT 'Application-generated password hash; never plaintext',
    `role` VARCHAR(30) NOT NULL DEFAULT 'USER' COMMENT 'Authorization role, e.g. USER or ADMIN',
    `created_at` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`user_id`),
    UNIQUE KEY `uk_user_username` (`username`)
) ENGINE=InnoDB
  DEFAULT CHARACTER SET utf8mb4
  COLLATE utf8mb4_0900_ai_ci;
