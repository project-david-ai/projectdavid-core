
/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;
DROP TABLE IF EXISTS `run_execution_epochs`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `run_execution_epochs` (
  `id` varchar(64) NOT NULL,
  `run_id` varchar(64) NOT NULL,
  `root_run_id` varchar(64) NOT NULL,
  `owner_user_id` varchar(64) NOT NULL,
  `epoch_number` int NOT NULL,
  `state` varchar(32) NOT NULL,
  `created_at` bigint NOT NULL,
  `started_at` bigint DEFAULT NULL,
  `last_seen_at` bigint DEFAULT NULL,
  `ended_at` bigint DEFAULT NULL,
  `terminal_status` varchar(32) DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_exec_epoch_run_number` (`run_id`,`epoch_number`),
  KEY `ix_exec_epoch_root_created` (`root_run_id`,`created_at`),
  KEY `ix_exec_epoch_owner_root` (`owner_user_id`,`root_run_id`),
  KEY `ix_exec_epoch_state_seen` (`state`,`last_seen_at`),
  CONSTRAINT `fk_exec_epoch_root` FOREIGN KEY (`root_run_id`) REFERENCES `runs` (`id`) ON DELETE RESTRICT,
  CONSTRAINT `fk_exec_epoch_run` FOREIGN KEY (`run_id`) REFERENCES `runs` (`id`) ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
DROP TABLE IF EXISTS `execution_actors`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `execution_actors` (
  `id` varchar(64) NOT NULL,
  `epoch_id` varchar(64) NOT NULL,
  `slot` varchar(48) NOT NULL,
  `execution_role` varchar(48) NOT NULL,
  `assistant_id_snapshot` varchar(64) DEFAULT NULL,
  `model_snapshot` varchar(128) DEFAULT NULL,
  `ephemeral` tinyint(1) NOT NULL,
  `state` varchar(32) NOT NULL,
  `reserved_at` bigint NOT NULL,
  `activated_at` bigint DEFAULT NULL,
  `retired_at` bigint DEFAULT NULL,
  `last_seen_at` bigint DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_exec_actor_epoch_slot` (`epoch_id`,`slot`),
  KEY `ix_exec_actor_assistant` (`assistant_id_snapshot`),
  KEY `ix_exec_actor_state_seen` (`state`,`last_seen_at`),
  CONSTRAINT `fk_exec_actor_epoch` FOREIGN KEY (`epoch_id`) REFERENCES `run_execution_epochs` (`id`) ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
DROP TABLE IF EXISTS `run_delegations`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `run_delegations` (
  `id` varchar(64) NOT NULL,
  `parent_epoch_id` varchar(64) NOT NULL,
  `child_epoch_id` varchar(64) NOT NULL,
  `parent_action_id` varchar(64) NOT NULL,
  `initiating_actor_id` varchar(64) DEFAULT NULL,
  `created_at` bigint NOT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uq_run_delegation_child_epoch` (`child_epoch_id`),
  KEY `fk_delegation_actor` (`initiating_actor_id`),
  KEY `ix_run_delegation_parent` (`parent_epoch_id`),
  KEY `ix_run_delegation_parent_action` (`parent_action_id`),
  CONSTRAINT `fk_delegation_action` FOREIGN KEY (`parent_action_id`) REFERENCES `actions` (`id`) ON DELETE RESTRICT,
  CONSTRAINT `fk_delegation_actor` FOREIGN KEY (`initiating_actor_id`) REFERENCES `execution_actors` (`id`) ON DELETE RESTRICT,
  CONSTRAINT `fk_delegation_child` FOREIGN KEY (`child_epoch_id`) REFERENCES `run_execution_epochs` (`id`) ON DELETE RESTRICT,
  CONSTRAINT `fk_delegation_parent` FOREIGN KEY (`parent_epoch_id`) REFERENCES `run_execution_epochs` (`id`) ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
DROP TABLE IF EXISTS `execution_events`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `execution_events` (
  `id` bigint NOT NULL AUTO_INCREMENT,
  `epoch_id` varchar(64) NOT NULL,
  `actor_id` varchar(64) DEFAULT NULL,
  `origin_run_id` varchar(64) NOT NULL,
  `event_type` varchar(64) NOT NULL,
  `schema_version` int NOT NULL,
  `evidence_class` varchar(24) NOT NULL,
  `occurred_at_us` bigint NOT NULL,
  `recorded_at_us` bigint NOT NULL,
  `evidence_expires_at` bigint DEFAULT NULL,
  `action_id_snapshot` varchar(64) DEFAULT NULL,
  `tool_call_id_snapshot` varchar(64) DEFAULT NULL,
  `details` json NOT NULL,
  `evidence_ref` varchar(255) DEFAULT NULL,
  `evidence_sha256` varchar(64) DEFAULT NULL,
  `previous_digest` varchar(64) DEFAULT NULL,
  `record_digest` varchar(64) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `fk_exec_event_actor` (`actor_id`),
  KEY `ix_exec_event_epoch_id` (`epoch_id`,`id`),
  KEY `ix_exec_event_origin_time` (`origin_run_id`,`occurred_at_us`),
  KEY `ix_exec_event_type_time` (`event_type`,`occurred_at_us`),
  CONSTRAINT `fk_exec_event_actor` FOREIGN KEY (`actor_id`) REFERENCES `execution_actors` (`id`) ON DELETE RESTRICT,
  CONSTRAINT `fk_exec_event_epoch` FOREIGN KEY (`epoch_id`) REFERENCES `run_execution_epochs` (`id`) ON DELETE RESTRICT,
  CONSTRAINT `fk_exec_event_run` FOREIGN KEY (`origin_run_id`) REFERENCES `runs` (`id`) ON DELETE RESTRICT
) ENGINE=InnoDB AUTO_INCREMENT=263 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
DROP TABLE IF EXISTS `evidence_retention_holds`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `evidence_retention_holds` (
  `id` varchar(64) NOT NULL,
  `root_run_id` varchar(64) NOT NULL,
  `case_reference` varchar(128) NOT NULL,
  `reason_code` varchar(48) NOT NULL,
  `authorised_by` varchar(64) NOT NULL,
  `created_at` bigint NOT NULL,
  `reviewed_at` bigint DEFAULT NULL,
  `expires_at` bigint DEFAULT NULL,
  `released_at` bigint DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `ix_evidence_hold_root_released` (`root_run_id`,`released_at`),
  KEY `ix_evidence_hold_case` (`case_reference`),
  CONSTRAINT `fk_evidence_hold_root` FOREIGN KEY (`root_run_id`) REFERENCES `runs` (`id`) ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;
