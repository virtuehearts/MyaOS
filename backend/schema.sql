-- MySQL schema for memory storage with source tagging and Virtueism markers.

CREATE TABLE memories (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    memory_id VARCHAR(32) NOT NULL UNIQUE,
    user_id VARCHAR(64) NOT NULL,
    content TEXT NOT NULL,
    salience DECIMAL(4, 3) NOT NULL DEFAULT 0.000,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NULL DEFAULT NULL,
    INDEX idx_memories_user_created (user_id, created_at)
);

CREATE TABLE memory_tags (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    memory_id BIGINT UNSIGNED NOT NULL,
    tag VARCHAR(128) NOT NULL,
    UNIQUE KEY uniq_memory_tag (memory_id, tag),
    CONSTRAINT fk_memory_tags_memory
        FOREIGN KEY (memory_id) REFERENCES memories(id)
        ON DELETE CASCADE
);

CREATE TABLE memory_sources (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    memory_id BIGINT UNSIGNED NOT NULL,
    source_tag VARCHAR(128) NOT NULL,
    UNIQUE KEY uniq_memory_source (memory_id, source_tag),
    CONSTRAINT fk_memory_sources_memory
        FOREIGN KEY (memory_id) REFERENCES memories(id)
        ON DELETE CASCADE
);

CREATE TABLE memory_virtue_markers (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    memory_id BIGINT UNSIGNED NOT NULL,
    virtue_name VARCHAR(128) NOT NULL,
    virtue_score DECIMAL(4, 3) NOT NULL DEFAULT 0.000,
    UNIQUE KEY uniq_memory_virtue (memory_id, virtue_name),
    CONSTRAINT fk_memory_virtues_memory
        FOREIGN KEY (memory_id) REFERENCES memories(id)
        ON DELETE CASCADE
);

CREATE TABLE personality_trait_history (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    snapshot_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    openness DECIMAL(4, 3) NOT NULL DEFAULT 0.500,
    conscientiousness DECIMAL(4, 3) NOT NULL DEFAULT 0.500,
    extraversion DECIMAL(4, 3) NOT NULL DEFAULT 0.500,
    agreeableness DECIMAL(4, 3) NOT NULL DEFAULT 0.500,
    neuroticism DECIMAL(4, 3) NOT NULL DEFAULT 0.500,
    signal_summary JSON NULL,
    INDEX idx_trait_history_user_snapshot (user_id, snapshot_at)
);

CREATE TABLE reflection_logs (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    reflection_id VARCHAR(32) NOT NULL UNIQUE,
    user_id VARCHAR(64) NOT NULL,
    event_type VARCHAR(64) NOT NULL,
    content TEXT NOT NULL,
    context JSON NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_reflection_user_created (user_id, created_at)
);

CREATE TABLE persona_states (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL UNIQUE,
    emotion_state JSON NOT NULL,
    emotion_baseline JSON NOT NULL,
    routine_phase VARCHAR(64) NOT NULL DEFAULT 'unspecified',
    emotion_transitions JSON NULL,
    personality_traits JSON NOT NULL,
    goals JSON NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE conversation_messages (
    id BIGINT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL,
    session_id VARCHAR(64) NOT NULL,
    role VARCHAR(32) NOT NULL,
    content TEXT NOT NULL,
    topic_tags JSON NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NULL DEFAULT NULL,
    INDEX idx_conversation_user_session (user_id, session_id, created_at),
    INDEX idx_conversation_user_created (user_id, created_at)
);
