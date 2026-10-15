-- Enable Required Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Runtime connection settings (e.g. SET hnsw.iterative_scan = strict_order)
-- and versioned upgrades are applied by memory_engine/db.py.

-- ============================================================================
-- 1. DEVELOPER & CODING AGENT (Workspace Symbol Memory)
--    Primitives: halfvec(1536) HNSW + pg_trgm GIN + Reciprocal Rank Fusion
-- ============================================================================
CREATE TABLE dev_code_symbols (
    symbol_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id VARCHAR(100) NOT NULL,
    git_branch VARCHAR(100) NOT NULL,
    file_path TEXT NOT NULL,
    symbol_name VARCHAR(255) NOT NULL,
    symbol_type VARCHAR(50) NOT NULL,
    signature TEXT NOT NULL,
    code_content TEXT NOT NULL,
    embedding halfvec(1536),
    created_at TIMESTAMPTZ DEFAULT clock_timestamp(),
    UNIQUE (project_id, git_branch, file_path, symbol_name, symbol_type)
);

CREATE INDEX dev_scope ON dev_code_symbols (project_id, git_branch);
CREATE INDEX idx_dev_symbols_trgm ON dev_code_symbols USING gin (symbol_name gin_trgm_ops);
CREATE INDEX idx_dev_symbols_hnsw ON dev_code_symbols USING hnsw (embedding halfvec_cosine_ops);

-- ============================================================================
-- 2. AUTONOMOUS WORKFLOW AGENT (Episodic Trajectory Memory)
--    Primitives: halfvec(1536) HNSW + success_score quality gate + JSONB actions
-- ============================================================================
CREATE TABLE task_trajectories (
    trajectory_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(100) NOT NULL,
    goal_description TEXT NOT NULL,
    goal_embedding halfvec(1536),
    action_sequence JSONB NOT NULL,
    execution_result TEXT NOT NULL,
    success_score REAL NOT NULL CHECK (success_score BETWEEN 0 AND 1),
    created_at TIMESTAMPTZ DEFAULT clock_timestamp()
);

CREATE INDEX idx_task_goal_hnsw ON task_trajectories USING hnsw (goal_embedding halfvec_cosine_ops);

-- ============================================================================
-- 3. ENTERPRISE KNOWLEDGE AGENT (Policy & Audit Memory)
--    Primitives: halfvec(1536) HNSW + tsvector GIN + RRF + RBAC + versioning
-- ============================================================================
CREATE TABLE enterprise_documents (
    doc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_title VARCHAR(255) NOT NULL,
    allowed_role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', doc_title || ' ' || content)) STORED,
    embedding halfvec(1536),
    valid_from TIMESTAMPTZ DEFAULT clock_timestamp(),
    valid_until TIMESTAMPTZ NULL,
    status VARCHAR(20) DEFAULT 'ACTIVE'
);

CREATE UNIQUE INDEX enterprise_active_document ON enterprise_documents (doc_title, allowed_role) WHERE status = 'ACTIVE';
CREATE INDEX idx_ent_doc_hnsw ON enterprise_documents USING hnsw (embedding halfvec_cosine_ops);
CREATE INDEX idx_ent_doc_tsv ON enterprise_documents USING gin (tsv);

-- ============================================================================
-- 4. ADAPTIVE TUTOR AGENT (Hierarchical Skill Tree & Forgetting-Curve Memory)
--    Primitives: Self-referential DAG + recursive CTE + Ebbinghaus decay math
-- ============================================================================
CREATE TABLE tutor_skills (
    skill_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    skill_name VARCHAR(100) NOT NULL UNIQUE,
    parent_skill_id UUID REFERENCES tutor_skills(skill_id)
);

CREATE TABLE tutor_user_progress (
    progress_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(100) NOT NULL,
    skill_id UUID REFERENCES tutor_skills(skill_id),
    proficiency_score REAL DEFAULT 0.0 CHECK (proficiency_score BETWEEN 0 AND 1),
    last_reviewed_at TIMESTAMPTZ DEFAULT clock_timestamp(),
    UNIQUE (user_id, skill_id)
);

-- ============================================================================
-- 5. MULTI-AGENT SWARM (Blackboard & Shared State Memory)
--    Primitives: FOR UPDATE SKIP LOCKED + lease tokens + expiring claims
-- ============================================================================
CREATE TABLE swarm_blackboard (
    task_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id VARCHAR(100) NOT NULL,
    task_name VARCHAR(100) NOT NULL,
    assigned_agent VARCHAR(100) NULL,
    status VARCHAR(50) DEFAULT 'PENDING',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    lease_token UUID NULL,
    lease_until TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ DEFAULT clock_timestamp(),
    updated_at TIMESTAMPTZ DEFAULT clock_timestamp()
);

CREATE INDEX swarm_pending ON swarm_blackboard (workflow_id, status, created_at);

-- ============================================================================
-- 6. RELATIONAL COMPANION AGENT (3-Subject Graph, Episodic & Ephemeral Memory)
--    Primitives: Versioned property graph + provenance + HNSW chunks + TTL
-- ============================================================================
CREATE TABLE companion_memory_state (
    user_id TEXT PRIMARY KEY,
    enabled BOOLEAN NOT NULL DEFAULT true,
    generation INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE companion_episodes (
    episode_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    ingestion_id TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT clock_timestamp(),
    UNIQUE (user_id, ingestion_id),
    UNIQUE (user_id, episode_id)
);

CREATE TABLE companion_chunks (
    chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id UUID REFERENCES companion_episodes(episode_id) ON DELETE CASCADE,
    user_id VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    embedding halfvec(1536),
    created_at TIMESTAMPTZ DEFAULT clock_timestamp(),
    CONSTRAINT chunk_episode_owner FOREIGN KEY (user_id, episode_id)
        REFERENCES companion_episodes(user_id, episode_id) ON DELETE CASCADE
);

CREATE TABLE companion_graph_nodes (
    node_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(100) NOT NULL,
    name VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    salience REAL DEFAULT 1.0,
    normalize_name_key VARCHAR(100) NOT NULL,
    subject VARCHAR(20) DEFAULT 'user',  -- 'user' | 'self' | 'shared'
    embedding halfvec(1536),
    is_subject BOOLEAN NOT NULL DEFAULT false,
    UNIQUE (user_id, name, entity_type, subject),
    UNIQUE (user_id, normalize_name_key, subject),
    UNIQUE (user_id, node_id)
);

CREATE TABLE companion_graph_edges (
    edge_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(100) NOT NULL,
    source_node_id UUID REFERENCES companion_graph_nodes(node_id),
    target_node_id UUID REFERENCES companion_graph_nodes(node_id),
    relationship_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'ACTIVE',
    valid_from TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    valid_until TIMESTAMPTZ NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    subject VARCHAR(20) DEFAULT 'user',       -- inherited from source node
    valence REAL DEFAULT 0.0,                  -- -1.0 (negative) .. +1.0 (positive); 0 = neutral
    intensity REAL DEFAULT 0.5,                -- 0.0 .. 1.0 emotional intensity
    source_episode_id UUID REFERENCES companion_episodes(episode_id),  -- provenance
    embedding halfvec(1536),
    CONSTRAINT edge_source_owner FOREIGN KEY (user_id, source_node_id)
        REFERENCES companion_graph_nodes(user_id, node_id),
    CONSTRAINT edge_target_owner FOREIGN KEY (user_id, target_node_id)
        REFERENCES companion_graph_nodes(user_id, node_id),
    CONSTRAINT edge_episode_owner FOREIGN KEY (user_id, source_episode_id)
        REFERENCES companion_episodes(user_id, episode_id)
);

CREATE TABLE companion_fact_sources (
    edge_id UUID REFERENCES companion_graph_edges(edge_id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    episode_id UUID NOT NULL,
    PRIMARY KEY (edge_id, episode_id),
    FOREIGN KEY (user_id, episode_id) REFERENCES companion_episodes(user_id, episode_id)
);

CREATE TABLE companion_ephemerals (
    state_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL
);

CREATE UNIQUE INDEX companion_active_edge
    ON companion_graph_edges (user_id, source_node_id, target_node_id, relationship_type, subject)
    WHERE status = 'ACTIVE';
CREATE INDEX companion_edge_owner ON companion_graph_edges (user_id, subject, status);
CREATE INDEX companion_edge_embedding ON companion_graph_edges USING hnsw (embedding halfvec_cosine_ops);
CREATE INDEX companion_node_embedding ON companion_graph_nodes USING hnsw (embedding halfvec_cosine_ops);
CREATE INDEX companion_ephemeral_owner ON companion_ephemerals (user_id, expires_at);
CREATE INDEX companion_chunk_owner ON companion_chunks (user_id);
CREATE INDEX idx_comp_chunks_hnsw ON companion_chunks USING hnsw (embedding halfvec_cosine_ops);
