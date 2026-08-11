-- Enable Required Extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

SET hnsw.iterative_scan = strict_order;

-- ============================================================================
-- 1. DEVELOPER & CODING AGENT (Workspace Memory)
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
    created_at TIMESTAMPTZ DEFAULT clock_timestamp()
);

CREATE INDEX idx_dev_symbols_trgm ON dev_code_symbols USING gin (symbol_name gin_trgm_ops);
CREATE INDEX idx_dev_symbols_hnsw ON dev_code_symbols USING hnsw (embedding halfvec_cosine_ops);

-- ============================================================================
-- 2. AUTONOMOUS WORKFLOW AGENT (Trajectory Memory)
-- ============================================================================
CREATE TABLE task_trajectories (
    trajectory_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id VARCHAR(100) NOT NULL,
    goal_description TEXT NOT NULL,
    goal_embedding halfvec(1536),
    action_sequence JSONB NOT NULL,
    execution_result TEXT NOT NULL,
    success_score REAL NOT NULL,
    created_at TIMESTAMPTZ DEFAULT clock_timestamp()
);

CREATE INDEX idx_task_goal_hnsw ON task_trajectories USING hnsw (goal_embedding halfvec_cosine_ops);

-- ============================================================================
-- 3. ENTERPRISE KNOWLEDGE AGENT (Policy & Audit Memory)
-- ============================================================================
CREATE TABLE enterprise_documents (
    doc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_title VARCHAR(255) NOT NULL,
    allowed_role VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding halfvec(1536),
    valid_from TIMESTAMPTZ DEFAULT clock_timestamp(),
    valid_until TIMESTAMPTZ NULL,
    status VARCHAR(20) DEFAULT 'ACTIVE'
);

CREATE INDEX idx_ent_doc_hnsw ON enterprise_documents USING hnsw (embedding halfvec_cosine_ops);
CREATE INDEX idx_ent_doc_tsv ON enterprise_documents USING gin (tsv);

-- ============================================================================
-- 4. ADAPTIVE TUTOR AGENT (Skill Tree Memory)
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
    proficiency_score REAL DEFAULT 0.0,
    last_reviewed_at TIMESTAMPTZ DEFAULT clock_timestamp(),
    UNIQUE(user_id, skill_id)
);

-- ============================================================================
-- 5. MULTI-AGENT SWARM (Blackboard & Shared State Memory)
-- ============================================================================
CREATE TABLE swarm_blackboard (
    task_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workflow_id VARCHAR(100) NOT NULL,
    task_name VARCHAR(100) NOT NULL,
    assigned_agent VARCHAR(100) NULL,
    status VARCHAR(50) DEFAULT 'PENDING',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ DEFAULT clock_timestamp()
);

-- ============================================================================
-- 6. RELATIONAL COMPANION AGENT (Identity & Intimacy Memory)
-- ============================================================================
CREATE TABLE companion_episodes (
    episode_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT clock_timestamp()
);

CREATE TABLE companion_chunks (
    chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    episode_id UUID REFERENCES companion_episodes(episode_id) ON DELETE CASCADE,
    user_id VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    embedding halfvec(1536),
    created_at TIMESTAMPTZ DEFAULT clock_timestamp()
);

CREATE TABLE companion_graph_nodes (
    node_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(100) NOT NULL,
    name VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    UNIQUE(user_id, name, entity_type)
);

CREATE TABLE companion_graph_edges (
    edge_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(100) NOT NULL,
    source_node_id UUID REFERENCES companion_graph_nodes(node_id),
    target_node_id UUID REFERENCES companion_graph_nodes(node_id),
    relationship_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'ACTIVE',
    valid_until TIMESTAMPTZ NULL
);

CREATE TABLE companion_ephemerals (
    state_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(100) NOT NULL,
    description TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX idx_comp_chunks_hnsw ON companion_chunks USING hnsw (embedding halfvec_cosine_ops);
