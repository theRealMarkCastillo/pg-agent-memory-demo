-- Idempotent upgrade migration for databases created prior to schema consolidation.
-- Fresh databases initialized from postgres/init.sql already include all objects below.

CREATE TABLE IF NOT EXISTS companion_memory_state (
    user_id text PRIMARY KEY,
    enabled boolean NOT NULL DEFAULT true,
    generation integer NOT NULL DEFAULT 0
);

ALTER TABLE companion_episodes ADD COLUMN IF NOT EXISTS ingestion_id text;
UPDATE companion_episodes SET ingestion_id=episode_id::text WHERE ingestion_id IS NULL;
ALTER TABLE companion_episodes ALTER COLUMN ingestion_id SET NOT NULL;

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid='companion_episodes'::regclass AND contype='u'
          AND pg_get_constraintdef(oid) LIKE '%(user_id, ingestion_id)%'
    ) THEN
        ALTER TABLE companion_episodes ADD UNIQUE(user_id, ingestion_id);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid='companion_episodes'::regclass AND contype='u'
          AND pg_get_constraintdef(oid) LIKE '%(user_id, episode_id)%'
    ) THEN
        ALTER TABLE companion_episodes ADD UNIQUE(user_id, episode_id);
    END IF;
END $$;

ALTER TABLE companion_graph_nodes ADD COLUMN IF NOT EXISTS embedding halfvec(1536);
ALTER TABLE companion_graph_nodes ADD COLUMN IF NOT EXISTS is_subject boolean NOT NULL DEFAULT false;
UPDATE companion_graph_nodes n SET is_subject=true
WHERE EXISTS(SELECT 1 FROM companion_graph_edges e WHERE e.source_node_id=n.node_id)
   OR NOT EXISTS(SELECT 1 FROM companion_graph_edges e WHERE e.target_node_id=n.node_id);

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid='companion_graph_nodes'::regclass AND contype='u'
          AND pg_get_constraintdef(oid) LIKE '%(user_id, node_id)%'
    ) THEN
        ALTER TABLE companion_graph_nodes ADD UNIQUE(user_id, node_id);
    END IF;
END $$;

ALTER TABLE companion_graph_edges ADD COLUMN IF NOT EXISTS valid_from timestamptz NOT NULL DEFAULT clock_timestamp();
ALTER TABLE companion_graph_edges ADD COLUMN IF NOT EXISTS recorded_at timestamptz NOT NULL DEFAULT clock_timestamp();
ALTER TABLE companion_graph_edges ADD COLUMN IF NOT EXISTS embedding halfvec(1536);

DO $$ DECLARE c record; BEGIN
    FOR c IN SELECT conname FROM pg_constraint WHERE conrelid='companion_graph_edges'::regclass AND contype='u' LOOP
        EXECUTE format('ALTER TABLE companion_graph_edges DROP CONSTRAINT %I', c.conname);
    END LOOP;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS companion_active_edge
    ON companion_graph_edges(user_id, source_node_id, target_node_id, relationship_type, subject)
    WHERE status='ACTIVE';

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='edge_source_owner') THEN
        ALTER TABLE companion_graph_edges ADD CONSTRAINT edge_source_owner
            FOREIGN KEY(user_id, source_node_id) REFERENCES companion_graph_nodes(user_id,node_id);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='edge_target_owner') THEN
        ALTER TABLE companion_graph_edges ADD CONSTRAINT edge_target_owner
            FOREIGN KEY(user_id, target_node_id) REFERENCES companion_graph_nodes(user_id,node_id);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='edge_episode_owner') THEN
        ALTER TABLE companion_graph_edges ADD CONSTRAINT edge_episode_owner
            FOREIGN KEY(user_id,source_episode_id) REFERENCES companion_episodes(user_id,episode_id);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='chunk_episode_owner') THEN
        ALTER TABLE companion_chunks ADD CONSTRAINT chunk_episode_owner
            FOREIGN KEY(user_id,episode_id) REFERENCES companion_episodes(user_id,episode_id) ON DELETE CASCADE;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS companion_fact_sources (
    edge_id uuid REFERENCES companion_graph_edges(edge_id) ON DELETE CASCADE,
    user_id text NOT NULL,
    episode_id uuid NOT NULL,
    PRIMARY KEY(edge_id,episode_id),
    FOREIGN KEY(user_id,episode_id) REFERENCES companion_episodes(user_id,episode_id)
);
INSERT INTO companion_fact_sources
SELECT edge_id,user_id,source_episode_id FROM companion_graph_edges
WHERE source_episode_id IS NOT NULL
ON CONFLICT DO NOTHING;

CREATE INDEX IF NOT EXISTS companion_edge_owner ON companion_graph_edges(user_id,subject,status);
CREATE INDEX IF NOT EXISTS companion_edge_embedding ON companion_graph_edges USING hnsw(embedding halfvec_cosine_ops);
CREATE INDEX IF NOT EXISTS companion_node_embedding ON companion_graph_nodes USING hnsw(embedding halfvec_cosine_ops);
CREATE INDEX IF NOT EXISTS companion_ephemeral_owner ON companion_ephemerals(user_id,expires_at);
CREATE INDEX IF NOT EXISTS companion_chunk_owner ON companion_chunks(user_id);

-- Preserve duplicate source rows before reconciling the current symbol projection on legacy volumes.
CREATE TABLE IF NOT EXISTS migration_duplicate_symbols AS SELECT * FROM dev_code_symbols WHERE symbol_id IN (
    SELECT symbol_id FROM (
        SELECT symbol_id,row_number() OVER(
            PARTITION BY project_id,git_branch,file_path,symbol_name,symbol_type
            ORDER BY created_at DESC,symbol_id DESC
        ) AS rn FROM dev_code_symbols
    ) d WHERE rn>1
);
DELETE FROM dev_code_symbols WHERE symbol_id IN (SELECT symbol_id FROM migration_duplicate_symbols);

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid='dev_code_symbols'::regclass AND contype='u'
    ) THEN
        ALTER TABLE dev_code_symbols ADD UNIQUE(project_id,git_branch,file_path,symbol_name,symbol_type);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS dev_scope ON dev_code_symbols(project_id,git_branch);

UPDATE enterprise_documents SET status='ARCHIVED' WHERE doc_id IN (
    SELECT doc_id FROM (
        SELECT doc_id,row_number() OVER(
            PARTITION BY doc_title,allowed_role ORDER BY valid_from DESC,doc_id DESC
        ) AS rn FROM enterprise_documents WHERE status='ACTIVE'
    ) d WHERE rn>1
);
CREATE UNIQUE INDEX IF NOT EXISTS enterprise_active_document
    ON enterprise_documents(doc_title,allowed_role) WHERE status='ACTIVE';

ALTER TABLE swarm_blackboard ADD COLUMN IF NOT EXISTS lease_token uuid;
ALTER TABLE swarm_blackboard ADD COLUMN IF NOT EXISTS lease_until timestamptz;
CREATE INDEX IF NOT EXISTS swarm_pending ON swarm_blackboard(workflow_id,status,created_at);

DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid='task_trajectories'::regclass AND contype='c'
    ) THEN
        ALTER TABLE task_trajectories ADD CHECK(success_score BETWEEN 0 AND 1);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid='tutor_user_progress'::regclass AND contype='c'
    ) THEN
        ALTER TABLE tutor_user_progress ADD CHECK(proficiency_score BETWEEN 0 AND 1);
    END IF;
END $$;

ALTER TABLE enterprise_documents DROP COLUMN IF EXISTS tsv;
ALTER TABLE enterprise_documents ADD COLUMN tsv tsvector
    GENERATED ALWAYS AS(to_tsvector('english',doc_title || ' ' || content)) STORED;
CREATE INDEX IF NOT EXISTS idx_ent_doc_tsv ON enterprise_documents USING gin(tsv);
