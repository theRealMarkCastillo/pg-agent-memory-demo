CREATE TABLE companion_memory_state (user_id text PRIMARY KEY, enabled boolean NOT NULL DEFAULT true, generation integer NOT NULL DEFAULT 0);
ALTER TABLE companion_episodes ADD COLUMN ingestion_id text;
UPDATE companion_episodes SET ingestion_id=episode_id::text;
ALTER TABLE companion_episodes ALTER COLUMN ingestion_id SET NOT NULL;
ALTER TABLE companion_episodes ADD UNIQUE(user_id, ingestion_id), ADD UNIQUE(user_id, episode_id);
ALTER TABLE companion_graph_nodes ADD COLUMN embedding halfvec(1536);
ALTER TABLE companion_graph_nodes ADD COLUMN is_subject boolean NOT NULL DEFAULT false;
UPDATE companion_graph_nodes n SET is_subject=true WHERE EXISTS(SELECT 1 FROM companion_graph_edges e WHERE e.source_node_id=n.node_id) OR NOT EXISTS(SELECT 1 FROM companion_graph_edges e WHERE e.target_node_id=n.node_id);
ALTER TABLE companion_graph_nodes ADD UNIQUE(user_id, node_id);
ALTER TABLE companion_graph_edges ADD COLUMN valid_from timestamptz NOT NULL DEFAULT clock_timestamp();
ALTER TABLE companion_graph_edges ADD COLUMN recorded_at timestamptz NOT NULL DEFAULT clock_timestamp();
ALTER TABLE companion_graph_edges ADD COLUMN embedding halfvec(1536);
DO $$ DECLARE c record; BEGIN
 FOR c IN SELECT conname FROM pg_constraint WHERE conrelid='companion_graph_edges'::regclass AND contype='u' LOOP
 EXECUTE format('ALTER TABLE companion_graph_edges DROP CONSTRAINT %I', c.conname);
 END LOOP;
END $$;
CREATE UNIQUE INDEX companion_active_edge ON companion_graph_edges(user_id, source_node_id, target_node_id, relationship_type, subject) WHERE status='ACTIVE';
ALTER TABLE companion_graph_edges ADD CONSTRAINT edge_source_owner FOREIGN KEY(user_id, source_node_id) REFERENCES companion_graph_nodes(user_id,node_id);
ALTER TABLE companion_graph_edges ADD CONSTRAINT edge_target_owner FOREIGN KEY(user_id, target_node_id) REFERENCES companion_graph_nodes(user_id,node_id);
ALTER TABLE companion_graph_edges ADD CONSTRAINT edge_episode_owner FOREIGN KEY(user_id,source_episode_id) REFERENCES companion_episodes(user_id,episode_id);
ALTER TABLE companion_chunks ADD CONSTRAINT chunk_episode_owner FOREIGN KEY(user_id,episode_id) REFERENCES companion_episodes(user_id,episode_id) ON DELETE CASCADE;
CREATE TABLE companion_fact_sources (
 edge_id uuid REFERENCES companion_graph_edges(edge_id) ON DELETE CASCADE,
 user_id text NOT NULL, episode_id uuid NOT NULL,
 PRIMARY KEY(edge_id,episode_id),
 FOREIGN KEY(user_id,episode_id) REFERENCES companion_episodes(user_id,episode_id)
);
INSERT INTO companion_fact_sources SELECT edge_id,user_id,source_episode_id FROM companion_graph_edges WHERE source_episode_id IS NOT NULL;
CREATE INDEX companion_edge_owner ON companion_graph_edges(user_id,subject,status);
CREATE INDEX companion_edge_embedding ON companion_graph_edges USING hnsw(embedding halfvec_cosine_ops);
CREATE INDEX companion_node_embedding ON companion_graph_nodes USING hnsw(embedding halfvec_cosine_ops);
CREATE INDEX companion_ephemeral_owner ON companion_ephemerals(user_id,expires_at);
CREATE INDEX companion_chunk_owner ON companion_chunks(user_id);
-- Preserve duplicate source rows before reconciling the current symbol projection.
CREATE TABLE migration_duplicate_symbols AS SELECT * FROM dev_code_symbols WHERE symbol_id IN (
 SELECT symbol_id FROM (SELECT symbol_id,row_number() OVER(PARTITION BY project_id,git_branch,file_path,symbol_name,symbol_type ORDER BY created_at DESC,symbol_id DESC) AS rn FROM dev_code_symbols) d WHERE rn>1
);
DELETE FROM dev_code_symbols WHERE symbol_id IN (SELECT symbol_id FROM migration_duplicate_symbols);
ALTER TABLE dev_code_symbols ADD UNIQUE(project_id,git_branch,file_path,symbol_name,symbol_type);
CREATE INDEX dev_scope ON dev_code_symbols(project_id,git_branch);
UPDATE enterprise_documents SET status='ARCHIVED' WHERE doc_id IN (
 SELECT doc_id FROM (SELECT doc_id,row_number() OVER(PARTITION BY doc_title,allowed_role ORDER BY valid_from DESC,doc_id DESC) AS rn FROM enterprise_documents WHERE status='ACTIVE') d WHERE rn>1
);
CREATE UNIQUE INDEX enterprise_active_document ON enterprise_documents(doc_title,allowed_role) WHERE status='ACTIVE';
ALTER TABLE swarm_blackboard ADD COLUMN lease_token uuid, ADD COLUMN lease_until timestamptz;
CREATE INDEX swarm_pending ON swarm_blackboard(workflow_id,status,created_at);
ALTER TABLE task_trajectories ADD CHECK(success_score BETWEEN 0 AND 1);
ALTER TABLE tutor_user_progress ADD CHECK(proficiency_score BETWEEN 0 AND 1);

ALTER TABLE enterprise_documents DROP COLUMN tsv;
ALTER TABLE enterprise_documents ADD COLUMN tsv tsvector GENERATED ALWAYS AS(to_tsvector('english',doc_title || ' ' || content)) STORED;
CREATE INDEX idx_ent_doc_tsv ON enterprise_documents USING gin(tsv);
