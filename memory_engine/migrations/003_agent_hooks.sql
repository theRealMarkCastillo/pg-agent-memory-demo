-- Session excerpts are unverified observations, separate from scored trajectories.
CREATE TABLE IF NOT EXISTS agent_hook_memories (
    memory_id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id text NOT NULL,
    git_branch text NOT NULL,
    source text NOT NULL,
    session_id text NOT NULL,
    turn_id text NOT NULL,
    prompt text NOT NULL,
    response text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    search_document tsvector GENERATED ALWAYS AS
        (to_tsvector('english', prompt || ' ' || response)) STORED,
    UNIQUE(project_id, git_branch, source, session_id, turn_id)
);
CREATE INDEX IF NOT EXISTS agent_hook_scope
    ON agent_hook_memories(project_id, git_branch, created_at);
CREATE INDEX IF NOT EXISTS agent_hook_search
    ON agent_hook_memories USING gin(search_document);
