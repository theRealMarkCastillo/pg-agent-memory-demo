-- NULL means existing vectors have unknown provenance and require reindexing.
ALTER TABLE companion_memory_state ADD COLUMN IF NOT EXISTS embedding_space text;
