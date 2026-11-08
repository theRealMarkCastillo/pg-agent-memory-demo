CREATE TABLE ext_incident_events (
    event_id uuid PRIMARY KEY,
    incident_id uuid NOT NULL,
    user_id text NOT NULL,
    source_id text NOT NULL,
    source_revision integer NOT NULL CHECK(source_revision > 0),
    occurred_at timestamptz NOT NULL,
    received_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    arrival_order bigint GENERATED ALWAYS AS IDENTITY UNIQUE,
    content text NOT NULL,
    digest text NOT NULL,
    supersedes uuid UNIQUE REFERENCES ext_incident_events,
    revokes uuid UNIQUE REFERENCES ext_incident_events,
    CHECK(supersedes IS NULL OR revokes IS NULL),
    UNIQUE(user_id,incident_id,source_id,source_revision)
);
CREATE TABLE ext_incident_summaries (
    user_id text NOT NULL,
    incident_id uuid NOT NULL,
    evidence_digest text NOT NULL,
    summary jsonb NOT NULL,
    PRIMARY KEY(user_id,incident_id)
);
