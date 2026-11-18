CREATE TABLE ext_attempts (
    attempt_id uuid PRIMARY KEY,
    user_id text NOT NULL,
    source_csv text NOT NULL,
    artifact text,
    artifact_digest text,
    procedure text NOT NULL,
    spec_version text NOT NULL,
    checker_version text NOT NULL,
    environment text NOT NULL,
    verified boolean NOT NULL,
    errors jsonb NOT NULL,
    status text NOT NULL DEFAULT 'candidate' CHECK(status IN ('candidate','promoted','retired')),
    promoter text,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK(status <> 'promoted' OR (verified AND promoter IS NOT NULL))
);
