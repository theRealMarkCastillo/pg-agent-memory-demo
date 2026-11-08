CREATE TABLE ext_research_documents (
    document_id uuid NOT NULL,
    version integer NOT NULL CHECK(version > 0),
    notebook_id uuid NOT NULL,
    user_id text NOT NULL,
    content text NOT NULL,
    digest text NOT NULL,
    PRIMARY KEY(document_id,version)
);
CREATE TABLE ext_research_reports (
    report_id uuid PRIMARY KEY,
    notebook_id uuid NOT NULL,
    user_id text NOT NULL,
    evidence_digest text NOT NULL,
    digest text NOT NULL,
    report jsonb NOT NULL,
    claims jsonb NOT NULL,
    reviewer text,
    accepted boolean,
    CHECK((reviewer IS NULL) = (accepted IS NULL))
);
