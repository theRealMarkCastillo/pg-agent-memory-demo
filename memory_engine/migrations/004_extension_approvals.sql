-- Application records own checkpoint identity and authorize exact revisions.
CREATE TABLE ext_proposals (
    proposal_id uuid PRIMARY KEY,
    user_id text NOT NULL,
    target text NOT NULL,
    base_digest text NOT NULL,
    content text NOT NULL,
    digest text NOT NULL,
    revision integer NOT NULL DEFAULT 1 CHECK (revision > 0),
    status text NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending','approved','rejected','cancelled','applied')),
    thread_id text NOT NULL UNIQUE,
    effect_key uuid NOT NULL UNIQUE,
    evidence jsonb NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now(),
    applied_at timestamptz
);
CREATE TABLE ext_reviews (
    proposal_id uuid NOT NULL REFERENCES ext_proposals,
    revision integer NOT NULL,
    digest text NOT NULL,
    reviewer text NOT NULL,
    decision text NOT NULL CHECK (decision IN ('approved','rejected')),
    decided_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (proposal_id, revision)
);
CREATE TABLE ext_proposal_revisions (
    proposal_id uuid NOT NULL REFERENCES ext_proposals,
    revision integer NOT NULL,
    content text NOT NULL,
    digest text NOT NULL,
    PRIMARY KEY (proposal_id, revision)
);
