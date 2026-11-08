CREATE TABLE ext_policies (
    document_id uuid NOT NULL,
    version integer NOT NULL CHECK (version > 0),
    user_id text NOT NULL,
    content text NOT NULL,
    digest text NOT NULL,
    refund_days integer NOT NULL CHECK (refund_days >= 0),
    valid_until timestamptz NOT NULL,
    PRIMARY KEY (document_id,version)
);
CREATE TABLE ext_assertions (
    assertion_id uuid PRIMARY KEY,
    case_id uuid NOT NULL,
    user_id text NOT NULL,
    source text NOT NULL,
    origin text NOT NULL CHECK (origin IN ('user_report','operator_correction')),
    purchase_age_days integer NOT NULL CHECK (purchase_age_days >= 0),
    digest text NOT NULL,
    supersedes uuid UNIQUE REFERENCES ext_assertions,
    created_at timestamptz NOT NULL DEFAULT now()
);
