-- Phase 2: user accounts and sessions, audit trail, review state, model deployments.

CREATE TABLE users (
    id bigserial PRIMARY KEY,
    username text NOT NULL UNIQUE CHECK (length(btrim(username)) > 0),
    password_hash text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE sessions (
    id bigserial PRIMARY KEY,
    user_id bigint NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    token_sha256 text NOT NULL UNIQUE,
    created_at timestamptz NOT NULL DEFAULT now(),
    expires_at timestamptz NOT NULL
);
CREATE INDEX sessions_expires_at ON sessions (expires_at);

CREATE TABLE audit_events (
    id bigserial PRIMARY KEY,
    principal text NOT NULL,
    action text NOT NULL,
    target text NOT NULL DEFAULT '',
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE dataset_items
    ADD COLUMN review_status text NOT NULL DEFAULT 'pending'
    CHECK (review_status IN ('pending', 'confirmed', 'excluded'));

CREATE TABLE deployments (
    id bigserial PRIMARY KEY,
    model_id bigint NOT NULL REFERENCES model_versions (id),
    consumer text NOT NULL CHECK (consumer IN ('local', 'studio')),
    project text NOT NULL CHECK (length(btrim(project)) > 0),
    activated_at timestamptz NOT NULL DEFAULT now(),
    activated_by text NOT NULL,
    deactivated_at timestamptz,
    external_id text
);
CREATE UNIQUE INDEX deployments_one_active
    ON deployments (consumer, project) WHERE deactivated_at IS NULL;
CREATE INDEX deployments_model ON deployments (model_id);
