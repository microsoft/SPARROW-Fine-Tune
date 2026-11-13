-- Phase 1A foundation schema: workspaces, datasets, frozen snapshots, job queue, model versions.

CREATE TABLE workspaces (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name        text NOT NULL CHECK (name <> ''),
    created_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE datasets (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    workspace_id  bigint NOT NULL REFERENCES workspaces(id) ON DELETE RESTRICT,
    name          text NOT NULL CHECK (name <> ''),
    task          text NOT NULL CHECK (task IN ('classification', 'detection')),
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX datasets_workspace_idx ON datasets (workspace_id);

CREATE TABLE dataset_images (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id  bigint NOT NULL REFERENCES datasets(id) ON DELETE RESTRICT,
    name        text NOT NULL,
    object_key  text NOT NULL,
    sha256      text NOT NULL,
    width       integer NOT NULL CHECK (width > 0),
    height      integer NOT NULL CHECK (height > 0),
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (dataset_id, name)
);

-- Current (mutable) labelling state: one item per annotated image.
CREATE TABLE dataset_items (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id  bigint NOT NULL REFERENCES datasets(id) ON DELETE RESTRICT,
    image_id    bigint NOT NULL UNIQUE REFERENCES dataset_images(id) ON DELETE RESTRICT,
    group_name  text,
    updated_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX dataset_items_dataset_idx ON dataset_items (dataset_id);

CREATE TABLE annotations (
    id       bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    item_id  bigint NOT NULL REFERENCES dataset_items(id) ON DELETE RESTRICT,
    label    text NOT NULL CHECK (label <> ''),
    bbox     double precision[] CHECK (bbox IS NULL OR cardinality(bbox) = 4)
);
CREATE INDEX annotations_item_idx ON annotations (item_id);

-- Frozen copies: later imports never change a snapshot.
CREATE TABLE snapshots (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    dataset_id    bigint NOT NULL REFERENCES datasets(id) ON DELETE RESTRICT,
    content_hash  text NOT NULL,
    item_count    integer NOT NULL,
    label_counts  jsonb NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX snapshots_dataset_idx ON snapshots (dataset_id);

CREATE TABLE snapshot_items (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    snapshot_id  bigint NOT NULL REFERENCES snapshots(id) ON DELETE RESTRICT,
    position     integer NOT NULL,
    image_name   text NOT NULL,
    object_key   text NOT NULL,
    sha256       text NOT NULL,
    width        integer NOT NULL,
    height       integer NOT NULL,
    annotations  jsonb NOT NULL,
    group_name   text,
    UNIQUE (snapshot_id, position)
);

CREATE TABLE jobs (
    id                   bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    pipeline             text NOT NULL,
    snapshot_id          bigint NOT NULL REFERENCES snapshots(id) ON DELETE RESTRICT,
    params               jsonb NOT NULL DEFAULT '{}'::jsonb,
    requires             text[] NOT NULL DEFAULT '{}',
    state                text NOT NULL DEFAULT 'queued'
                         CHECK (state IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    attempts             integer NOT NULL DEFAULT 0,
    max_attempts         integer NOT NULL DEFAULT 3 CHECK (max_attempts > 0),
    lease_owner          text,
    lease_expires_at     timestamptz,
    cancel_requested_at  timestamptz,
    error                text,
    created_at           timestamptz NOT NULL DEFAULT now(),
    started_at           timestamptz,
    finished_at          timestamptz
);
CREATE INDEX jobs_queued_idx ON jobs (id) WHERE state = 'queued';
CREATE INDEX jobs_running_lease_idx ON jobs (lease_expires_at) WHERE state = 'running';

CREATE TABLE job_events (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id      bigint NOT NULL REFERENCES jobs(id) ON DELETE RESTRICT,
    seq         integer NOT NULL,
    stage       text NOT NULL,
    message     text NOT NULL DEFAULT '',
    metrics     jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (job_id, seq)
);

CREATE TABLE model_versions (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    job_id       bigint NOT NULL UNIQUE REFERENCES jobs(id) ON DELETE RESTRICT,
    pipeline     text NOT NULL,
    snapshot_id  bigint NOT NULL REFERENCES snapshots(id) ON DELETE RESTRICT,
    metrics      jsonb NOT NULL DEFAULT '{}'::jsonb,
    format       text NOT NULL,
    object_key   text NOT NULL,
    sha256       text NOT NULL,
    metadata     jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at   timestamptz NOT NULL DEFAULT now()
);
