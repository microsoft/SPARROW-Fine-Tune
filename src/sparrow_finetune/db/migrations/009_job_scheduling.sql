-- Multi-user GPU scheduling: who submitted each job (fair ordering, per-user running cap) and a
-- registry of live workers (queue status: "all 2 GPU workers are busy", "no GPU worker online").

ALTER TABLE jobs ADD COLUMN IF NOT EXISTS submitted_by text;
CREATE INDEX IF NOT EXISTS jobs_running_submitter_idx
    ON jobs (submitted_by) WHERE state = 'running';

CREATE TABLE IF NOT EXISTS workers (
    id         text PRIMARY KEY,
    resources  text[] NOT NULL,
    pipelines  text[],            -- pipeline ids the worker can run; NULL means any
    last_seen  timestamptz NOT NULL DEFAULT now()
);
