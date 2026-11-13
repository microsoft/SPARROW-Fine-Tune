-- The worker that wrote a job's terminal state. NULL when the reaper or a cancel of a
-- queued job ended it: a stale worker on the same attempt then cannot log events to it.
ALTER TABLE jobs ADD COLUMN finished_by text;
