-- A job can be retried once; a second retry of the same job is refused (409).
ALTER TABLE jobs ADD COLUMN retry_of bigint UNIQUE REFERENCES jobs (id) ON DELETE SET NULL;
