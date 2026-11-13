-- Studio launch: a workspace can stand for one Studio project, and each dataset remembers
-- what its last Studio import contained so the UI can say "up to date".

ALTER TABLE workspaces ADD COLUMN studio_project text;
CREATE UNIQUE INDEX workspaces_studio_project_idx
    ON workspaces (studio_project) WHERE studio_project IS NOT NULL;

CREATE TABLE studio_imports (
    dataset_id     bigint PRIMARY KEY REFERENCES datasets(id) ON DELETE CASCADE,
    project        text NOT NULL,
    reviewed_only  boolean NOT NULL,
    -- sha256 of Studio's export listing (ids, names, review state, boxes).
    export_sha256  text NOT NULL,
    -- Names of the dataset images the import wrote.
    names          jsonb NOT NULL,
    imported_at    timestamptz NOT NULL DEFAULT now()
);
