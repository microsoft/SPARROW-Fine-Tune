-- Local base models: a ResNet-50 package uploaded directly (POST /v1/base-models/local), with no
-- Sparrow Studio involved. origin tells the two kinds apart. A local row has no Studio project,
-- import or source ids (none are invented); a Studio row keeps all of them as before.
ALTER TABLE base_models
    ADD COLUMN origin text NOT NULL DEFAULT 'studio'
        CHECK (origin IN ('studio', 'local'));
ALTER TABLE base_models ALTER COLUMN studio_project_id DROP NOT NULL;
ALTER TABLE base_models ALTER COLUMN studio_import_id DROP NOT NULL;
ALTER TABLE base_models ADD CONSTRAINT base_models_origin_ids CHECK (
    (origin = 'studio' AND studio_project_id IS NOT NULL AND studio_import_id IS NOT NULL)
    OR (origin = 'local' AND studio_project_id IS NULL AND studio_import_id IS NULL
        AND source_model_id IS NULL)
);
-- One row per local package: re-uploading the same bytes returns the existing row.
CREATE UNIQUE INDEX base_models_local_sha256 ON base_models (sha256) WHERE origin = 'local';
