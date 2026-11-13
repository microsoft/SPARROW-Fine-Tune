-- Base models for warm start: a model this service pushed to Sparrow Studio and imported back.
-- model_versions is not reused: it requires the job and snapshot that produced the model.
-- One row per Studio model: re-importing the same (project, Studio id) returns the existing row.
CREATE TABLE base_models (
    id                bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    studio_project_id text NOT NULL,
    studio_import_id  text NOT NULL,
    source_model_id   text,
    pipeline          text NOT NULL,
    model_arch        text NOT NULL,
    labels            jsonb NOT NULL,
    sha256            text NOT NULL,
    object_key        text NOT NULL,
    size_bytes        bigint NOT NULL,
    created_at        timestamptz NOT NULL DEFAULT now(),
    UNIQUE (studio_project_id, studio_import_id)
);
