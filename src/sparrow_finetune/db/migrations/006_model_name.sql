-- A user-chosen display name for a model version. NULL shows the default "Model #<id>".
ALTER TABLE model_versions ADD COLUMN name text;
