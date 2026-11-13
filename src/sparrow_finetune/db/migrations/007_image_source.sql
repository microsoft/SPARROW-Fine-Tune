-- Where an image came from: 'upload' (added in this app) or 'studio' (imported from Sparrow Studio).
-- Rows that existed before this migration are marked 'upload'; re-importing from Studio corrects them.
ALTER TABLE dataset_images
    ADD COLUMN source text NOT NULL DEFAULT 'upload' CHECK (source IN ('upload', 'studio'));
