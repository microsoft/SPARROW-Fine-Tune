-- Classification datasets train on box crops by default. With whole_image set, a dataset keeps
-- image-level labels only (boxes are folded into the image's labels) and trains on full images.
ALTER TABLE datasets ADD COLUMN whole_image boolean NOT NULL DEFAULT false;
ALTER TABLE datasets ADD CONSTRAINT datasets_whole_image_classification
    CHECK (NOT whole_image OR task = 'classification');
