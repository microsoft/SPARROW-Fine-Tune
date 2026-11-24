-- DINO feature preparation (Data Preparation). A prepared vector is keyed by the crop's
-- pixels (image sha256 + crop box), the encoder (its model hash) and the preprocessing (input
-- size); labels and training hyperparameters are not part of the key. The vector itself is a
-- file under <data_dir>/features; this table is the index the status checks read.
CREATE TABLE feature_vectors (
    key           text PRIMARY KEY,
    image_sha256  text NOT NULL,
    box           integer[] NOT NULL,
    encoder_hash  text NOT NULL,
    encoder_id    text NOT NULL,
    input_size    integer NOT NULL,
    dim           integer NOT NULL CHECK (dim > 0),
    path          text NOT NULL,
    created_at    timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX feature_vectors_image_idx ON feature_vectors (image_sha256);

-- An opt-in snapshot records which encoder and preprocessing its features were prepared for:
-- {"encoder_hash", "input_size", "total", "reused", "to_compute", "job_id"}. NULL = not opted in.
ALTER TABLE snapshots ADD COLUMN features jsonb;
