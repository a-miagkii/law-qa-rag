BEGIN;

ALTER TABLE answers
    ADD COLUMN IF NOT EXISTS needs_clarification boolean NOT NULL DEFAULT false;

ALTER TABLE answers
    ADD COLUMN IF NOT EXISTS retrieval_method varchar(64);

ALTER TABLE answers
    ADD COLUMN IF NOT EXISTS retrieved_chunk_ids jsonb NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE answers
    ADD COLUMN IF NOT EXISTS dropped_chunk_ids jsonb NOT NULL DEFAULT '[]'::jsonb;

COMMIT;
