BEGIN;

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS password_hash text NULL;

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS display_name varchar(128) NULL;

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS last_login_at timestamptz NULL;

CREATE TABLE IF NOT EXISTS feedback (
    id bigserial PRIMARY KEY,
    answer_id bigint NOT NULL REFERENCES answers(id) ON DELETE CASCADE,
    user_id bigint NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    rating int NOT NULL CHECK (rating BETWEEN 1 AND 5),
    comment text,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (answer_id, user_id)
);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'feedback_answer_user_unique'
          AND conrelid = 'feedback'::regclass
    ) AND NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE contype = 'u'
          AND conrelid = 'feedback'::regclass
          AND conkey = ARRAY[
              (SELECT attnum FROM pg_attribute WHERE attrelid = 'feedback'::regclass AND attname = 'answer_id'),
              (SELECT attnum FROM pg_attribute WHERE attrelid = 'feedback'::regclass AND attname = 'user_id')
          ]::smallint[]
    ) THEN
        ALTER TABLE feedback
            ADD CONSTRAINT feedback_answer_user_unique UNIQUE (answer_id, user_id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_feedback_answer_id
    ON feedback (answer_id);

COMMIT;
