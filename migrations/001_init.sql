BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

-- =========================
-- USERS
-- =========================
CREATE TABLE IF NOT EXISTS users (
    id bigserial PRIMARY KEY,
    external_uid varchar(128) NOT NULL UNIQUE,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- =========================
-- ACTS
-- =========================
CREATE TABLE IF NOT EXISTS acts (
    id bigserial PRIMARY KEY,
    source_nd varchar(128) NOT NULL UNIQUE,
    doc_type varchar(64) NOT NULL,
    title text NOT NULL,
    issued_by varchar(255),
    doc_number varchar(64) NOT NULL,
    doc_date date NOT NULL,
    status varchar(64) NOT NULL,
    source_url text,
    imported_at timestamptz NOT NULL DEFAULT now()
);

-- =========================
-- CHUNKS
-- vector(768) под E5-base
-- =========================
CREATE TABLE IF NOT EXISTS chunks (
    id bigserial PRIMARY KEY,
    act_id bigint NOT NULL REFERENCES acts(id) ON DELETE CASCADE,
    chunk_index int NOT NULL CHECK (chunk_index >= 0),
    text text NOT NULL,
    start_char int CHECK (start_char >= 0),
    end_char int CHECK (end_char >= start_char),
    token_count int NOT NULL CHECK (token_count > 0),
    structure_ref text,
    embedding vector(768),
    search_vector tsvector NOT NULL,
    embedding_model varchar(128) NOT NULL,
    hash varchar(64) NOT NULL UNIQUE,
    UNIQUE (act_id, chunk_index)
);

-- =========================
-- QUERIES
-- =========================
CREATE TABLE IF NOT EXISTS queries (
    id bigserial PRIMARY KEY,
    user_id bigint NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    question text NOT NULL,
    normalized_question text,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- =========================
-- ANSWERS
-- =========================
CREATE TABLE IF NOT EXISTS answers (
    id bigserial PRIMARY KEY,
    query_id bigint NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    answer_text text NOT NULL,
    llm_model varchar(128) NOT NULL,
    prompt_version varchar(64),
    latency_ms int NOT NULL CHECK (latency_ms >= 0),
    created_at timestamptz NOT NULL DEFAULT now()
);

-- =========================
-- ANSWER CITATIONS
-- =========================
CREATE TABLE IF NOT EXISTS answer_citations (
    id bigserial PRIMARY KEY,
    answer_id bigint NOT NULL REFERENCES answers(id) ON DELETE CASCADE,
    chunk_id bigint NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    rank int NOT NULL CHECK (rank > 0),
    relevance_score double precision CHECK (relevance_score >= 0),
    quote text NOT NULL
);

-- =========================
-- FEEDBACK
-- =========================
CREATE TABLE IF NOT EXISTS feedback (
    id bigserial PRIMARY KEY,
    answer_id bigint NOT NULL REFERENCES answers(id) ON DELETE CASCADE,
    user_id bigint NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    rating int NOT NULL CHECK (rating BETWEEN 1 AND 5),
    comment text,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (answer_id, user_id)
);

-- =========================
-- EXPERIMENTS
-- =========================
CREATE TABLE IF NOT EXISTS experiments (
    id bigserial PRIMARY KEY,
    name varchar(128) NOT NULL,
    description text,
    params_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- =========================
-- EXPERIMENT RUNS
-- =========================
CREATE TABLE IF NOT EXISTS experiment_runs (
    id bigserial PRIMARY KEY,
    experiment_id bigint NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    query_id bigint NOT NULL REFERENCES queries(id) ON DELETE CASCADE,
    metrics_json jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- =========================
-- INDEXES
-- =========================
CREATE INDEX IF NOT EXISTS idx_acts_number_date
    ON acts (doc_number, doc_date);

CREATE INDEX IF NOT EXISTS idx_queries_user_created_at
    ON queries (user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_answers_query_id
    ON answers (query_id);

CREATE INDEX IF NOT EXISTS idx_answer_citations_answer_id
    ON answer_citations (answer_id);

CREATE INDEX IF NOT EXISTS idx_answer_citations_chunk_id
    ON answer_citations (chunk_id);

CREATE INDEX IF NOT EXISTS idx_feedback_answer_id
    ON feedback (answer_id);

CREATE INDEX IF NOT EXISTS idx_chunks_act_id
    ON chunks (act_id);

CREATE INDEX IF NOT EXISTS idx_experiment_runs_experiment_query
    ON experiment_runs (experiment_id, query_id);

CREATE INDEX IF NOT EXISTS idx_chunks_search_vector
    ON chunks USING gin (search_vector);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

COMMIT;