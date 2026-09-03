-- src/db/schema.sql
-- World Cup RAG knowledge base + monitoring tables

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- Knowledge base
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS chunks (
    id                   TEXT PRIMARY KEY,
    match_id             TEXT NOT NULL,
    chunk_type           TEXT NOT NULL,
    chunk_index          INTEGER NOT NULL DEFAULT 0,
    parent_id            TEXT,
    text                 TEXT NOT NULL,
    embedding            vector(768),              -- all-mpnet-base-v2

    -- denormalized metadata for filtering
    year                 INTEGER,
    tournament           TEXT,
    round                TEXT,
    date                 DATE,
    team1                TEXT,
    team2                TEXT,
    teams                TEXT[],
    ground               TEXT,
    winner               TEXT,
    is_draw              BOOLEAN,
    is_final             BOOLEAN,
    went_to_extra_time   BOOLEAN,
    had_penalties        BOOLEAN,
    team                 TEXT,                     -- lineup chunks only
    players              TEXT[],
    goal_scorers         TEXT[],
    final_score          TEXT,

    -- full original metadata
    metadata             JSONB,

    created_at           TIMESTAMPTZ DEFAULT NOW()
);

-- Vector similarity index
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks
    USING hnsw (embedding vector_cosine_ops);

-- Filter indexes
CREATE INDEX IF NOT EXISTS chunks_chunk_type_idx ON chunks (chunk_type);
CREATE INDEX IF NOT EXISTS chunks_year_idx       ON chunks (year);
CREATE INDEX IF NOT EXISTS chunks_match_id_idx   ON chunks (match_id);
CREATE INDEX IF NOT EXISTS chunks_is_final_idx   ON chunks (is_final);
CREATE INDEX IF NOT EXISTS chunks_team_idx       ON chunks (team);
CREATE INDEX IF NOT EXISTS chunks_parent_id_idx  ON chunks (parent_id);

-- ---------------------------------------------------------------------------
-- Monitoring
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS queries (
    id                   BIGSERIAL PRIMARY KEY,
    question             TEXT NOT NULL,
    rewritten_question   TEXT,
    retrieved_ids        TEXT[],
    response             TEXT,
    model                TEXT,
    latency_ms           INTEGER,
    created_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS feedback (
    id                   BIGSERIAL PRIMARY KEY,
    query_id             BIGINT REFERENCES queries(id) ON DELETE CASCADE,
    is_positive          BOOLEAN NOT NULL,
    comment              TEXT,
    created_at           TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS feedback_query_id_idx ON feedback (query_id);
CREATE INDEX IF NOT EXISTS queries_created_at_idx ON queries (created_at);


CREATE TABLE IF NOT EXISTS eval_questions (
    id                    BIGSERIAL PRIMARY KEY,
    question              TEXT NOT NULL,
    category              TEXT NOT NULL,       -- exact_fact | year_final | aggregate | lineup | multihop | negative
    expected_answer       TEXT,
    expected_match_ids    TEXT[],
    expected_chunk_types  TEXT[],
    created_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS eval_results (
    id                BIGSERIAL PRIMARY KEY,
    question_id       BIGINT REFERENCES eval_questions(id) ON DELETE CASCADE,
    run_label         TEXT NOT NULL,      -- e.g. "baseline_2026_09_03"
    retrieved_ids     TEXT[],
    hit_at_12         BOOLEAN,
    mrr               FLOAT,
    parent_hit_at_12  BOOLEAN,
    child_type_hit_at_12 BOOLEAN,
    manual_score      TEXT,               -- correct | partial | wrong
    llm_answer        TEXT,
    latency_ms        INTEGER,
    created_at        TIMESTAMPTZ DEFAULT NOW()
);