-- eval_schema_migration.sql
-- Adds evaluation tables to the World Cup RAG project.
--
-- These are separate from `queries` / `feedback` (which are for logging
-- live production traffic once that's wired up). Eval runs get their own
-- tables so batch test-harness runs never mix with real user data, and so
-- each run is re-runnable/comparable via `run_label`.
--
-- Apply with (container already running, so schema.sql's initdb mount
-- won't re-run automatically):
--   docker exec -i worldcup-postgres psql -U worldcup -d worldcup < eval_schema_migration.sql
--
-- Also append this block to src/db/schema.sql so a fresh `docker compose up`
-- creates it from scratch next time.

CREATE TABLE IF NOT EXISTS eval_questions (
    id                    BIGSERIAL PRIMARY KEY,
    question              TEXT NOT NULL UNIQUE,
    category              TEXT NOT NULL,          -- exact_fact | year_final | aggregate | lineup | multihop | negative
    expected_answer       TEXT,
    expected_match_ids    TEXT[] NOT NULL,
    expected_chunk_types  TEXT[],                  -- optional, e.g. {goals} or {lineup,parent}
    created_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS eval_results (
    id                    BIGSERIAL PRIMARY KEY,
    question_id           BIGINT NOT NULL REFERENCES eval_questions(id) ON DELETE CASCADE,
    run_label             TEXT NOT NULL,           -- groups all rows from one eval run, e.g. "baseline_2026_09_03"
    retrieved_ids         TEXT[],                  -- match_ids in retrieved rank order (top-k)
    hit_at_12             BOOLEAN,
    mrr                   FLOAT,
    parent_hit_at_12      BOOLEAN,
    child_type_hit_at_12  BOOLEAN,                 -- NULL when the question has no expected_chunk_types
    manual_score          TEXT,                    -- correct | partial | wrong — filled in by hand after review
    llm_answer            TEXT,
    latency_ms            INTEGER,
    created_at            TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS eval_results_run_label_idx   ON eval_results (run_label);
CREATE INDEX IF NOT EXISTS eval_results_question_id_idx ON eval_results (question_id);
