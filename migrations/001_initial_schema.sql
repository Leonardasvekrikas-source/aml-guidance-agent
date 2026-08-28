-- 001_initial_schema.sql
--
-- Applied automatically by the postgres entrypoint when the data volume is
-- first initialised, and reapplied idempotently by `make migrate`.
--
-- Design notes, because these are the questions this schema invites:
--
-- Why two tables rather than one?
--   Document-level provenance (publisher, source URL, publication date, the
--   hash of the bytes we actually downloaded) is recorded once per document.
--   Denormalising it onto every chunk would repeat it thousands of times and
--   allow two chunks of the same document to disagree about where they came
--   from. Provenance that can contradict itself is not provenance.
--
-- Why are page and section_heading columns rather than a JSON blob?
--   Because they are queried and constrained, not just carried. A citation is
--   only checkable if you can say "this claim came from page 34 of this
--   document" and have the database enforce that page is an integer. A JSONB
--   blob makes provenance unqueryable without a cast and unvalidatable at
--   write time, and provenance is the point of this project.
--
-- Why chunk_profile?
--   M1 requires comparing two chunk sizes. Storing both populations in one
--   table, tagged, means the retrieval benchmark can run over either without
--   re-ingesting, and the comparison is a WHERE clause rather than a rebuild.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id               text PRIMARY KEY,
    title            text        NOT NULL,
    publisher        text        NOT NULL,
    source_url       text        NOT NULL,
    publication_date date,
    retrieved_at     timestamptz NOT NULL,
    sha256           text        NOT NULL,
    page_count       integer,
    doc_type         text,
    created_at       timestamptz NOT NULL DEFAULT now()
);

COMMENT ON COLUMN documents.sha256 IS
    'Hash of the downloaded bytes. If a publisher silently replaces a PDF, '
    'this is how the corpus notices rather than quietly changing underneath '
    'the evaluation set.';

CREATE TABLE IF NOT EXISTS chunks (
    id              bigserial PRIMARY KEY,
    document_id     text    NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_profile   text    NOT NULL,
    chunk_index     integer NOT NULL,
    page            integer,
    section_heading text,
    text            text    NOT NULL,
    char_count      integer NOT NULL,
    token_count     integer,
    embedding       vector(768),
    created_at      timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT chunks_unique_position UNIQUE (document_id, chunk_profile, chunk_index),
    CONSTRAINT chunks_text_not_blank  CHECK (length(btrim(text)) > 0),
    CONSTRAINT chunks_page_positive   CHECK (page IS NULL OR page > 0)
);

COMMENT ON COLUMN chunks.chunk_profile IS
    'Which chunking configuration produced this row, e.g. "p512" or "p1024". '
    'Two profiles coexist so M1 can compare chunk sizes without re-ingesting.';

CREATE INDEX IF NOT EXISTS chunks_document_idx
    ON chunks (document_id);

CREATE INDEX IF NOT EXISTS chunks_profile_idx
    ON chunks (chunk_profile);

-- Cosine distance, matching the normalised embeddings bge-base produces.
-- HNSW rather than IVFFlat: this corpus is small enough that build time is
-- irrelevant and HNSW does not need a training step or a tuned list count.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Postgres full-text index. NOT the lexical retriever — BM25 is computed in
-- Python by rank_bm25, because ts_rank is not BM25 and claiming otherwise in
-- a README would be a lie. This index exists only for corpus inspection.
CREATE INDEX IF NOT EXISTS chunks_text_fts
    ON chunks USING gin (to_tsvector('english', text));

CREATE TABLE IF NOT EXISTS schema_migrations (
    version    text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

INSERT INTO schema_migrations (version) VALUES ('001_initial_schema')
    ON CONFLICT (version) DO NOTHING;
