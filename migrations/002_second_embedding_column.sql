-- 002_second_embedding_column.sql
--
-- A second embedding column so two embedding models can be compared on the
-- same corpus without re-ingesting the text.
--
-- Why a column rather than a row-per-model table: a pgvector column has a
-- FIXED dimension, and an index is built for that dimension. Two models of
-- different width therefore cannot share a column, and a generic
-- (chunk_id, model, embedding) table would still need one column per distinct
-- width. Two columns is the honest shape of the constraint rather than an
-- abstraction that pretends the constraint is not there.
--
-- The cost is that adding a third width means another migration. That is
-- acceptable at this scale and is written down rather than discovered later.

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding_lg vector(1024);

COMMENT ON COLUMN chunks.embedding_lg IS
    '1024-dimensional embeddings, e.g. bge-m3. The 768-dimensional column '
    'holds bge-base-en-v1.5. Which column a query uses is decided by the '
    'active embedding model in config, not by the caller.';

CREATE INDEX IF NOT EXISTS chunks_embedding_lg_hnsw
    ON chunks USING hnsw (embedding_lg vector_cosine_ops);

INSERT INTO schema_migrations (version) VALUES ('002_second_embedding_column')
    ON CONFLICT (version) DO NOTHING;
