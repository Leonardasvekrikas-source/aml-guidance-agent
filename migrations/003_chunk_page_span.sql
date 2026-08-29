-- 003_chunk_page_span.sql
--
-- A chunk records the page it STARTS on. That was adequate while every chunk
-- fitted inside a page or two, and it silently stopped being adequate when a
-- larger chunk profile was introduced: a 1024-token chunk averages ~4,900
-- characters against ~2,700 characters per page, so it routinely spans two or
-- three pages while being credited to one.
--
-- Page-level retrieval scoring compares the pages a chunk covers against the
-- pages the answer is on. Under start-page-only attribution, a large chunk
-- that literally contains the answer is scored as a miss whenever the answer
-- falls on its second or third page — which made a larger chunk profile look
-- far worse than it is. That is a measurement artifact, not a retrieval
-- result, and it is the kind that quietly produces a confident wrong
-- conclusion.
--
-- page stays the first page; page_end is the last.

ALTER TABLE chunks ADD COLUMN IF NOT EXISTS page_end integer;

COMMENT ON COLUMN chunks.page_end IS
    'Last page this chunk covers. chunks.page is the first. A chunk spanning '
    'pages 7-9 has page=7 and page_end=9; page-level scoring must consider the '
    'whole range or it penalises large chunks for spanning pages.';

ALTER TABLE chunks DROP CONSTRAINT IF EXISTS chunks_page_span_ordered;
ALTER TABLE chunks ADD CONSTRAINT chunks_page_span_ordered
    CHECK (page_end IS NULL OR page IS NULL OR page_end >= page);

INSERT INTO schema_migrations (version) VALUES ('003_chunk_page_span')
    ON CONFLICT (version) DO NOTHING;
