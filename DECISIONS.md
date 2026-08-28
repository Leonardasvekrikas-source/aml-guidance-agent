# Decisions

Why each significant choice was made, and what would change the answer in a
different context.

---

## Postgres with pgvector, rather than a dedicated vector database

**Chosen because** the corpus is 20–30 documents, on the order of ten thousand
chunks. At that size the retrieval bottleneck is not the vector index. What
does matter is that chunk text, document provenance and embeddings sit in one
transactional store, so a chunk cannot exist without the document it came from
and a citation cannot point at a row that was never ingested.

**Different context.** At tens of millions of vectors, or with heavy filtered
search and frequent reindexing, a dedicated store earns its second service.
Below roughly a million vectors, running two databases is usually cost without
benefit.

---

## `documents` and `chunks` as two tables, not one

**Chosen because** provenance is a property of the document, not of each chunk.
Denormalising publisher, source URL and publication date onto every chunk would
repeat them thousands of times and — more importantly — would allow two chunks
of the same document to disagree about where they came from. The whole point of
this project is that citations are checkable, and provenance that can
contradict itself is not checkable.

**Different context.** If chunks were ever ingested from sources with no stable
document identity — streaming transcripts, say — the join stops paying for
itself.

---

## Provenance as columns, not a JSONB blob

**Chosen because** these fields are constrained and queried, not merely carried.
`page` is an integer the database refuses to accept as zero. A JSONB blob would
make every provenance check a cast, and would let a malformed page number reach
a citation unchallenged.

**Different context.** JSONB is the right answer for genuinely heterogeneous
per-source metadata that nothing queries. That is not this.

---

## `chunk_profile` as a column, so two chunk sizes coexist

**Chosen because** M1 requires reporting retrieval quality at two chunk sizes.
Tagging each chunk with the configuration that produced it means the comparison
is a `WHERE` clause rather than a full re-ingestion, and both populations stay
available for later analysis.

**Cost.** The table holds roughly twice the rows, and every retrieval query must
filter by profile or it will silently mix populations and produce a meaningless
benchmark. That filter is a correctness requirement, not an optimisation.

---

## HNSW rather than IVFFlat for the vector index

**Chosen because** IVFFlat needs a training step and a tuned list count that is
only correct for a given corpus size — and this corpus grows as documents are
added. HNSW needs neither, and at this scale its slower build time is
irrelevant.

**Different context.** At large scale with tight memory limits, IVFFlat's
smaller index earns the tuning it demands.

---

## BM25 in Python, not Postgres `ts_rank`

**Chosen because** `ts_rank` is not BM25. It is a different scoring function
with different saturation behaviour, and describing it as BM25 in a README
would be inaccurate. `rank_bm25` computes real Okapi BM25, and the corpus is
small enough to hold in memory.

**Cost.** The lexical index lives in the application process rather than the
database, so it is rebuilt on startup and does not survive as a queryable
artifact. At this corpus size that rebuild is seconds. It would not scale, and
at that point the honest move is a real search engine, not a mislabelled
`ts_rank`.

The Postgres full-text index still exists, for corpus inspection only. It is
not the lexical retriever.

---

## Python 3.12 in the container, not the newest release

**Chosen because** `torch` and `sentence-transformers` wheels lag new CPython
releases. The development machine runs 3.14, where those wheels do not yet
exist. Pinning 3.12 in the image means the container builds regardless of what
is installed on the host.

---

## Both a Makefile and a PowerShell script

**Chosen because** the project is developed on Windows, where `make` is not
installed by default, but is meant to be cloned and run by strangers who are
mostly on Linux and macOS. Documenting a `make` workflow the author cannot run
would mean the reproducibility claims were never actually exercised.

**Cost.** Two entry points to keep in sync. The alternative was one entry point
that half the readers cannot run.
