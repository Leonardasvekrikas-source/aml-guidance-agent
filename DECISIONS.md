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

---

## Cross-encoder reranking as a second stage, not a replacement

**Chosen because** BM25 and dense retrieval both score a query against a
passage without letting the two interact — BM25 counts term overlap, and a
dense passage embedding is computed before the query exists. That independence
is what makes them fast enough to search 8,000 chunks, and it is also their
ceiling: neither can represent "this passage answers *that* question" as
opposed to "this passage is about the same subject". A cross-encoder puts both
in one forward pass and can.

Measured: recall@5 0.750 to 0.867, MRR 0.607 to 0.781.

**Cost.** Every candidate costs a model forward pass, so median query latency
goes from 31 ms to 536 ms — roughly 17x. That is why it reranks 50 candidates
rather than searching the corpus: a cross-encoder over 8,000 chunks would take
minutes per query.

**Candidate count: 50.** Measured, not guessed. The ceiling saturates there;
100 and 200 candidates cost more and retrieve nothing new.

**Different context.** Under a hard latency budget below ~200 ms, rerank 25
candidates instead: it keeps most of the gain (0.833) at half the cost. Below
that, drop reranking and accept 0.750.

---

## Reporting the first-stage ceiling separately from the achieved score

**Chosen because** "recall@5 is 0.867" does not say which stage to improve. The
ceiling — what fraction of gold passages reached the candidate pool at all —
separates a pool that never contained the answer from a reranker that mis-ordered
one that did.

The answer here turned out to be unambiguous: the reranker uses 100% of the
available headroom at every candidate count, so every remaining failure is a
first-stage recall miss. That rules out two plausible next steps (a stronger
reranker, a larger candidate pool) and points at the ones that remain — a better
embedding model, contextual retrieval, different chunking.

Without this measurement the obvious move would have been "try more candidates",
which the sweep shows would have bought exactly nothing.

---

## Chunk boundaries come from a fixed reference tokenizer, not the active model

**Chosen because** an ablation must vary one thing. Tokenizers disagree: the
same corpus chunked with bge-base's WordPiece and bge-m3's SentencePiece
produced 8,574 and 9,392 chunks respectively. Letting the chunker follow the
embedding model meant every embedding experiment silently changed the corpus
too, and the resulting comparison would have measured two variables at once.

**Cost.** `token_count` is exact for the reference model and approximate for
others, and a model with a smaller context window might not fit a chunk built
for the reference. That is why `check_no_truncation()` measures the real token
count under the active model and refuses rather than truncating silently — the
alternative is a passage whose tail is findable by BM25 and invisible to dense
retrieval.

**Different context.** If only one embedding model will ever be used,
chunking to that model's tokenizer is strictly better. The fixed reference only
earns its cost when models are compared.

---

## bge-m3 over bge-base-en-v1.5

**Chosen because** it measurably raises the first-stage ceiling from 0.867 to
0.917 — see `results/embedding_ablation.md`. That is the number that matters:
the candidate sweep established that the reranker already uses 100% of the
headroom it is given, so the only route to a better final answer is putting
more gold passages into the candidate pool.

Dense recall@5 rose from 0.600 to 0.700 on identical chunks.

**Why the ablation is trustworthy.** BM25 does not use embeddings, so it is the
control: it is identical to three decimals across both runs. If it had moved,
the comparison would have been void.

**Cost.** 1024 dimensions rather than 768, so a larger index and a second
vector column (a pgvector column has a fixed width, so two models of different
width cannot share one). bge-m3 is also a considerably larger model to load.

**Different context.** For an English-only corpus with a hard memory budget,
bge-base is 33% narrower and gives up about 5 points of ceiling. That is a
reasonable trade to make deliberately, and it is why the model stays selectable
rather than being deleted.

---

## Chunk profile `t480`, and keeping `t1024` in the repository

**Chosen because** it measures best: page-level recall@5 of 0.811 against 0.794
for `t256` and 0.639 for `t1024`, and recall@10 of 0.911 against 0.861 and
0.767.

**Why `t1024` stays.** It is the evidence that the conclusion survived two
measurement bugs. Its first reading was 0.483, and both the page-attribution
artifact and the reranker-truncation artifact had to be fixed before the real
figure emerged. Deleting the profile would leave the README asserting that
larger chunks are worse with nothing to show for it.

It also exercises a path nothing else does: `t1024` is the only profile
`bge-base` cannot embed, so it is what proves the truncation guard fires rather
than corrupting an index.

---

## Reranker candidate count 50, chosen against a rising ceiling

**Chosen because** it is the measured optimum, not the largest pool. The sweep
shows the first-stage ceiling still climbing at 100 candidates (0.950), while
achieved recall@5 *falls* from 0.883 to 0.850 — the reranker uses 96.4% of the
headroom at 50 and only 89.5% at 100.

More candidates means more plausible distractors, and past a point the
cross-encoder loses more to confusion than the wider pool contributes. This is
the opposite of the earlier finding under `bge-base`, where the reranker
absorbed 100% of headroom at every pool size and the first stage was the sole
bottleneck. Improving one stage moved the constraint to the other, which is
worth stating because it means the sweep has to be re-run after any first-stage
change rather than assumed to still hold.

**Cost.** 687 ms median at 50 candidates against 187 ms at 10, for 5 points of
recall.
