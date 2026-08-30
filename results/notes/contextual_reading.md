**BM25 is NOT a control here.** Unlike the embedding-model ablation, this
changes the indexed text itself, so lexical retrieval legitimately moves with
it — and it did, +0.033 recall@5. The situating sentence adds searchable
vocabulary the passage never contained, which is exactly the intended effect
and is why the control argument does not transfer between the two experiments.

**The ceiling is the number that matters.** The candidate sweep established
that the reranker cannot exceed the fraction of gold passages reaching the
candidate pool. That ceiling moved 0.917 → 0.983: almost every gold passage now
reaches the pool, which is what this was aimed at.

**MRR falls slightly on the first-stage retrievers while recall rises.**
Situating sentences make more chunks look plausible, so the right one is
retrieved more often but ranked marginally less sharply. The reranker more than
recovers it — MRR 0.733 → 0.777 — which is the division of labour the two
stages exist for.
