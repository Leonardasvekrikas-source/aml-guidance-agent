**BM25 is the control here.** It does not use embeddings, so swapping the
embedding model must leave it untouched — and it is identical to three decimals
across both runs. Had it moved, the comparison would have been void.

**The ceiling is the number that matters.** It is the fraction of gold passages
reaching the candidate pool, and the reranker cannot exceed it. It moved 0.867 →
0.917, so the better model puts genuinely more of the gold set within reach
rather than merely reordering what was already there.
