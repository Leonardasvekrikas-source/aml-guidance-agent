"""Cross-encoder reranking.

The retrievers in this package are all *bi-encoders or bag-of-words*: they
score a query against a passage without ever letting the two interact. BM25
counts term overlap; dense retrieval compares two vectors that were computed
independently, so the passage embedding was fixed before the query existed.
That independence is what makes them fast enough to search 8,000 chunks, and
it is also their ceiling — neither can represent "this passage answers *that*
question" as opposed to "this passage is about the same subject".

A cross-encoder puts the query and the passage in the same forward pass and
outputs a relevance score directly. Every pair costs a full model evaluation,
so it cannot search a corpus — but over 50 candidates it is cheap, and it sees
interactions the first stage structurally cannot.

Hence the standard two-stage shape used here: retrieve broadly and cheaply,
then rerank narrowly and expensively.

    hybrid RRF (50 candidates)  ->  cross-encoder  ->  top k

The cost is real and reported rather than hidden: reranking adds a model
forward pass per candidate, which dominates the latency figures in the README.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from ..config import settings
from .base import Hit, Retriever


@lru_cache(maxsize=2)
def get_reranker(model_name: str | None = None, device: str | None = None):
    """Load the cross-encoder once per process.

    Falls back to CPU when CUDA is unavailable. Reranking on CPU is slow enough
    to be unpleasant but not slow enough to be useless at this candidate count,
    and failing outright would make the repository unrunnable for anyone
    without a GPU.
    """
    import torch
    from sentence_transformers import CrossEncoder

    name = model_name or settings.reranker_model
    requested = device or settings.embedding_device

    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("  CUDA requested but not available; reranking on CPU instead")
        requested = "cpu"

    # max_length must cover the largest chunk profile. At 512 a t1024 chunk
    # is scored on its first half only, and the missing half is invisible —
    # no error, just a worse ranking that looks like the chunk size being
    # bad.
    return CrossEncoder(name, device=requested, max_length=settings.rerank_max_length)


class RerankedRetriever:
    """Wraps a first-stage retriever with a cross-encoder second stage."""

    def __init__(
        self,
        base: Retriever,
        name: str = "hybrid+rerank",
        candidate_k: int | None = None,
        model_name: str | None = None,
    ):
        self.base = base
        self.name = name
        # More candidates means a better chance the right passage is somewhere
        # in the pool, and a linearly higher reranking cost. The reranker can
        # only reorder what the first stage handed it: if the gold passage is
        # not in these candidates, no amount of reranking will find it, and
        # first-stage recall@candidate_k is the hard ceiling on the whole
        # pipeline.
        self.candidate_k = candidate_k or settings.rerank_candidates
        self.model_name = model_name or settings.reranker_model

    def search(self, query: str, k: int = 10) -> list[Hit]:
        candidates = self.base.search(query, self.candidate_k)
        if not candidates:
            return []

        model = get_reranker(self.model_name)
        _warn_if_truncating(model, candidates)
        scores = model.predict(
            [(query, hit.text) for hit in candidates],
            batch_size=settings.rerank_batch_size,
            show_progress_bar=False,
        )

        ordered = sorted(
            # strict=True: the model must return exactly one score per
            # candidate. A silent length mismatch would misalign every
            # score with the wrong passage and still produce a ranking.
            zip(candidates, (float(s) for s in scores), strict=True),
            # Ties broken by chunk id so the ranking is deterministic; a
            # benchmark whose order depends on sort stability is not
            # reproducible.
            key=lambda pair: (-pair[1], pair[0].chunk_id),
        )

        reranked: list[Hit] = []
        for rank, (hit, score) in enumerate(ordered[:k], start=1):
            reranked.append(
                replace(
                    hit,
                    score=score,
                    rank=rank,
                    retriever=self.name,
                    components={
                        **hit.components,
                        "first_stage_rank": hit.rank,
                        "first_stage_retriever": hit.retriever,
                    },
                )
            )
        return reranked


_TRUNCATION_WARNED: set[str] = set()


def _warn_if_truncating(model, candidates: list[Hit]) -> None:
    """Say so, once, if the reranker cannot see whole passages.

    Truncation here is invisible: the model returns a score either way, and the
    only symptom is a slightly worse ranking that looks like the chunk size
    being wrong. Silence is exactly what makes it dangerous.
    """
    window = settings.rerank_max_length
    longest = max((hit.token_count or 0) for hit in candidates) if candidates else 0
    if not longest or longest + 32 <= window:
        return

    key = f"{settings.reranker_model}:{window}"
    if key in _TRUNCATION_WARNED:
        return
    _TRUNCATION_WARNED.add(key)
    print(
        f"  WARNING: reranker window is {window} tokens but candidates reach "
        f"{longest}. Passages are being scored on their opening only. "
        f"Raise RERANK_MAX_LENGTH above the chunk profile in use."
    )


def first_stage_ceiling(base: Retriever, query: str, candidate_k: int) -> list[int]:
    """Chunk ids the reranker will be allowed to choose from.

    Exposed so the evaluation can report the ceiling separately from the
    achieved score. The gap between "the gold passage was in the candidate
    pool" and "the reranker put it in the top k" is the reranker's own error
    rate, and conflating the two hides which stage to improve.
    """
    return [hit.chunk_id for hit in base.search(query, candidate_k)]
