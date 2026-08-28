"""Reciprocal Rank Fusion.

RRF combines ranked lists using only positions, never scores:

    score(d) = sum over lists L of  1 / (k + rank_L(d))

Using ranks rather than scores is the whole point. BM25 returns unbounded
positive numbers whose scale depends on corpus statistics; cosine similarity
is bounded in [-1, 1]. There is no principled constant that puts those on a
common scale, and normalising them per-query makes fusion depend on the score
distribution of whichever query happened to be asked. Ranks are comparable by
construction.

k = 60 is the value from the original paper. It damps the influence of the top
rank so that one retriever's confident first place cannot by itself outvote
broad agreement further down the other list.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from .base import Hit, Retriever

RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[Hit]],
    k: int = 10,
    rrf_k: int = RRF_K,
) -> list[Hit]:
    scores: dict[int, float] = defaultdict(float)
    contributions: dict[int, dict[str, int]] = defaultdict(dict)
    representative: dict[int, Hit] = {}

    for retriever_name, hits in ranked_lists.items():
        for hit in hits:
            scores[hit.chunk_id] += 1.0 / (rrf_k + hit.rank)
            contributions[hit.chunk_id][retriever_name] = hit.rank
            # Keep whichever hit we saw first as the source of chunk metadata;
            # the text and provenance are identical across retrievers because
            # they come from the same row.
            representative.setdefault(hit.chunk_id, hit)

    ordered = sorted(
        scores.items(),
        # Ties broken by best single rank, then by chunk id, so the output is
        # deterministic. A benchmark whose ordering depends on dict iteration
        # is not reproducible.
        key=lambda item: (-item[1], min(contributions[item[0]].values()), item[0]),
    )

    fused: list[Hit] = []
    for rank, (chunk_id, score) in enumerate(ordered[:k], start=1):
        base = representative[chunk_id]
        fused.append(
            replace(
                base,
                score=score,
                rank=rank,
                retriever="hybrid-rrf",
                components={"ranks": dict(contributions[chunk_id])},
            )
        )
    return fused


class HybridRetriever:
    name = "hybrid"

    def __init__(self, lexical: Retriever, dense: Retriever, candidate_k: int = 50):
        self.lexical = lexical
        self.dense = dense
        # Each retriever contributes more candidates than the final k, so that
        # a chunk ranked 30th by one retriever and 3rd by the other can still
        # surface. Fusing only the top-k of each would discard exactly the
        # agreement RRF exists to exploit.
        self.candidate_k = candidate_k

    def search(self, query: str, k: int = 10) -> list[Hit]:
        return reciprocal_rank_fusion(
            {
                self.lexical.name: self.lexical.search(query, self.candidate_k),
                self.dense.name: self.dense.search(query, self.candidate_k),
            },
            k=k,
        )
