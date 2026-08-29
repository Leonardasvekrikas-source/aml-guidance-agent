"""Reciprocal Rank Fusion.

RRF uses rank positions and never scores. These tests pin that down, because
"improving" it to blend normalised scores would make fusion depend on each
query's score distribution and quietly change every hybrid number.
"""

from __future__ import annotations

import pytest

from aml_agent.retrieval.base import Hit
from aml_agent.retrieval.fusion import RRF_K, reciprocal_rank_fusion


def hit(chunk_id: int, rank: int, retriever: str, score: float = 0.0) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        score=score,
        rank=rank,
        text=f"text {chunk_id}",
        document_id="doc",
        title="Doc",
        publisher="Pub",
        source_url="https://example.org/d.pdf",
        page=1,
        section_heading=None,
        retriever=retriever,
    )


class TestFusion:
    def test_agreement_beats_a_single_strong_result(self):
        # Chunk 2 is ranked 2nd by both retrievers; chunk 1 is 1st in one list
        # and absent from the other. RRF should prefer the agreed-upon chunk.
        #   chunk 1: 1/(60+1)                = 0.01639
        #   chunk 2: 1/(60+2) + 1/(60+2)     = 0.03226
        fused = reciprocal_rank_fusion(
            {
                "bm25": [hit(1, 1, "bm25"), hit(2, 2, "bm25")],
                "dense": [hit(3, 1, "dense"), hit(2, 2, "dense")],
            },
            k=3,
        )
        assert fused[0].chunk_id == 2

    def test_scores_are_ignored_entirely(self):
        # Identical ranks, wildly different scores. Fusion must not care.
        with_small = reciprocal_rank_fusion(
            {"a": [hit(1, 1, "a", score=0.001)], "b": [hit(1, 1, "b", score=0.002)]}, k=1
        )
        with_huge = reciprocal_rank_fusion(
            {"a": [hit(1, 1, "a", score=9999.0)], "b": [hit(1, 1, "b", score=8888.0)]}, k=1
        )
        assert with_small[0].score == pytest.approx(with_huge[0].score)

    def test_score_formula(self):
        fused = reciprocal_rank_fusion({"a": [hit(5, 1, "a")]}, k=1)
        assert fused[0].score == pytest.approx(1.0 / (RRF_K + 1))

    def test_ranks_are_renumbered_from_one(self):
        fused = reciprocal_rank_fusion({"a": [hit(1, 1, "a"), hit(2, 2, "a"), hit(3, 3, "a")]}, k=3)
        assert [f.rank for f in fused] == [1, 2, 3]

    def test_components_record_where_each_hit_came_from(self):
        # Needed to explain a hybrid result in an interview or a trace.
        fused = reciprocal_rank_fusion(
            {"bm25": [hit(7, 3, "bm25")], "dense": [hit(7, 1, "dense")]}, k=1
        )
        assert fused[0].components["ranks"] == {"bm25": 3, "dense": 1}
        assert fused[0].retriever == "hybrid-rrf"

    def test_deterministic_ordering_on_ties(self):
        # Two chunks with identical RRF scores must always come back in the
        # same order, or the benchmark is not reproducible.
        lists = {
            "a": [hit(20, 1, "a"), hit(10, 2, "a")],
            "b": [hit(10, 1, "b"), hit(20, 2, "b")],
        }
        first = [h.chunk_id for h in reciprocal_rank_fusion(lists, k=2)]
        second = [h.chunk_id for h in reciprocal_rank_fusion(lists, k=2)]
        assert first == second

    def test_empty_lists_produce_no_hits(self):
        assert reciprocal_rank_fusion({"a": [], "b": []}, k=5) == []

    def test_k_truncates_output(self):
        fused = reciprocal_rank_fusion({"a": [hit(i, i, "a") for i in range(1, 11)]}, k=3)
        assert len(fused) == 3
