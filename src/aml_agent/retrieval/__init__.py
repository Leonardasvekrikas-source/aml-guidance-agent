"""Retrieval implementations and the factory that builds them."""

from __future__ import annotations

from ..config import DEFAULT_PROFILE
from .base import Hit, Retriever
from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .fusion import HybridRetriever, reciprocal_rank_fusion
from .rerank import RerankedRetriever

__all__ = [
    "Hit",
    "Retriever",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "RerankedRetriever",
    "reciprocal_rank_fusion",
    "build_retrievers",
    "build_retriever",
]


def build_retrievers(
    profile: str = DEFAULT_PROFILE,
    with_reranker: bool = True,
) -> dict[str, Retriever]:
    """All three retrievers over one chunk profile.

    The BM25 index is built once here and shared with the hybrid retriever
    rather than rebuilt, because rebuilding it per query would dominate the
    latency numbers reported in the README.
    """
    lexical = BM25Retriever(profile)
    dense = DenseRetriever(profile)
    hybrid = HybridRetriever(lexical, dense)

    retrievers: dict[str, Retriever] = {
        "bm25": lexical,
        "dense": dense,
        "hybrid": hybrid,
    }
    if with_reranker:
        # Wraps hybrid rather than replacing it, so the benchmark reports the
        # first stage and the reranked result side by side and the reranker's
        # contribution is visible rather than assumed.
        retrievers["hybrid+rerank"] = RerankedRetriever(hybrid)
    return retrievers


def build_retriever(name: str = "hybrid", profile: str = DEFAULT_PROFILE) -> Retriever:
    retrievers = build_retrievers(profile)
    if name not in retrievers:
        raise KeyError(f"unknown retriever {name!r}; choose from {sorted(retrievers)}")
    return retrievers[name]
