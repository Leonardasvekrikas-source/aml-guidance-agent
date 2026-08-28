"""Retrieval metrics.

Definitions are written out because the name of a metric is not its definition,
and two systems reporting "recall@5" often mean different things.
"""

from __future__ import annotations

from typing import Sequence


def recall_at_k(retrieved: Sequence[int], gold: Sequence[int], k: int) -> float:
    """Fraction of gold chunks appearing in the top k.

    This is per-question recall over the gold SET, not hit-rate. A question
    with three gold chunks of which two are retrieved scores 0.667, not 1.0.
    Hit-rate — did we get at least one — flatters multi-chunk questions, and
    the difference matters when questions vary in how many chunks answer them.
    """
    if not gold:
        return 0.0
    top = set(retrieved[:k])
    return sum(1 for g in gold if g in top) / len(gold)


def hit_at_k(retrieved: Sequence[int], gold: Sequence[int], k: int) -> float:
    """1.0 if any gold chunk is in the top k. Reported alongside recall so the
    difference between the two is visible rather than hidden by a label."""
    if not gold:
        return 0.0
    return 1.0 if set(retrieved[:k]) & set(gold) else 0.0


def reciprocal_rank(retrieved: Sequence[int], gold: Sequence[int]) -> float:
    """1 / rank of the FIRST gold chunk, or 0 if none was retrieved.

    Measures how far a reader has to scan before hitting something correct.
    Averaged over questions this is MRR.
    """
    gold_set = set(gold)
    for position, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in gold_set:
            return 1.0 / position
    return 0.0


def precision_at_k(retrieved: Sequence[int], gold: Sequence[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    gold_set = set(gold)
    return sum(1 for c in top if c in gold_set) / len(top)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
