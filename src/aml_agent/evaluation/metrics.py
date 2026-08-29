"""Retrieval metrics.

Definitions are written out because the name of a metric is not its definition,
and two systems reporting "recall@5" often mean different things.

The functions are generic over the identifier type. Gold is a chunk id under
exact scoring and a (document, page) pair under page-level scoring, and the
arithmetic is identical either way — so the type variable is not decoration, it
is what lets one implementation serve both without a cast that would hide a
genuine mix-up between the two.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TypeVar

ItemId = TypeVar("ItemId", bound=Hashable)


def recall_at_k(retrieved: Sequence[ItemId], gold: Sequence[ItemId], k: int) -> float:
    """Fraction of gold items appearing in the top k.

    This is per-question recall over the gold SET, not hit-rate. A question
    with three gold chunks of which two are retrieved scores 0.667, not 1.0.
    Hit-rate — did we get at least one — flatters multi-chunk questions, and
    the difference matters when questions vary in how many chunks answer them.
    """
    if not gold:
        return 0.0
    top = set(retrieved[:k])
    return sum(1 for g in gold if g in top) / len(gold)


def hit_at_k(retrieved: Sequence[ItemId], gold: Sequence[ItemId], k: int) -> float:
    """1.0 if any gold item is in the top k.

    Reported alongside recall so the difference between the two is visible
    rather than hidden by a label.
    """
    if not gold:
        return 0.0
    return 1.0 if set(retrieved[:k]) & set(gold) else 0.0


def reciprocal_rank(retrieved: Sequence[ItemId], gold: Sequence[ItemId]) -> float:
    """1 / rank of the FIRST gold item, or 0 if none was retrieved.

    Measures how far a reader has to scan before hitting something correct.
    Averaged over questions this is MRR.
    """
    gold_set = set(gold)
    for position, item in enumerate(retrieved, start=1):
        if item in gold_set:
            return 1.0 / position
    return 0.0


def precision_at_k(retrieved: Sequence[ItemId], gold: Sequence[ItemId], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    gold_set = set(gold)
    return sum(1 for c in top if c in gold_set) / len(top)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
