"""Resolving gold passages across chunk profiles.

The evaluation set names gold passages by chunk id, and those ids belong to the
profile the questions were authored against (`t480`). Chunks are separate rows
per profile, so those ids do not exist in `t256` — which means chunk-id gold
cannot be used to compare chunk sizes at all. Scoring `t256` against `t480` ids
would report zero recall for every question and look like catastrophic
retrieval failure rather than a category error in the metric.

Two gold definitions are therefore used, and both are reported:

**Exact (chunk-id).** A retrieved chunk counts only if it is literally the gold
chunk. Strictest and cleanest, but only meaningful for the authoring profile.
This is the primary number.

**Page-level.** Gold is the set of (document, page) pairs the gold chunks came
from, and a retrieved chunk counts if it comes from one of those pages. This is
profile-independent, so it is the only definition under which a chunk-size
comparison means anything. It is looser — a page can hold several chunks, and
any of them counts — so page-level figures are systematically higher than exact
ones and the two must never be compared with each other.

The alternative, remapping gold by text overlap, was rejected: a smaller
profile splits one gold chunk into two, each a partial match, so any overlap
threshold hands the smaller profile extra chances at the same passage and
quietly biases the very comparison the experiment exists to make.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..db import connect


@dataclass(frozen=True)
class GoldResolution:
    """Gold for one question, in both definitions."""

    chunk_ids: frozenset[int]
    pages: frozenset[tuple[str, int]]  # (document_id, page)


def resolve_gold(
    gold_chunk_ids: Iterable[int],
    authoring_profile: str = "t480",
) -> dict[int, tuple[str, int | None]]:
    """Map gold chunk ids to their (document_id, page).

    Looked up once for the whole evaluation set rather than per question.
    """
    wanted = sorted({int(g) for g in gold_chunk_ids})
    if not wanted:
        return {}

    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, document_id, page
            FROM chunks
            WHERE id = ANY(%s) AND chunk_profile = %s
            """,
            (wanted, authoring_profile),
        ).fetchall()

    return {row["id"]: (row["document_id"], row["page"]) for row in rows}


def page_key(document_id: str, page: int | None) -> tuple[str, int]:
    """Page 0 stands for "page unknown", so chunks with no page still group."""
    return (document_id, page if page is not None else 0)


def gold_pages(
    gold_chunk_ids: Iterable[int],
    location: dict[int, tuple[str, int | None]],
) -> frozenset[tuple[str, int]]:
    pages = set()
    for chunk_id in gold_chunk_ids:
        if chunk_id in location:
            document_id, page = location[chunk_id]
            pages.add(page_key(document_id, page))
    return frozenset(pages)


def retrieved_pages(hits) -> list[tuple[str, int]]:
    """Page keys of retrieved hits, in rank order, de-duplicated.

    De-duplication matters: two chunks from the same page are one page-level
    hit, and counting them twice would let a retriever fill the top k with one
    page and score as though it had found several.
    """
    seen: set[tuple[str, int]] = set()
    ordered: list[tuple[str, int]] = []
    for hit in hits:
        key = page_key(hit.document_id, hit.page)
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered
