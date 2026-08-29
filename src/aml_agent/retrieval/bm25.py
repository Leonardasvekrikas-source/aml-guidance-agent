"""Lexical retrieval with Okapi BM25.

Real BM25 via rank_bm25, computed in the application process, not Postgres
`ts_rank`. ts_rank is a different scoring function with different term
saturation behaviour; calling it BM25 in a README would be inaccurate, and the
corpus is small enough to hold the index in memory.

The cost of that choice is stated in DECISIONS.md: the index is rebuilt on
startup and is not a queryable database artifact.
"""

from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi

from ..db import connect, fetch_chunks
from .base import Hit, row_to_hit

TOKEN = re.compile(r"[a-z0-9]+")

# Very common words carry no discriminative signal and slow scoring. This is a
# deliberately short list: aggressive stopword removal hurts phrase-like
# queries, which are common in regulatory text ("source of funds").
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the "
    "this to was were will with".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class BM25Retriever:
    name = "bm25"

    def __init__(self, profile: str, rows: list[dict[str, Any]] | None = None):
        self.profile = profile
        if rows is None:
            with connect() as conn:
                rows = fetch_chunks(conn, profile)
        if not rows:
            raise RuntimeError(f"no chunks for profile {profile!r}. Run `make ingest` first.")
        self.rows = rows
        self.index = BM25Okapi([tokenize(row["text"]) for row in rows])

    def search(self, query: str, k: int = 10) -> list[Hit]:
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.index.get_scores(tokens)

        # argsort descending, then drop zero-scored documents. A BM25 score of
        # zero means no query term appears at all; returning those to pad k
        # would inflate recall@k with documents the retriever did not actually
        # match.
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits: list[Hit] = []
        for rank, position in enumerate(order[:k], start=1):
            if scores[position] <= 0:
                break
            hits.append(row_to_hit(self.rows[position], scores[position], rank, self.name))
        return hits
