"""Shared retrieval types.

Every retriever returns the same shape, which is what makes the three
implementations comparable on the same evaluation set and fusible by RRF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Hit:
    chunk_id: int
    score: float
    rank: int  # 1-based position in this retriever's ranking
    text: str
    document_id: str
    title: str
    publisher: str
    source_url: str
    page: int | None
    page_end: int | None
    section_heading: str | None
    retriever: str
    token_count: int | None = None
    components: dict[str, Any] = field(default_factory=dict)

    def citation(self) -> str:
        location = f"p.{self.page}" if self.page else "page unknown"
        return f"{self.title} ({self.publisher}), {location}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": round(self.score, 6),
            "rank": self.rank,
            "document_id": self.document_id,
            "title": self.title,
            "publisher": self.publisher,
            "source_url": self.source_url,
            "page": self.page,
            "page_end": self.page_end,
            "section_heading": self.section_heading,
            "retriever": self.retriever,
            "components": self.components,
        }


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[Hit]: ...


def row_to_hit(row: dict[str, Any], score: float, rank: int, retriever: str) -> Hit:
    return Hit(
        chunk_id=row["id"],
        score=float(score),
        rank=rank,
        text=row["text"],
        document_id=row["document_id"],
        title=row["title"],
        publisher=row["publisher"],
        source_url=row["source_url"],
        page=row.get("page"),
        page_end=row.get("page_end"),
        section_heading=row.get("section_heading"),
        retriever=retriever,
        token_count=row.get("token_count"),
    )
