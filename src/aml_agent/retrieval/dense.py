"""Dense retrieval over pgvector.

The query is embedded with the model's query instruction prefix and compared
by cosine distance, which matches both the normalised vectors produced at
ingestion and the `vector_cosine_ops` index in the migration. Getting any one
of those three out of step degrades recall silently rather than erroring.
"""

from __future__ import annotations

from ..ingest.embed import embed_query
from ..db import connect
from .base import Hit, row_to_hit


class DenseRetriever:
    name = "dense"

    def __init__(self, profile: str):
        self.profile = profile

    def search(self, query: str, k: int = 10) -> list[Hit]:
        vector = embed_query(query)

        with connect() as conn:
            rows = list(
                conn.execute(
                    """
                    SELECT c.id, c.document_id, c.page, c.section_heading, c.text,
                           d.title, d.publisher, d.source_url,
                           c.embedding <=> %s AS distance
                    FROM chunks c
                    JOIN documents d ON d.id = c.document_id
                    WHERE c.chunk_profile = %s
                      AND c.embedding IS NOT NULL
                    ORDER BY c.embedding <=> %s
                    LIMIT %s
                    """,
                    (vector, self.profile, vector, k),
                )
            )

        # Cosine distance in [0, 2]; similarity = 1 - distance so that higher is
        # better, consistent with BM25 and with what RRF expects to rank.
        return [
            row_to_hit(row, 1.0 - float(row["distance"]), rank, self.name)
            for rank, row in enumerate(rows, start=1)
        ]
