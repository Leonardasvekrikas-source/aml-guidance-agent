"""Database access.

Thin deliberately. There is no ORM here because every query in this project is
either a bulk insert or a similarity search, and both are clearer as SQL than
as an ORM expression that compiles to SQL you then have to reverse-engineer.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import DictRow, dict_row

from .config import settings


@contextlib.contextmanager
def connect(autocommit: bool = False) -> Iterator[psycopg.Connection[DictRow]]:
    """Open a connection with pgvector's type adapters registered.

    Without ``register_vector`` a ``vector`` column comes back as a string and
    every embedding silently becomes text, which fails much later and much
    less obviously than it should.
    """
    try:
        # Parametrised on DictRow so every caller gets dict rows, and a type
        # checker knows it. Without this, row["id"] type-checks as a tuple
        # index and every access is an error.
        conn: psycopg.Connection[DictRow] = psycopg.connect(
            settings.dsn, autocommit=autocommit, row_factory=dict_row
        )
    except psycopg.OperationalError as exc:
        raise RuntimeError(
            f"cannot reach Postgres at {settings.pg_host}:{settings.pg_port}. "
            "Is the stack up? Try `make up` (or `.\run.ps1 up` on Windows).\n"
            f"psycopg said: {exc}"
        ) from exc

    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()


def upsert_document(conn: psycopg.Connection[DictRow], doc: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO documents (
            id, title, publisher, source_url, publication_date,
            retrieved_at, sha256, page_count, doc_type
        )
        VALUES (
            %(id)s, %(title)s, %(publisher)s, %(source_url)s, %(publication_date)s,
            %(retrieved_at)s, %(sha256)s, %(page_count)s, %(doc_type)s
        )
        ON CONFLICT (id) DO UPDATE SET
            title            = EXCLUDED.title,
            publisher        = EXCLUDED.publisher,
            source_url       = EXCLUDED.source_url,
            publication_date = EXCLUDED.publication_date,
            retrieved_at     = EXCLUDED.retrieved_at,
            sha256           = EXCLUDED.sha256,
            page_count       = EXCLUDED.page_count,
            doc_type         = EXCLUDED.doc_type
        """,
        doc,
    )


def delete_chunks(conn: psycopg.Connection[DictRow], document_id: str, profile: str) -> int:
    cur = conn.execute(
        "DELETE FROM chunks WHERE document_id = %s AND chunk_profile = %s",
        (document_id, profile),
    )
    return cur.rowcount


def insert_chunks(conn: psycopg.Connection[DictRow], rows: Sequence[dict[str, Any]]) -> int:
    """Insert a batch of chunks.

    Re-ingestion deletes the profile's chunks for a document first, so this
    does not need conflict handling; a conflict here means the caller skipped
    that step and should hear about it rather than have it silently ignored.
    """
    if not rows:
        return 0
    # The embedding column is chosen by the active model, not by the caller:
    # a 1024-wide vector cannot go into a 768-wide column, and letting call
    # sites pick would make that a runtime surprise.
    column = settings.embedding_column
    with conn.cursor() as cur:
        cur.executemany(
            f"""
            INSERT INTO chunks (
                document_id, chunk_profile, chunk_index, page, page_end,
                section_heading, text, char_count, token_count, {column}
            )
            VALUES (
                %(document_id)s, %(chunk_profile)s, %(chunk_index)s, %(page)s,
                %(page_end)s, %(section_heading)s, %(text)s, %(char_count)s,
                %(token_count)s,
                %(embedding)s
            )
            ON CONFLICT (document_id, chunk_profile, chunk_index) DO UPDATE
                SET {column} = EXCLUDED.{column}
            """,
            rows,
        )
    return len(rows)


def fetch_chunks(
    conn: psycopg.Connection[DictRow],
    profile: str,
    with_embeddings: bool = False,
) -> list[dict[str, Any]]:
    """Every chunk in a profile, ordered deterministically.

    Order matters: BM25 is built over this list and its internal indices must
    line up with chunk ids the same way on every run, or a benchmark is not
    reproducible.
    """
    embedding_col = f", {settings.embedding_column}" if with_embeddings else ""
    cur = conn.execute(
        f"""
        SELECT c.id, c.document_id, c.chunk_index, c.page, c.page_end, c.section_heading,
               c.text, c.char_count, c.token_count,
               d.title, d.publisher, d.source_url, d.publication_date
               {embedding_col}
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.chunk_profile = %s
        ORDER BY c.document_id, c.chunk_index
        """,
        (profile,),
    )
    return list(cur)


def corpus_stats(conn: psycopg.Connection[DictRow]) -> dict[str, Any]:
    # fetchone() is Optional. A count query always returns a row, but saying
    # so explicitly is cheaper than an AttributeError in six months.
    row = conn.execute("SELECT count(*) AS n FROM documents").fetchone()
    documents = row["n"] if row else 0
    per_profile = list(
        conn.execute(
            """
            SELECT chunk_profile,
                   count(*)                              AS chunks,
                   count(embedding)                      AS embedded_768,
                   count(embedding_lg)                   AS embedded_1024,
                   round(avg(char_count))                AS avg_chars,
                   round(avg(token_count))               AS avg_tokens,
                   count(DISTINCT document_id)           AS documents
            FROM chunks
            GROUP BY chunk_profile
            ORDER BY chunk_profile
            """
        )
    )
    return {"documents": documents, "profiles": per_profile}
