"""Full ingestion: download, extract, chunk, embed, load.

Idempotent by design. Re-running replaces a document's chunks for each profile
rather than appending, so ingesting twice does not silently double the corpus
and quietly change every retrieval metric.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone

from ..config import CHUNK_PROFILES, settings
from ..db import connect, delete_chunks, insert_chunks, upsert_document
from .chunk import chunk_document
from .download import download_all
from .embed import embed_passages
from .extract import ExtractionError, extract_pages, page_count
from .manifest import load_manifest


def ingest(skip_download: bool = False) -> int:
    entries = load_manifest()
    print(f"manifest: {len(entries)} documents")

    if skip_download:
        print("\nskipping download (--skip-download)")
        available = {e.id for e in entries if e.raw_path().exists()}
    else:
        print("\n=== download ===")
        results = download_all()
        available = {r.entry.id for r in results if r.ok}

    if not available:
        print("\nnothing available to ingest.")
        return 1

    print(f"\n=== extract, chunk, embed, load ===")
    retrieved_at = datetime.now(timezone.utc)
    failures: list[tuple[str, str]] = []
    loaded_documents = 0
    loaded_chunks = 0

    with connect() as conn:
        for entry in entries:
            if entry.id not in available:
                continue

            path = entry.raw_path()
            try:
                pages = extract_pages(path)
            except ExtractionError as exc:
                failures.append((entry.id, str(exc)))
                print(f"  ! {entry.id}: extraction failed")
                continue

            from .download import sha256_of

            upsert_document(
                conn,
                {
                    "id": entry.id,
                    "title": entry.title,
                    "publisher": entry.publisher,
                    "source_url": entry.source_url,
                    "publication_date": entry.publication_date,
                    "retrieved_at": retrieved_at,
                    "sha256": sha256_of(path),
                    "page_count": page_count(path),
                    "doc_type": entry.doc_type,
                },
            )

            document_chunk_total = 0
            for profile in CHUNK_PROFILES:
                chunks = chunk_document(entry.id, pages, profile)
                if not chunks:
                    continue

                vectors = embed_passages([c.text for c in chunks])

                # Replace rather than append: re-ingesting must not double the
                # corpus and silently move every retrieval number.
                delete_chunks(conn, entry.id, profile.name)
                insert_chunks(
                    conn,
                    [
                        {
                            "document_id": c.document_id,
                            "chunk_profile": c.chunk_profile,
                            "chunk_index": c.chunk_index,
                            "page": c.page,
                            "section_heading": c.section_heading,
                            "text": c.text,
                            "char_count": c.char_count,
                            "token_count": c.token_count,
                            "embedding": vector,
                        }
                        for c, vector in zip(chunks, vectors)
                    ],
                )
                document_chunk_total += len(chunks)

            conn.commit()
            loaded_documents += 1
            loaded_chunks += document_chunk_total
            print(
                f"  + {entry.id:<40} {len(pages):>4} pages  "
                f"{document_chunk_total:>5} chunks across {len(CHUNK_PROFILES)} profiles"
            )

    print(f"\nloaded {loaded_documents} documents, {loaded_chunks} chunks")

    if failures:
        print(f"\n{len(failures)} document(s) failed extraction:")
        for doc_id, message in failures:
            print(f"  {doc_id}: {message}")
        print("\nLog these in findings/what-broke.md rather than removing them quietly.")

    return 0


def main() -> int:
    return ingest(skip_download="--skip-download" in sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
