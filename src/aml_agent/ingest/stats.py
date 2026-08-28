"""Corpus statistics, for the README and for sanity-checking an ingestion."""

from __future__ import annotations

import json

from ..config import settings
from ..db import connect, corpus_stats


def main() -> int:
    with connect() as conn:
        stats = corpus_stats(conn)

        print(f"documents: {stats['documents']}")
        if not stats["profiles"]:
            print("no chunks ingested yet — run `make ingest`")
            return 1

        # Both embedding columns are reported. A profile fully embedded by one
        # model and not the other is the normal state while an embedding
        # ablation is in progress, and it must not look like corruption.
        print(f"\n{'profile':<10}{'docs':>6}{'chunks':>9}{'768d':>8}{'1024d':>8}"
              f"{'avg chars':>11}{'avg tokens':>12}")
        for row in stats["profiles"]:
            print(
                f"{row['chunk_profile']:<10}{row['documents']:>6}{row['chunks']:>9}"
                f"{row['embedded_768']:>8}{row['embedded_1024']:>8}"
                f"{row['avg_chars'] or 0:>11}{row['avg_tokens'] or 0:>12}"
            )

        active = "embedded_768" if settings.embedding_dim == 768 else "embedded_1024"
        missing = [r for r in stats["profiles"] if r["chunks"] != r[active]]
        if missing:
            print(
                f"\nWARNING: some chunks have no {settings.embedding_key} embedding. "
                "Dense retrieval will silently miss them. Re-run ingestion."
            )

        by_publisher = list(
            conn.execute(
                "SELECT publisher, count(*) AS n FROM documents "
                "GROUP BY publisher ORDER BY n DESC, publisher"
            )
        )
        print("\ndocuments by publisher:")
        for row in by_publisher:
            print(f"  {row['publisher']:<48}{row['n']:>3}")

        settings.results_dir.mkdir(parents=True, exist_ok=True)
        out = settings.results_dir / "corpus.json"
        out.write_text(
            json.dumps(
                {
                    "documents": stats["documents"],
                    "profiles": [dict(r) for r in stats["profiles"]],
                    "by_publisher": [dict(r) for r in by_publisher],
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {out.relative_to(settings.results_dir.parent)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
