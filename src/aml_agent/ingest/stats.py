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

        print(f"\n{'profile':<10}{'docs':>6}{'chunks':>9}{'embedded':>10}"
              f"{'avg chars':>11}{'avg tokens':>12}")
        for row in stats["profiles"]:
            print(
                f"{row['chunk_profile']:<10}{row['documents']:>6}{row['chunks']:>9}"
                f"{row['embedded']:>10}{row['avg_chars'] or 0:>11}"
                f"{row['avg_tokens'] or 0:>12}"
            )

        missing = [r for r in stats["profiles"] if r["chunks"] != r["embedded"]]
        if missing:
            print("\nWARNING: some chunks have no embedding. Dense retrieval will "
                  "silently miss them. Re-run ingestion.")

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
