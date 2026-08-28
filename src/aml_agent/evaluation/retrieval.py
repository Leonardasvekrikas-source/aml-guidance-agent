"""M1: the retrieval benchmark.

Runs BM25, dense and hybrid RRF over the answerable questions, for every chunk
profile, and writes results/retrieval.json.

Two gold definitions are computed side by side — exact chunk-id and page-level.
See `gold.py` for why. The short version: chunk ids belong to the profile the
questions were authored against, so only the page-level figures can be compared
across chunk sizes, and the two definitions must never be compared with each
other.

Per-question rows are kept, not just the averages. The averages are what you
report; the per-question rows are what you learn from.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from ..config import CHUNK_PROFILES, DEFAULT_PROFILE, settings
from ..retrieval import build_retrievers
from .gold import gold_pages, resolve_gold, retrieved_pages
from .metrics import hit_at_k, mean, precision_at_k, recall_at_k, reciprocal_rank
from .questions import load_questions, validate_gold_chunks

MAX_K = 10
AUTHORING_PROFILE = DEFAULT_PROFILE


def evaluate_profile(
    profile: str,
    questions: list,
    location: dict[int, tuple[str, int | None]],
) -> dict[str, Any]:
    answerable = [q for q in questions if q.answerable]
    exact_applies = profile == AUTHORING_PROFILE

    retrievers = build_retrievers(profile)
    per_retriever: dict[str, Any] = {}

    for name, retriever in retrievers.items():
        rows: list[dict[str, Any]] = []
        latencies: list[float] = []

        for question in answerable:
            started = time.perf_counter()
            hits = retriever.search(question.question, MAX_K)
            latencies.append((time.perf_counter() - started) * 1000.0)

            gold_ids = list(question.gold_chunk_ids)
            retrieved_ids = [h.chunk_id for h in hits]

            g_pages = sorted(gold_pages(gold_ids, location))
            r_pages = retrieved_pages(hits)

            row: dict[str, Any] = {
                "question_id": question.id,
                "topic": question.topic,
                "gold_chunk_ids": gold_ids,
                "retrieved_chunk_ids": retrieved_ids,
                "page_recall_at_5": recall_at_k(r_pages, g_pages, 5),
                "page_recall_at_10": recall_at_k(r_pages, g_pages, 10),
                "page_hit_at_5": hit_at_k(r_pages, g_pages, 5),
                "page_reciprocal_rank": reciprocal_rank(r_pages, g_pages),
            }

            if exact_applies:
                row.update(
                    {
                        "recall_at_5": recall_at_k(retrieved_ids, gold_ids, 5),
                        "recall_at_10": recall_at_k(retrieved_ids, gold_ids, 10),
                        "hit_at_5": hit_at_k(retrieved_ids, gold_ids, 5),
                        "reciprocal_rank": reciprocal_rank(retrieved_ids, gold_ids),
                        "precision_at_5": precision_at_k(retrieved_ids, gold_ids, 5),
                    }
                )

            rows.append(row)

        latencies.sort()
        summary: dict[str, Any] = {
            "questions": len(rows),
            "median_latency_ms": latencies[len(latencies) // 2] if latencies else 0.0,
            "page_level": {
                "recall_at_5": mean([r["page_recall_at_5"] for r in rows]),
                "recall_at_10": mean([r["page_recall_at_10"] for r in rows]),
                "hit_at_5": mean([r["page_hit_at_5"] for r in rows]),
                "mrr": mean([r["page_reciprocal_rank"] for r in rows]),
            },
            "per_question": rows,
        }

        if exact_applies:
            summary["exact"] = {
                "recall_at_5": mean([r["recall_at_5"] for r in rows]),
                "recall_at_10": mean([r["recall_at_10"] for r in rows]),
                "hit_at_5": mean([r["hit_at_5"] for r in rows]),
                "mrr": mean([r["reciprocal_rank"] for r in rows]),
                "precision_at_5": mean([r["precision_at_5"] for r in rows]),
            }

        per_retriever[name] = summary

        exact_note = ""
        if exact_applies:
            e = summary["exact"]
            exact_note = f"  |  exact R@5 {e['recall_at_5']:.3f}  MRR {e['mrr']:.3f}"
        p = summary["page_level"]
        print(
            f"  {profile:<6} {name:<8} page R@5 {p['recall_at_5']:.3f}  "
            f"R@10 {p['recall_at_10']:.3f}  MRR {p['mrr']:.3f}  "
            f"{summary['median_latency_ms']:.0f}ms{exact_note}",
            flush=True,
        )

    return per_retriever


def main() -> int:
    questions = load_questions()
    answerable = [q for q in questions if q.answerable]
    print(
        f"evaluation set: {len(questions)} questions "
        f"({len(answerable)} answerable, {len(questions) - len(answerable)} unanswerable)"
    )

    # Gold ids must exist in the profile they were authored against. A stale id
    # would sink recall and look exactly like a retrieval failure, so this
    # refuses rather than reporting a number that is wrong for an invisible
    # reason.
    problems = validate_gold_chunks(answerable, AUTHORING_PROFILE)
    if problems:
        print(f"\nevaluation set is inconsistent with profile {AUTHORING_PROFILE!r}:")
        for problem in problems:
            print(f"  {problem}")
        return 1

    all_gold = [g for q in answerable for g in q.gold_chunk_ids]
    location = resolve_gold(all_gold, AUTHORING_PROFILE)
    print(
        f"gold: {len(set(all_gold))} chunks across "
        f"{len({v[0] for v in location.values()})} documents\n"
    )

    profiles = [p.name for p in CHUNK_PROFILES]
    if "--profile" in sys.argv:
        profiles = [sys.argv[sys.argv.index("--profile") + 1]]

    results: dict[str, Any] = {}
    for profile in profiles:
        results[profile] = evaluate_profile(profile, questions, location)
        print()

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    output = settings.results_dir / "retrieval.json"
    output.write_text(
        json.dumps(
            {
                "settings": settings.provenance(),
                "questions": len(answerable),
                "authoring_profile": AUTHORING_PROFILE,
                "gold_definitions": {
                    "exact": (
                        "A retrieved chunk counts only if it is literally the gold "
                        f"chunk. Only computed for {AUTHORING_PROFILE}, the profile "
                        "the questions were authored against."
                    ),
                    "page_level": (
                        "Gold is the set of (document, page) pairs the gold chunks "
                        "came from; a retrieved chunk counts if it comes from one of "
                        "those pages. Profile-independent, so this is the only "
                        "definition under which the chunk-size comparison is valid. "
                        "Looser than exact, so the two must not be compared."
                    ),
                },
                "profiles": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
