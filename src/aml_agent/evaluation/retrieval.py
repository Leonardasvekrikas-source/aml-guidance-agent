"""M1: the retrieval benchmark.

Runs BM25, dense and hybrid RRF over the answerable questions, for every chunk
profile, and writes results/retrieval.json. Per-question results are kept, not
just the averages, because the averages are what you report and the
per-question rows are what you learn from.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from ..config import CHUNK_PROFILES, settings
from ..retrieval import build_retrievers
from .metrics import hit_at_k, mean, precision_at_k, recall_at_k, reciprocal_rank
from .questions import load_questions, validate_gold_chunks

K_VALUES = (1, 5, 10)
MAX_K = max(K_VALUES)


def evaluate_profile(profile: str, questions: list) -> dict[str, Any]:
    answerable = [q for q in questions if q.answerable]
    if not answerable:
        raise RuntimeError("no answerable questions in the evaluation set")

    problems = validate_gold_chunks(answerable, profile)
    if problems:
        # Refuse rather than report a number that is wrong for a reason the
        # reader cannot see.
        raise RuntimeError(
            f"evaluation set is inconsistent with profile {profile!r}:\n  "
            + "\n  ".join(problems)
        )

    retrievers = build_retrievers(profile)
    per_retriever: dict[str, Any] = {}

    for name, retriever in retrievers.items():
        rows: list[dict[str, Any]] = []
        latencies: list[float] = []

        for question in answerable:
            started = time.perf_counter()
            hits = retriever.search(question.question, MAX_K)
            latencies.append((time.perf_counter() - started) * 1000.0)

            retrieved = [h.chunk_id for h in hits]
            gold = list(question.gold_chunk_ids)

            rows.append(
                {
                    "question_id": question.id,
                    "topic": question.topic,
                    "gold_chunk_ids": gold,
                    "retrieved_chunk_ids": retrieved,
                    "recall_at_5": recall_at_k(retrieved, gold, 5),
                    "recall_at_10": recall_at_k(retrieved, gold, 10),
                    "hit_at_5": hit_at_k(retrieved, gold, 5),
                    "reciprocal_rank": reciprocal_rank(retrieved, gold),
                }
            )

        latencies.sort()
        per_retriever[name] = {
            "recall_at_5": mean([r["recall_at_5"] for r in rows]),
            "recall_at_10": mean([r["recall_at_10"] for r in rows]),
            "hit_at_5": mean([r["hit_at_5"] for r in rows]),
            "mrr": mean([r["reciprocal_rank"] for r in rows]),
            "precision_at_5": mean(
                [
                    precision_at_k(r["retrieved_chunk_ids"], r["gold_chunk_ids"], 5)
                    for r in rows
                ]
            ),
            "median_latency_ms": latencies[len(latencies) // 2] if latencies else 0.0,
            "questions": len(rows),
            "per_question": rows,
        }
        print(
            f"  {profile:<6} {name:<8} "
            f"R@5 {per_retriever[name]['recall_at_5']:.3f}  "
            f"R@10 {per_retriever[name]['recall_at_10']:.3f}  "
            f"MRR {per_retriever[name]['mrr']:.3f}  "
            f"{per_retriever[name]['median_latency_ms']:.0f}ms",
            flush=True,
        )

    return per_retriever


def main() -> int:
    questions = load_questions()
    answerable = [q for q in questions if q.answerable]
    print(
        f"evaluation set: {len(questions)} questions "
        f"({len(answerable)} answerable, {len(questions) - len(answerable)} unanswerable)\n"
    )

    profiles = [p.name for p in CHUNK_PROFILES]
    if "--profile" in sys.argv:
        profiles = [sys.argv[sys.argv.index("--profile") + 1]]

    results: dict[str, Any] = {}
    for profile in profiles:
        print(f"profile {profile}:")
        results[profile] = evaluate_profile(profile, questions)
        print()

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    output = settings.results_dir / "retrieval.json"
    output.write_text(
        json.dumps(
            {
                "settings": settings.provenance(),
                "questions": len(answerable),
                "k_values": list(K_VALUES),
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
