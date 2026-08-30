"""How many first-stage candidates should the reranker see?

The reranker can only reorder what the first stage hands it. More candidates
means a higher ceiling and a linearly higher cost, and the interesting question
is where that curve flattens — past the point where extra candidates stop
containing new gold passages, the extra latency buys nothing.

Reports, for each candidate count:

  ceiling   recall of the first stage over the whole candidate pool — the best
            the reranker could possibly do
  achieved  recall@5 after reranking — what it actually did
  latency   median wall-clock per query

Written to results/candidate_sweep.json and summarised on stdout.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from ..config import DEFAULT_PROFILE, settings
from ..retrieval import build_retrievers
from ..retrieval.rerank import RerankedRetriever
from .metrics import mean, recall_at_k
from .questions import load_questions, resolve_gold_ids

CANDIDATE_COUNTS = (10, 25, 50, 100, 200)
TOP_K = 5


def main() -> int:
    profile = DEFAULT_PROFILE
    if "--profile" in sys.argv:
        profile = sys.argv[sys.argv.index("--profile") + 1]

    questions = load_questions()
    answerable = [q for q in questions if q.answerable]

    problems = resolve_gold_ids(answerable, profile)
    if problems:
        for problem in problems:
            print(f"  {problem}")
        return 1

    retrievers = build_retrievers(profile, with_reranker=False)
    first_stage = retrievers["hybrid"]

    rows: list[dict[str, Any]] = []
    print(
        f"{'candidates':>11}{'ceiling':>10}{'achieved':>10}{'headroom used':>15}{'median ms':>11}"
    )

    for count in CANDIDATE_COUNTS:
        reranked = RerankedRetriever(first_stage, candidate_k=count)

        ceilings: list[float] = []
        achieved: list[float] = []
        latencies: list[float] = []

        for question in answerable:
            gold = list(question.gold_chunk_ids)

            pool = [h.chunk_id for h in first_stage.search(question.question, count)]
            ceilings.append(recall_at_k(pool, gold, count))

            started = time.perf_counter()
            hits = reranked.search(question.question, TOP_K)
            latencies.append((time.perf_counter() - started) * 1000.0)
            achieved.append(recall_at_k([h.chunk_id for h in hits], gold, TOP_K))

        latencies.sort()
        ceiling = mean(ceilings)
        got = mean(achieved)
        # What fraction of what was available did the reranker actually deliver.
        # 1.0 means the reranker is not the bottleneck; every remaining miss is
        # a first-stage recall failure.
        used = (got / ceiling) if ceiling else 0.0

        row = {
            "candidates": count,
            "ceiling_recall": ceiling,
            "achieved_recall_at_5": got,
            "headroom_used": used,
            "median_latency_ms": latencies[len(latencies) // 2],
        }
        rows.append(row)
        print(
            f"{count:>11}{ceiling:>10.3f}{got:>10.3f}{used:>15.1%}"
            f"{row['median_latency_ms']:>11.0f}",
            flush=True,
        )

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if profile == DEFAULT_PROFILE else f"_{profile}"
    output = settings.results_dir / f"candidate_sweep{suffix}.json"
    output.write_text(
        json.dumps(
            {
                "settings": settings.provenance(),
                "profile": profile,
                "top_k": TOP_K,
                "questions": len(answerable),
                "sweep": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
