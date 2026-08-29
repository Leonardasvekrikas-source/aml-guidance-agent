"""M4: answer-quality evaluation.

Runs the full pipeline over the evaluation set and reports:

  * groundedness      — LLM judge, over the answerable questions that produced
                        an answer
  * citation validity — pure code: every cited chunk was actually retrieved
  * refusal accuracy  — over the 10 unanswerable questions. This is the cheapest
                        hallucination detector in the project: a system that
                        answers a question the corpus cannot support has
                        hallucinated, and no judge is needed to see it.
  * answer rate       — how often an answerable question produced an answer.
                        Reported next to groundedness on purpose: a system that
                        refuses everything scores perfect groundedness and is
                        useless, and the two numbers only mean something
                        together.

Every judge decision is written out so that `make judge-audit` can sample them
for human grading.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from typing import Any

from ..config import DEFAULT_PROFILE, settings
from ..llm import have_credentials
from ..retrieval import build_retriever
from .judge import GroundednessJudge
from .metrics import mean
from .questions import load_questions, resolve_gold_ids


def evaluate(profile: str = DEFAULT_PROFILE, limit: int | None = None) -> dict[str, Any]:
    from ..agent.pipeline import Pipeline, write_trace

    questions = load_questions()
    resolve_gold_ids([q for q in questions if q.answerable], profile)
    if limit:
        answerable = [q for q in questions if q.answerable][:limit]
        unanswerable = [q for q in questions if not q.answerable][: max(1, limit // 3)]
        questions = answerable + unanswerable

    retriever = build_retriever("hybrid", profile)
    pipeline = Pipeline(retriever)
    judge = GroundednessJudge()

    rows: list[dict[str, Any]] = []

    for position, question in enumerate(questions, start=1):
        result = pipeline.ask(question.question)
        write_trace(result)

        row: dict[str, Any] = {
            "question_id": question.id,
            "question": question.question,
            "answerable": question.answerable,
            "outcome": result.outcome,
            "attempts": result.attempts,
            "searches": result.total_searches,
            "latency_ms": round(result.latency_ms, 1),
            "input_tokens": result.total_input_tokens,
            "output_tokens": result.total_output_tokens,
            "trace_id": result.trace_id,
            "citation_validity": (
                result.validation.citation_validity() if result.validation else None
            ),
        }

        # Groundedness is only meaningful for answers that were actually given.
        if question.answerable and result.outcome == "answered":
            draft = result.agent_results[-1]
            passages = [
                f"[chunk {cid}] {draft.retrieved_hits[cid].text}"
                for cid in sorted(draft.retrieved_chunk_ids)
                if cid in draft.retrieved_hits
            ]
            judgement = judge.judge(result.summary, passages)
            row["judge"] = judgement.to_dict()

        rows.append(row)
        marker = {"answered": "+", "refused": "-", "error": "!"}.get(result.outcome, "?")
        print(
            f"  {marker} [{position}/{len(questions)}] {question.id:<12} "
            f"{result.outcome:<9} {result.total_searches} searches  "
            f"{result.latency_ms / 1000:.1f}s",
            flush=True,
        )

    return summarise(rows)


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]

    answered = [r for r in answerable if r["outcome"] == "answered"]
    judged = [r for r in answered if r.get("judge") and not r["judge"].get("error")]

    # Refusal accuracy: on an unanswerable question, refusing is correct.
    correct_refusals = [r for r in unanswerable if r["outcome"] == "refused"]

    latencies = sorted(r["latency_ms"] for r in rows if r["latency_ms"])
    searches = [r["searches"] for r in rows]

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "settings": settings.provenance(),
        "counts": {
            "questions": len(rows),
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
            "answered": len(answered),
            "judged": len(judged),
        },
        "metrics": {
            "groundedness": (
                mean([1.0 if r["judge"]["grounded"] else 0.0 for r in judged]) if judged else None
            ),
            "citation_validity": (
                mean(
                    [r["citation_validity"] for r in answered if r["citation_validity"] is not None]
                )
                if answered
                else None
            ),
            "refusal_accuracy": (
                len(correct_refusals) / len(unanswerable) if unanswerable else None
            ),
            "answer_rate": (len(answered) / len(answerable) if answerable else None),
            "median_latency_ms": latencies[len(latencies) // 2] if latencies else None,
            "median_searches": sorted(searches)[len(searches) // 2] if searches else None,
        },
        "per_question": rows,
    }


def main() -> int:
    if not have_credentials():
        print(
            "M4 needs an Anthropic API key: it runs the agent loop and the judge.\n"
            "  cp .env.example .env  and set ANTHROPIC_API_KEY\n\n"
            "The M1 retrieval benchmark runs without one: `make eval-retrieval`."
        )
        return 2

    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    print("running the full pipeline over the evaluation set\n")
    report = evaluate(limit=limit)

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    output = settings.results_dir / "answers.json"
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    metrics = report["metrics"]
    print("\n" + "=" * 60)
    for label, key in (
        ("groundedness", "groundedness"),
        ("citation validity", "citation_validity"),
        ("refusal accuracy", "refusal_accuracy"),
        ("answer rate", "answer_rate"),
    ):
        value = metrics[key]
        print(f"  {label:<22}{'n/a' if value is None else f'{value:.3f}'}")
    print(f"  {'median latency':<22}{(metrics['median_latency_ms'] or 0) / 1000:.1f}s")
    print(f"  {'median searches':<22}{metrics['median_searches']}")
    print("=" * 60)
    print(f"\nwrote {output}")
    print("Now audit the judge: `make judge-audit` samples 20 decisions for human grading.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
