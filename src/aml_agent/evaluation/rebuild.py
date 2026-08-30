"""Rebuild results/answers.json from the traces on disk.

Traces are the source of truth for a run: every question writes one, and it
records the outcome, the validation verdicts, the citations and the cost. The
aggregate file is derived from them, so it can be derived again.

This exists because it was needed. A targeted re-run was launched without
`--merge` and overwrote a completed forty-question run with two rows. The
traces survived, so almost everything was recoverable — except the judge
verdicts, which were only ever written to the aggregate. Those are recomputed
here, which costs one cheap model call per answered question rather than
re-running the whole pipeline.

The lesson is in the design, not just the recovery: anything expensive to
produce should be reconstructible from a durable artifact, and a trace is that
artifact. The gap was that judge decisions never reached one.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from ..config import DEFAULT_PROFILE, settings
from ..db import connect
from .answers import summarise
from .judge import GroundednessJudge
from .questions import load_questions


def _passages_for(chunk_ids: list[int], profile: str) -> list[str]:
    """Re-read the passages a run retrieved, by id.

    The trace records which chunk ids were returned but not their text, since
    duplicating thousands of characters per question would make traces
    unreadable. The corpus is immutable between runs, so the text is fetched
    back from the database instead.
    """
    if not chunk_ids:
        return []
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, text FROM chunks WHERE id = ANY(%s) AND chunk_profile = %s",
            (chunk_ids, profile),
        ).fetchall()
    by_id = {r["id"]: r["text"] for r in rows}
    return [f"[chunk {cid}] {by_id[cid]}" for cid in chunk_ids if cid in by_id]


def rebuild(profile: str = DEFAULT_PROFILE, rejudge: bool = True) -> dict[str, Any]:
    questions = load_questions()
    by_text = {q.question.strip(): q for q in questions}

    judge = GroundednessJudge()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    traces = sorted(
        settings.traces_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
    )

    for path in traces:
        trace = json.loads(path.read_text(encoding="utf-8"))
        question = by_text.get(str(trace.get("question", "")).strip())
        if question is None:
            # A trace from an ad-hoc `make ask`, not part of the evaluation set.
            continue

        # Later traces win: a question re-run after a fix should be represented
        # by its most recent attempt, not its first.
        if question.id in seen:
            rows = [r for r in rows if r["question_id"] != question.id]
        seen.add(question.id)

        validation = trace.get("validation") or {}
        row: dict[str, Any] = {
            "question_id": question.id,
            "question": question.question,
            "answerable": question.answerable,
            "outcome": trace.get("outcome", "error"),
            "attempts": trace.get("attempts", 1),
            "searches": trace.get("searches", 0),
            "latency_ms": trace.get("latency_ms", 0.0),
            "input_tokens": trace.get("input_tokens", 0),
            "output_tokens": trace.get("output_tokens", 0),
            "trace_id": trace.get("trace_id", path.stem),
            "agent_usd": trace.get("agent_usd", 0.0),
            "validation_usd": trace.get("validation_usd", 0.0),
            "total_usd": trace.get("total_usd", 0.0),
            "citation_validity": validation.get("citation_validity"),
        }

        if rejudge and question.answerable and row["outcome"] == "answered":
            chunk_ids: list[int] = []
            for draft in trace.get("drafts", []):
                chunk_ids.extend(int(c) for c in draft.get("retrieved_chunk_ids", []))
            passages = _passages_for(sorted(set(chunk_ids)), profile)
            judgement = judge.judge(str(trace.get("summary", "")), passages)
            row["judge"] = judgement.to_dict()
            marker = "grounded" if judgement.grounded else "NOT grounded"
            print(f"  judged {question.id}: {marker}", flush=True)

        rows.append(row)

    rows.sort(key=lambda r: r["question_id"])
    return summarise(rows)


def main() -> int:
    rejudge = "--no-judge" not in sys.argv
    print(f"rebuilding from traces in {settings.traces_dir}")
    report = rebuild(rejudge=rejudge)

    output = settings.results_dir / "answers.json"
    output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    counts = report["counts"]
    metrics = report["metrics"]
    print(
        f"\nrebuilt {counts['questions']} questions "
        f"({counts['answerable']} answerable, {counts['unanswerable']} unanswerable)"
    )
    for label, key in (
        ("groundedness", "groundedness"),
        ("citation validity", "citation_validity"),
        ("refusal accuracy", "refusal_accuracy"),
        ("answer rate", "answer_rate"),
    ):
        value = metrics[key]
        print(f"  {label:<22}{'n/a' if value is None else f'{value:.3f}'}")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
