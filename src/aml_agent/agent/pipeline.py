"""Question in, validated answer or refusal out.

The full path:

    question -> agent loop -> draft -> validation
                                 |          |
                                 |          +-- passed -> answer
                                 |          +-- failed -> ONE retry with the
                                 |                        failure fed back
                                 +-- refused/exhausted -> refusal

A second rejection returns a refusal. Not a third attempt, not the best of the
three drafts. If two attempts with explicit feedback cannot produce claims the
corpus supports, the honest output is that the corpus does not support an
answer — and retrying until something passes is how a validator gets optimised
into a rubber stamp.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..config import settings
from ..retrieval.base import Retriever
from ..validation.validator import ValidationReport, Validator
from .loop import AgentLoop, AgentResult

MAX_DRAFTS = 2


@dataclass
class PipelineResult:
    question: str
    trace_id: str
    outcome: str  # answered | refused | error
    summary: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    refusal_reason: str = ""
    attempts: int = 0
    validation: ValidationReport | None = None
    agent_results: list[AgentResult] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def total_searches(self) -> int:
        return sum(r.searches for r in self.agent_results)

    @property
    def total_input_tokens(self) -> int:
        return sum(r.input_tokens for r in self.agent_results)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.agent_results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "question": self.question,
            "outcome": self.outcome,
            "summary": self.summary,
            "citations": self.citations,
            "refusal_reason": self.refusal_reason,
            "attempts": self.attempts,
            "searches": self.total_searches,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "validation": self.validation.to_dict() if self.validation else None,
            "drafts": [r.to_dict() for r in self.agent_results],
        }


class Pipeline:
    def __init__(
        self,
        retriever: Retriever,
        validator: Validator | None = None,
        loop: AgentLoop | None = None,
        max_drafts: int = MAX_DRAFTS,
    ):
        self.loop = loop or AgentLoop(retriever)
        self.validator = validator or Validator()
        self.max_drafts = max_drafts

    def ask(self, question: str) -> PipelineResult:
        started = time.perf_counter()
        outcome = PipelineResult(question=question, trace_id="", outcome="error")

        feedback: str | None = None

        for attempt in range(1, self.max_drafts + 1):
            outcome.attempts = attempt

            prompt = question if feedback is None else f"{question}\n\n{feedback}"
            draft = self.loop.run(prompt)
            outcome.agent_results.append(draft)
            if not outcome.trace_id:
                outcome.trace_id = draft.trace_id

            # The agent declined or ran out of iterations. Both are terminal —
            # validation has nothing to check, and pushing it to try again would
            # be arguing with a correct refusal.
            if draft.outcome in {"refused", "exhausted"}:
                outcome.outcome = "refused"
                outcome.refusal_reason = draft.refusal_reason
                break

            if draft.outcome == "error":
                outcome.outcome = "error"
                outcome.refusal_reason = draft.error
                break

            report = self.validator.validate(draft)
            outcome.validation = report

            if report.passed:
                outcome.outcome = "answered"
                outcome.summary = draft.summary
                outcome.citations = self._citations(draft)
                break

            if attempt < self.max_drafts:
                # One retry, with the specific failures fed back.
                feedback = report.feedback()
                continue

            # Second rejection. Refuse rather than return claims that failed.
            outcome.outcome = "refused"
            outcome.refusal_reason = (
                "The draft answer could not be supported by the retrieved passages "
                "after two attempts, so it was withheld. "
                + "; ".join(v.detail for v in report.failures[:3])
            )

        outcome.latency_ms = (time.perf_counter() - started) * 1000.0
        return outcome

    @staticmethod
    def _citations(draft: AgentResult) -> list[dict[str, Any]]:
        cited: dict[int, dict[str, Any]] = {}
        for claim in draft.claims:
            for chunk_id in claim.chunk_ids:
                hit = draft.retrieved_hits.get(chunk_id)
                if hit and chunk_id not in cited:
                    cited[chunk_id] = {
                        "chunk_id": chunk_id,
                        "title": hit.title,
                        "publisher": hit.publisher,
                        "page": hit.page,
                        "source_url": hit.source_url,
                        "section_heading": hit.section_heading,
                    }
        return list(cited.values())


def write_trace(result: PipelineResult) -> None:
    """Persist the full trace, including rejected drafts.

    Rejected drafts are the interesting part: a trace that only records the
    answer that passed cannot show that validation ever did anything.
    """
    settings.traces_dir.mkdir(parents=True, exist_ok=True)
    path = settings.traces_dir / f"{result.trace_id}.json"
    payload = result.to_dict()
    payload["written_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
