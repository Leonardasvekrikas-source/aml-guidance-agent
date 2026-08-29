"""LLM-as-judge for groundedness.

An unaudited judge is not a metric — it is a second model's opinion, presented
with a decimal point. So two things are true of this module by design:

  1. The judge sees only the answer and the passages that were actually
     retrieved. It never sees the question's expected answer, and it is never
     told that a human wrote the question, because both invite it to grade
     agreement rather than groundedness.
  2. Its decisions are recorded individually, so that a human can grade a sample
     of them and the disagreement rate can be reported. That audit is what
     turns this from an opinion into a measurement with a known error rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import anthropic

from ..config import settings
from ..llm import make_client

JUDGE_TOOL: dict[str, Any] = {
    "name": "record_judgement",
    "description": "Record the groundedness judgement for this answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "grounded": {
                "type": "boolean",
                "description": (
                    "True only if EVERY factual assertion in the answer is supported "
                    "by the supplied passages."
                ),
            },
            "unsupported_assertions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Each assertion not supported by the passages. Empty if grounded.",
            },
            "reason": {
                "type": "string",
                "description": "Two sentences at most, citing what the passages do and do not say.",
            },
        },
        "required": ["grounded", "unsupported_assertions", "reason"],
        "additionalProperties": False,
    },
}

JUDGE_SYSTEM = """You judge whether an answer is GROUNDED in a set of passages.

Grounded means every factual assertion in the answer is stated by, or directly
entailed by, the passages provided. Nothing else counts.

You are NOT judging:
- whether the answer is correct in the real world
- whether it is well written, complete, or useful
- whether it agrees with what you know about anti-money-laundering practice

An answer that is entirely true and entirely absent from the passages is NOT
grounded. An answer that is dull, partial, and fully supported IS grounded.

Ignore hedging language and framing. Judge the factual assertions.

If an assertion is only partially supported — the passages support a weaker
version, or support it for a different scope — treat it as unsupported and name
it.

Call record_judgement exactly once."""


@dataclass
class Judgement:
    grounded: bool
    unsupported: list[str]
    reason: str
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "grounded": self.grounded,
            "unsupported_assertions": self.unsupported,
            "reason": self.reason,
            "error": self.error,
        }


class GroundednessJudge:
    def __init__(self, client: anthropic.Anthropic | None = None, model: str | None = None):
        self.client = client
        self.model = model or settings.anthropic_model

    def _get_client(self) -> anthropic.Anthropic:
        if self.client is None:
            self.client = make_client()
        return self.client

    def judge(self, answer: str, passages: list[str]) -> Judgement:
        if not answer.strip():
            return Judgement(False, [], "Empty answer.")
        if not passages:
            return Judgement(
                False, [], "No passages were retrieved, so nothing can ground the answer."
            )

        body = "\n\n---\n\n".join(passages)
        prompt = (
            f"PASSAGES:\n\n{body}\n\n"
            f"ANSWER TO JUDGE:\n\n{answer}\n\n"
            "Is every factual assertion in the answer supported by the passages? "
            "Call record_judgement."
        )

        try:
            # The tool and message payloads are plain dicts matching the
            # documented wire format. The SDK types them as TypedDicts,
            # which a dict literal does not satisfy structurally.
            response = self._get_client().messages.create(  # type: ignore[call-overload]
                model=self.model,
                max_tokens=2000,
                system=JUDGE_SYSTEM,
                tools=[JUDGE_TOOL],
                tool_choice={"type": "tool", "name": "record_judgement"},
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APIError as exc:
            # Recorded as an error rather than counted as ungrounded: a failed
            # API call is not evidence about the answer, and silently scoring it
            # zero would understate groundedness for reasons unrelated to the
            # system being measured.
            return Judgement(False, [], "", error=f"{type(exc).__name__}: {exc}")

        for block in response.content:
            if block.type == "tool_use" and block.name == "record_judgement":
                data = cast(dict[str, Any], block.input)
                return Judgement(
                    grounded=bool(data.get("grounded")),
                    unsupported=[str(x) for x in (data.get("unsupported_assertions") or [])],
                    reason=str(data.get("reason", "")),
                )

        return Judgement(False, [], "", error="judge returned no verdict")
