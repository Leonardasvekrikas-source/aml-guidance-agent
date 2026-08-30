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

import json
from dataclasses import dataclass
from typing import Any, cast

import anthropic

from ..config import settings
from ..cost import usd
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


def _recover_overflow(value: Any) -> tuple[list[str], str]:
    """Coerce `unsupported_assertions` into a list, recovering a swallowed reason.

    The schema says array of strings. Two things the model actually did:

    1. Returned the array as a JSON-encoded STRING. Iterating that yields one
       entry per character — a single judged answer became 2,173 "unsupported
       assertions", each one letter long, in a report meant for a human.
    2. Serialised the *whole remainder of the tool-call object* into this one
       field: the array, then `, "reason": "..."`. The reason field then came
       back empty, so the judge's explanation vanished from the audit file
       exactly where a grader most needs it.

    Both are recoverable without another API call, because the information is
    all there — only mis-split. Returns (assertions, recovered_reason), where
    the reason is empty unless it had been swallowed.
    """
    if isinstance(value, list):
        return [str(x) for x in value], ""
    if not isinstance(value, str):
        return [], ""

    text = value.strip()
    if not text:
        return [], ""
    if not text.startswith("["):
        return [text], ""

    try:
        head, consumed = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return [text], ""

    items = [str(x) for x in head] if isinstance(head, list) else [str(head)]

    # Anything after the array is the rest of the object that leaked in with
    # it. Wrapping it in braces makes it parseable again.
    remainder = text[consumed:].strip().lstrip(",").strip()
    if not remainder:
        return items, ""
    try:
        recovered = json.loads("{" + remainder + "}")
    except json.JSONDecodeError:
        return items, ""
    return items, str(recovered.get("reason", "")) if isinstance(recovered, dict) else ""


@dataclass
class Judgement:
    grounded: bool
    unsupported: list[str]
    reason: str
    error: str = ""
    # The judge is a model call and therefore costs money. Leaving it out
    # of the accounting is how a run's reported cost drifts below what was
    # actually charged.
    usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "grounded": self.grounded,
            "unsupported_assertions": self.unsupported,
            "reason": self.reason,
            "error": self.error,
            "usd": round(self.usd, 5),
        }


class GroundednessJudge:
    def __init__(self, client: anthropic.Anthropic | None = None, model: str | None = None):
        self.client = client
        self.model = model or settings.grader_model

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

        spend = usd(
            self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == "record_judgement":
                data = cast(dict[str, Any], block.input)
                unsupported, recovered_reason = _recover_overflow(
                    data.get("unsupported_assertions")
                )
                return Judgement(
                    grounded=bool(data.get("grounded")),
                    unsupported=unsupported,
                    reason=str(data.get("reason", "")) or recovered_reason,
                    usd=spend,
                )

        return Judgement(False, [], "", error="judge returned no verdict")
