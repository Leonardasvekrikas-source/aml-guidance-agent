"""Citation validation.

Two checks, deliberately separated, because they fail differently and one of
them is far more trustworthy than the other.

**Check 1 — provenance. Pure code, deterministic, cheap.**
Every cited chunk id must have been returned by a search in THIS run. This
catches the failure mode that matters most in a regulated domain: a fluent
answer citing a document that was never retrieved, or a chunk id the model
invented because the number looked plausible. No model opinion is involved —
the id is either in the set of retrieved ids or it is not.

**Check 2 — support. An LLM entailment call, and therefore fallible.**
The cited text must actually support the claim, not merely discuss the same
topic. This cannot be done in code without solving natural-language entailment,
so it is delegated to a model with a narrow, structured task: given this claim
and this passage, is the claim supported? The model is given the passage and
the claim only — never the question, and never the rest of the answer — so it
cannot be swayed by a persuasive surrounding argument.

**What this validator cannot catch.** Stated plainly, because a validator whose
limits are undocumented invites more trust than it has earned:

  1. A claim that is supported by its cited chunk but *misleading in context* —
     a true sentence quoted from a passage that qualifies it three sentences
     later, outside the chunk boundary.
  2. Errors of omission. Nothing here notices that the answer left out the
     exception that makes it wrong in practice.
  3. Aggregation across claims. Each claim is checked alone, so two individually
     supported claims that together imply something neither supports will pass.
  4. Judge error. Check 2 is a model, and a model that says "supported" when it
     is not shifts the failure from the drafter to the validator rather than
     removing it. The M4 judge-agreement audit exists because of this.
  5. Correctness of the source. If a guidance document is out of date, a claim
     faithfully citing it is validated and still wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import anthropic

from ..agent.loop import AgentResult, Claim
from ..config import settings
from ..cost import usd
from ..llm import make_client
from ..retrieval.base import Hit

SUPPORT_TOOL: dict[str, Any] = {
    "name": "record_support",
    "description": "Record whether the passage supports the claim.",
    "input_schema": {
        "type": "object",
        "properties": {
            "supported": {
                "type": "boolean",
                "description": (
                    "True only if the passage states or directly entails the claim. "
                    "False if the passage merely discusses the same topic, or "
                    "supports something similar but not this."
                ),
            },
            "reason": {
                "type": "string",
                "description": (
                    "One sentence. If supported, quote the phrase that does it. "
                    "If not, say what the passage actually says instead."
                ),
            },
        },
        "required": ["supported", "reason"],
        "additionalProperties": False,
    },
}

SUPPORT_SYSTEM = """You check whether a passage supports a claim. You are the \
last line of defence before an answer about financial-crime regulation reaches \
a reader, so err towards "not supported".

Rules:

- Judge ONLY whether this passage supports this claim. You do not know the
  question that was asked, and you must not reward a claim for being plausible,
  well-written, or true in general.
- A passage that discusses the same topic without stating the claim does NOT
  support it.
- A passage supporting a weaker or stronger version of the claim does NOT
  support it. "May indicate" is not "indicates".
- If the claim adds specifics the passage does not contain, it is not supported.
- If you are unsure, it is not supported.

Call record_support exactly once."""


@dataclass
class ClaimVerdict:
    claim: str
    chunk_ids: list[int]
    citation_ok: bool
    support_ok: bool
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.citation_ok and self.support_ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "chunk_ids": self.chunk_ids,
            "citation_ok": self.citation_ok,
            "support_ok": self.support_ok,
            "detail": self.detail,
        }


@dataclass
class ValidationReport:
    verdicts: list[ClaimVerdict] = field(default_factory=list)
    checked_support: bool = True
    # Validation calls a model once per claim, so its cost is real and is
    # reported separately from the agent loop rather than folded in.
    usd: float = 0.0

    @property
    def passed(self) -> bool:
        return bool(self.verdicts) and all(v.ok for v in self.verdicts)

    @property
    def failures(self) -> list[ClaimVerdict]:
        return [v for v in self.verdicts if not v.ok]

    def citation_validity(self) -> float:
        """Fraction of claims whose citations were all genuinely retrieved."""
        if not self.verdicts:
            return 0.0
        return sum(1 for v in self.verdicts if v.citation_ok) / len(self.verdicts)

    def groundedness(self) -> float:
        """Fraction of claims supported by their cited text."""
        if not self.verdicts:
            return 0.0
        return sum(1 for v in self.verdicts if v.ok) / len(self.verdicts)

    def feedback(self) -> str:
        """The message fed back to the agent on a rejected draft.

        Specific rather than scolding: the agent is told which claim failed and
        why, so the retry can fix that claim instead of rewriting blindly.
        """
        lines = [
            "Your draft was REJECTED by citation validation. The following claims did not survive:",
            "",
        ]
        for verdict in self.failures:
            lines.append(f"- Claim: {verdict.claim}")
            lines.append(f"  Cited: {verdict.chunk_ids}")
            lines.append(f"  Problem: {verdict.detail}")
            lines.append("")
        lines.append(
            "Search for evidence that actually supports these claims and answer "
            "again, or drop the claims you cannot support. If the corpus does "
            "not support them at all, call refuse instead of weakening the "
            "wording until it passes."
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "citation_validity": round(self.citation_validity(), 4),
            "groundedness": round(self.groundedness(), 4),
            "checked_support": self.checked_support,
            "usd": round(self.usd, 5),
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


class Validator:
    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str | None = None,
        check_support: bool = True,
    ):
        self.client = client
        self.model = model or settings.grader_model
        self.check_support = check_support
        self._spend = 0.0

    def _client(self) -> anthropic.Anthropic:
        if self.client is None:
            self.client = make_client()
        return self.client

    def validate(self, result: AgentResult) -> ValidationReport:
        report = ValidationReport(checked_support=self.check_support)
        self._spend = 0.0

        for claim in result.claims:
            verdict = self._validate_claim(claim, result)
            report.verdicts.append(verdict)

        report.usd = self._spend
        return report

    def _validate_claim(self, claim: Claim, result: AgentResult) -> ClaimVerdict:
        # --- check 1: provenance, in code -----------------------------------
        if not claim.chunk_ids:
            return ClaimVerdict(
                claim=claim.text,
                chunk_ids=[],
                citation_ok=False,
                support_ok=False,
                detail="The claim cites no chunks at all.",
            )

        uncited = [c for c in claim.chunk_ids if c not in result.retrieved_chunk_ids]
        if uncited:
            return ClaimVerdict(
                claim=claim.text,
                chunk_ids=claim.chunk_ids,
                citation_ok=False,
                support_ok=False,
                detail=(
                    f"Chunk id(s) {uncited} were never returned by a search in this "
                    "run. A citation to a passage the system did not retrieve cannot "
                    "be verified and is treated as fabricated."
                ),
            )

        if not self.check_support:
            return ClaimVerdict(
                claim=claim.text,
                chunk_ids=claim.chunk_ids,
                citation_ok=True,
                support_ok=True,
                detail="Provenance verified; support check disabled.",
            )

        # --- check 2: support, by entailment --------------------------------
        hits = [result.retrieved_hits[c] for c in claim.chunk_ids if c in result.retrieved_hits]
        supported, reason = self._check_support(claim.text, hits)

        return ClaimVerdict(
            claim=claim.text,
            chunk_ids=claim.chunk_ids,
            citation_ok=True,
            support_ok=supported,
            detail=reason,
        )

    def _check_support(self, claim_text: str, hits: list[Hit]) -> tuple[bool, str]:
        if not hits:
            return False, "No retrieved passage was available for the cited ids."

        passages = "\n\n---\n\n".join(
            f"[chunk {h.chunk_id}] {h.title}, p.{h.page}\n{h.text}" for h in hits
        )
        prompt = (
            f"PASSAGE(S):\n\n{passages}\n\n"
            f"CLAIM:\n\n{claim_text}\n\n"
            "Does the passage support the claim? Call record_support."
        )

        try:
            # The tool and message payloads are plain dicts matching the
            # documented wire format. The SDK types them as TypedDicts,
            # which a dict literal does not satisfy structurally.
            response = self._client().messages.create(  # type: ignore[call-overload]
                model=self.model,
                max_tokens=1000,
                system=SUPPORT_SYSTEM,
                tools=[SUPPORT_TOOL],
                tool_choice={"type": "tool", "name": "record_support"},
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APIError as exc:
            # A validator that fails open would silently disable the check that
            # justifies this project. Fail closed and say so.
            return False, f"support check could not run ({type(exc).__name__}); failing closed"

        self._spend += usd(
            self.model,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == "record_support":
                data = cast(dict[str, Any], block.input)
                return bool(data.get("supported")), str(data.get("reason", ""))

        return False, "support check returned no verdict; failing closed"
