"""Validation layer.

The provenance check is pure code with no model in it, so it can be tested
exhaustively — and it is the check that catches the failure this project exists
to prevent: a confident answer citing a passage that was never retrieved.

The support check is stubbed here. Its behaviour under a real model is measured
by the M4 judge-agreement audit, not asserted in a unit test.
"""

from __future__ import annotations

import pytest

from aml_agent.agent.loop import AgentResult, Claim
from aml_agent.retrieval.base import Hit
from aml_agent.validation.validator import Validator


def make_hit(chunk_id: int, text: str = "Some retrieved passage text.") -> Hit:
    return Hit(
        chunk_id=chunk_id,
        score=1.0,
        rank=1,
        text=text,
        document_id="doc-a",
        title="A Guidance Document",
        publisher="FATF",
        source_url="https://example.org/doc.pdf",
        page=12,
        section_heading="3.1 Indicators",
        retriever="hybrid",
    )


def make_result(claims: list[Claim], retrieved: list[int]) -> AgentResult:
    result = AgentResult(question="q", trace_id="t", outcome="answered", claims=claims)
    result.retrieved_chunk_ids = set(retrieved)
    result.retrieved_hits = {c: make_hit(c) for c in retrieved}
    return result


class TestProvenanceCheck:
    """Check 1: was the cited chunk actually retrieved in this run?"""

    def test_fabricated_chunk_id_is_rejected(self):
        # The core failure mode. Chunk 999 was never retrieved, so the claim
        # citing it must not pass, regardless of how reasonable it sounds.
        result = make_result([Claim("A plausible claim.", [999])], retrieved=[1, 2, 3])
        report = Validator(check_support=False).validate(result)

        assert not report.passed
        assert report.verdicts[0].citation_ok is False
        assert "999" in report.verdicts[0].detail

    def test_claim_with_no_citation_is_rejected(self):
        result = make_result([Claim("Unsupported assertion.", [])], retrieved=[1])
        report = Validator(check_support=False).validate(result)

        assert not report.passed
        assert report.verdicts[0].citation_ok is False

    def test_retrieved_chunk_passes_provenance(self):
        result = make_result([Claim("A cited claim.", [2])], retrieved=[1, 2, 3])
        report = Validator(check_support=False).validate(result)

        assert report.passed
        assert report.verdicts[0].citation_ok is True

    def test_one_bad_citation_among_several_rejects_the_claim(self):
        # Partial credit here would let a fabricated citation ride along beside
        # a real one, which is exactly the smuggling route worth closing.
        result = make_result([Claim("Mixed claim.", [1, 999])], retrieved=[1, 2])
        report = Validator(check_support=False).validate(result)

        assert not report.passed
        assert "999" in report.verdicts[0].detail

    def test_one_bad_claim_rejects_the_whole_draft(self):
        result = make_result(
            [Claim("Good.", [1]), Claim("Bad.", [999])],
            retrieved=[1, 2],
        )
        report = Validator(check_support=False).validate(result)

        assert not report.passed
        assert len(report.failures) == 1
        assert report.citation_validity() == pytest.approx(0.5)


class TestReportMetrics:
    def test_empty_draft_does_not_pass(self):
        # A draft with zero claims must not count as validated. Vacuous truth
        # would make "all claims supported" trivially satisfiable by saying
        # nothing.
        report = Validator(check_support=False).validate(make_result([], retrieved=[1]))
        assert not report.passed

    def test_feedback_names_the_failing_claim(self):
        result = make_result([Claim("The failing claim.", [999])], retrieved=[1])
        report = Validator(check_support=False).validate(result)

        feedback = report.feedback()
        assert "The failing claim." in feedback
        assert "REJECTED" in feedback
        # The retry must not be told it can simply soften the wording.
        assert "refuse" in feedback.lower()


class TestSupportCheckStub:
    """Check 2 with the model stubbed out, to test the wiring not the model."""

    def test_unsupported_claim_fails_even_with_valid_provenance(self, monkeypatch):
        validator = Validator(check_support=True)
        monkeypatch.setattr(
            validator, "_check_support", lambda claim, hits: (False, "passage says otherwise")
        )

        result = make_result([Claim("Cited but unsupported.", [1])], retrieved=[1])
        report = validator.validate(result)

        assert not report.passed
        assert report.verdicts[0].citation_ok is True
        assert report.verdicts[0].support_ok is False
        # Citation validity and groundedness must diverge here: the citation is
        # real, the support is not.
        assert report.citation_validity() == 1.0
        assert report.groundedness() == 0.0

    def test_supported_claim_passes(self, monkeypatch):
        validator = Validator(check_support=True)
        monkeypatch.setattr(
            validator, "_check_support", lambda claim, hits: (True, "stated on p.12")
        )

        report = validator.validate(make_result([Claim("Supported.", [1])], retrieved=[1]))
        assert report.passed
        assert report.groundedness() == 1.0
