"""Parsing the model's `answer` tool arguments.

Regression test for a live failure: the model returned a bare string where the
schema specifies an object, and the loop crashed part-way through an
evaluation run.
"""

from __future__ import annotations

from aml_agent.agent.loop import _parse_claims


class TestParseClaims:
    def test_well_formed_claims(self):
        claims = _parse_claims([{"text": "A claim.", "chunk_ids": [1, 2]}])
        assert len(claims) == 1
        assert claims[0].text == "A claim."
        assert claims[0].chunk_ids == [1, 2]

    def test_bare_string_becomes_an_uncitable_claim(self):
        # The failure that crashed a live run. It must survive parsing AND
        # carry no citations, so the provenance check rejects the draft rather
        # than the claim vanishing and the rest passing validation.
        claims = _parse_claims(["just a sentence"])
        assert len(claims) == 1
        assert claims[0].text == "just a sentence"
        assert claims[0].chunk_ids == []

    def test_missing_chunk_ids(self):
        claims = _parse_claims([{"text": "No citations."}])
        assert claims[0].chunk_ids == []

    def test_non_integer_chunk_ids_are_dropped_not_fatal(self):
        claims = _parse_claims([{"text": "x", "chunk_ids": [1, "nonsense", None, 3]}])
        assert claims[0].chunk_ids == [1, 3]

    def test_string_digits_are_accepted(self):
        # The SDK may hand back numbers as strings depending on how the model
        # serialised them.
        claims = _parse_claims([{"text": "x", "chunk_ids": ["7"]}])
        assert claims[0].chunk_ids == [7]

    def test_none_and_empty(self):
        assert _parse_claims(None) == []
        assert _parse_claims([]) == []
