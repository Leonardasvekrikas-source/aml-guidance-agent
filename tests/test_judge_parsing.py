"""Parsing the judge's tool arguments.

Regression tests for a live failure. The judge's `unsupported_assertions`
field is specified as an array of strings; the model returned it as a
JSON-encoded string carrying the rest of the tool-call object as well. Iterating
that string produced 2,173 single-character "assertions" in a file written for
a human to read, and silently lost the judge's reasoning.
"""

from __future__ import annotations

from aml_agent.evaluation.judge import _recover_overflow


class TestRecoverOverflow:
    def test_plain_list_passes_through(self):
        items, reason = _recover_overflow(["one", "two"])
        assert items == ["one", "two"]
        assert reason == ""

    def test_json_encoded_array_is_unpacked_not_iterated(self):
        # The core bug: without parsing, this becomes one entry per character.
        items, reason = _recover_overflow('["alpha", "beta"]')
        assert items == ["alpha", "beta"]
        assert reason == ""

    def test_swallowed_reason_is_recovered(self):
        # The model serialised the whole remainder of the object into this one
        # field, which also emptied `reason`.
        raw = '["only claim"], "reason": "the passage says something else"'
        items, reason = _recover_overflow(raw)
        assert items == ["only claim"]
        assert reason == "the passage says something else"

    def test_plain_sentence_becomes_one_assertion(self):
        items, reason = _recover_overflow("the answer overstates the obligation")
        assert items == ["the answer overstates the obligation"]
        assert reason == ""

    def test_malformed_json_is_kept_whole_rather_than_shredded(self):
        raw = '["unterminated'
        items, _ = _recover_overflow(raw)
        assert items == [raw]

    def test_trailing_garbage_after_the_array_does_not_lose_the_array(self):
        items, reason = _recover_overflow('["a", "b"] not json at all')
        assert items == ["a", "b"]
        assert reason == ""

    def test_empty_and_none(self):
        assert _recover_overflow(None) == ([], "")
        assert _recover_overflow("") == ([], "")
        assert _recover_overflow([]) == ([], "")
