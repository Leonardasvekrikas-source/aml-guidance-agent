"""Metric definitions.

These assertions encode what the README's numbers mean. If someone later
"optimises" recall_at_k into hit-rate, these fail — which is the point, because
that change would silently inflate every figure in the results table.
"""

from __future__ import annotations

import pytest

from aml_agent.evaluation.metrics import (
    hit_at_k,
    mean,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestRecallAtK:
    def test_all_gold_in_top_k(self):
        assert recall_at_k([1, 2, 3, 4, 5], [1, 3], 5) == 1.0

    def test_partial_recall_is_a_fraction_not_a_hit(self):
        # Two gold chunks, one retrieved. Recall is 0.5, NOT 1.0.
        # This is the difference between recall and hit-rate, and conflating
        # them overstates multi-chunk questions.
        assert recall_at_k([1, 9, 9, 9, 9], [1, 2], 5) == 0.5

    def test_gold_beyond_k_does_not_count(self):
        assert recall_at_k([9, 9, 9, 9, 9, 1], [1], 5) == 0.0
        assert recall_at_k([9, 9, 9, 9, 9, 1], [1], 10) == 1.0

    def test_no_gold_scores_zero_rather_than_dividing_by_zero(self):
        assert recall_at_k([1, 2, 3], [], 5) == 0.0

    def test_empty_retrieval(self):
        assert recall_at_k([], [1, 2], 5) == 0.0


class TestHitAtK:
    def test_one_of_two_gold_is_a_full_hit(self):
        # The deliberate contrast with recall_at_k above.
        assert hit_at_k([1, 9, 9, 9, 9], [1, 2], 5) == 1.0
        assert recall_at_k([1, 9, 9, 9, 9], [1, 2], 5) == 0.5

    def test_no_gold_retrieved(self):
        assert hit_at_k([7, 8, 9], [1, 2], 5) == 0.0


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank([1, 2, 3], [1]) == 1.0

    def test_third_position(self):
        assert reciprocal_rank([7, 8, 1], [1]) == pytest.approx(1 / 3)

    def test_uses_first_gold_encountered_not_best_gold(self):
        # Gold 3 appears at rank 1, gold 1 at rank 3. RR is 1.0.
        assert reciprocal_rank([3, 8, 1], [1, 3]) == 1.0

    def test_absent_gold_scores_zero(self):
        assert reciprocal_rank([7, 8, 9], [1]) == 0.0


class TestPrecisionAtK:
    def test_two_of_five_relevant(self):
        assert precision_at_k([1, 2, 9, 9, 9], [1, 2], 5) == pytest.approx(0.4)

    def test_shorter_result_list_uses_actual_length(self):
        assert precision_at_k([1, 2], [1, 2], 5) == 1.0

    def test_zero_k(self):
        assert precision_at_k([1], [1], 0) == 0.0


class TestMean:
    def test_empty_is_zero_not_an_error(self):
        assert mean([]) == 0.0

    def test_average(self):
        assert mean([1.0, 0.0, 0.5]) == pytest.approx(0.5)
