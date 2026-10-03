"""Metric tests against hand-computed examples (written before any model runs)."""
import math

import pytest

from recl2bench.eval import metrics as M

L2 = math.log2


def test_ndcg_single_positive_at_rank_1_is_one():
    assert M.ndcg_at_k(["a", "b", "c"], {"a"}, 10) == pytest.approx(1.0)


def test_ndcg_single_positive_at_rank_3():
    # DCG = 1/log2(4) = 0.5; IDCG = 1  -> 0.5
    assert M.ndcg_at_k(["x", "y", "a"], {"a"}, 10) == pytest.approx(0.5)


def test_ndcg_two_positives_hand_computed():
    # ranks 1 and 3 hit: DCG = 1 + 0.5 = 1.5
    # IDCG with 2 positives = 1 + 1/log2(3)
    ranked = ["a", "x", "b", "y"]
    expected = 1.5 / (1 + 1 / L2(3))
    assert M.ndcg_at_k(ranked, {"a", "b"}, 10) == pytest.approx(expected)


def test_ndcg_counts_positives_l1_missed():
    # 3 positives, only 1 in pool at rank 1. IDCG uses all 3.
    expected = 1 / (1 + 1 / L2(3) + 1 / L2(4))
    assert M.ndcg_at_k(["a", "x"], {"a", "m1", "m2"}, 10) == pytest.approx(expected)


def test_ndcg_idcg_capped_at_k():
    # 5 positives, k=2, both top-2 hit -> perfect
    assert M.ndcg_at_k(["a", "b"], set("abcde"), 2) == pytest.approx(1.0)


def test_ndcg_cutoff_ignores_hits_after_k():
    assert M.ndcg_at_k(["x", "y", "a"], {"a"}, 2) == 0.0


def test_precision_recall_hit_rate():
    ranked = ["a", "x", "b", "y", "z"]
    pos = {"a", "b", "c", "d"}
    assert M.precision_at_k(ranked, pos, 5) == pytest.approx(2 / 5)
    assert M.precision_at_k(ranked, pos, 10) == pytest.approx(2 / 10)  # fixed denominator k
    assert M.recall_at_k(ranked, pos, 5) == pytest.approx(2 / 4)
    assert M.hit_rate_at_k(ranked, pos, 1) == 1.0
    assert M.hit_rate_at_k(["x", "a"], pos, 1) == 0.0


def test_zero_positives_raises():
    with pytest.raises(ValueError):
        M.ndcg_at_k(["a"], set(), 10)


def test_oracle_order_puts_positives_first_and_is_stable():
    assert M.oracle_order(["x", "a", "y", "b"], {"a", "b"}) == ["a", "b", "x", "y"]


def test_order_by_scores_breaks_ties_by_l1_rank():
    assert M.order_by_scores(["a", "b", "c"], [0.5, 0.9, 0.5]) == ["b", "a", "c"]


def test_order_by_scores_rejects_nan():
    with pytest.raises(ValueError):
        M.order_by_scores(["a", "b"], [0.1, float("nan")])


def test_user_metrics_keys():
    out = M.user_metrics(["a"], {"a"})
    assert out["ndcg@10"] == pytest.approx(1.0)
    assert set(out) == {"ndcg@10", "precision@10", "precision@30", "ndcg@30",
                        "recall@10", "recall@30", "hit_rate@10", "hit_rate@30"}
