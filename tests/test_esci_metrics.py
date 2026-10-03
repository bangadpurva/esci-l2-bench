import math

import pytest

from recl2bench.esci import metrics as M
from recl2bench.esci.data import clean, product_text

L = {"a": "E", "b": "S", "c": "I", "d": "E", "e": "C"}   # x, y, z unjudged


def test_strict_and_lenient_precision():
    r = ["a", "x", "b", "d", "y"]
    assert M.precision(r, L, 5) == pytest.approx(2 / 5)
    assert M.precision(r, L, 5, ("E", "S")) == pytest.approx(3 / 5)
    assert M.precision(r, L, 10) == pytest.approx(2 / 10)     # fixed denominator k


def test_judged_only_precision_drops_unjudged_first():
    r = ["x", "y", "a", "z", "d", "c"]
    # condensed: a, d, c -> p@5 = 2/5
    assert M.judged_precision(r, L, 5) == pytest.approx(2 / 5)
    assert M.judged_at(r, L, 5) == pytest.approx(2 / 5)


def test_graded_ndcg_hand_computed():
    r = ["b", "a", "x"]
    dcg = 0.1 / math.log2(2) + 1.0 / math.log2(3)
    ideal = 1 / math.log2(2) + 1 / math.log2(3) + 0.1 / math.log2(4) + 0.01 / math.log2(5)
    assert M.ndcg(r, L, 10) == pytest.approx(dcg / ideal)
    assert M.ndcg(["a", "d", "b", "e", "c"], L, 10) == pytest.approx(1.0)


def test_recall_of_exact():
    assert M.recall(["a", "x"], L, 100) == pytest.approx(0.5)
    assert M.recall(["x"], {"c": "I"}, 100) == 0.0


def test_query_metrics_columns():
    assert set(M.query_metrics(["a"], L)) == set(M.COLUMNS)


def test_product_text_strips_html_and_keeps_title_first():
    row = {"product_title": "Fan 80 CFM", "product_brand": "Acme", "product_color": None,
           "product_bullet_point": "Quiet\nEnergy Star", "product_description": "<p>Great&nbsp;fan</p><br/>"}
    t = product_text(row)
    assert t.splitlines()[0] == "Fan 80 CFM"
    assert "Brand: Acme" in t and "Features: Quiet; Energy Star" in t and "Description: Great fan" in t
    assert "<" not in t and "Color" not in t
    assert clean(float("nan")) == ""
