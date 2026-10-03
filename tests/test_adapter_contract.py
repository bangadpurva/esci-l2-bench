import math

import pytest

from recl2bench.rerankers.base import ContractError, Reranker, RerankResult
from recl2bench.rerankers.decision import (BackendReply, DecisionReranker, RetryableError,
                                           YesNoQuestion, Backend, make_backend)
from recl2bench.rerankers.dummy import L1OrderReranker, RandomReranker
from recl2bench.rerankers.text import WhitespaceTokenizer, cut, render_item

Q = YesNoQuestion("yes_no_v1", "Will they buy it?", "matches", "doesn't")


def test_l1_order_preserves_rank(profile, candidates):
    s = L1OrderReranker().rerank(profile, candidates).scores
    assert s == sorted(s, reverse=True)


def test_random_is_deterministic_per_user(profile, candidates):
    assert RandomReranker(3).rerank(profile, candidates).scores == \
        RandomReranker(3).rerank(profile, candidates).scores


class _Short(Reranker):
    name = "short"
    def _score(self, p, c):
        return RerankResult(scores=[0.0])


class _SilentNaN(Reranker):
    name = "silent"
    def _score(self, p, c):
        return RerankResult(scores=[math.nan] * len(c), failures=0)


def test_contract_rejects_wrong_length(profile, candidates):
    with pytest.raises(ContractError):
        _Short().rerank(profile, candidates)


def test_contract_rejects_unlogged_nan(profile, candidates):
    with pytest.raises(ContractError):
        _SilentNaN().rerank(profile, candidates)


class FakeBackend(Backend):
    """Scores by candidate index found in the state; fails on 'item 3'; 429 once."""
    name = "fake"
    max_questions = 2

    def __init__(self):
        self.calls = 0
        self.raised = False

    def ask(self, state, questions):
        self.calls += 1
        if not self.raised:
            self.raised = True
            raise RetryableError("429")
        if "item 3" in state and len(questions) == 1:
            raise RuntimeError("boom")
        return BackendReply([0.1 * (k + 1) for k in range(len(questions))], "fake-1.0")


def test_decision_per_pair_counts_retries_and_failures(profile, candidates):
    rr = DecisionReranker(FakeBackend(), Q, "per_pair", concurrency=1, sleep=lambda s: None)
    res = rr.rerank(profile, candidates)
    assert res.retries == 1 and res.failures == 1
    assert rr.error_samples == {"RuntimeError: boom": 1}
    assert math.isnan(res.scores[3])
    assert res.model_version == "fake-1.0"


def test_decision_fan_out_chunks_by_max_questions(profile, candidates):
    be = FakeBackend()
    res = DecisionReranker(be, Q, "fan_out", sleep=lambda s: None).rerank(profile, candidates)
    assert be.calls == 1 + 3          # one 429 retry + ceil(5/2) requests
    assert res.failures == 0 and not any(math.isnan(s) for s in res.scores)


def test_truncation_and_item_render():
    tok = WhitespaceTokenizer()
    t = cut("a b c d e", tok, 3)
    assert t.truncated and t.text == "a b c"
    r = render_item({"title": "Zelda", "categories": ["Games", "Switch"], "price": None}, tok, 160)
    assert r.text == "Title: Zelda\nCategories: Games; Switch"
