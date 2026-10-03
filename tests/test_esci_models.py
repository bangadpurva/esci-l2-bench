import sys
from pathlib import Path

import pytest

from recl2bench.rerankers.base import CandidateItem, UserProfile
from recl2bench.rerankers.clm import CLMReranker

ROOT = Path(__file__).resolve().parents[1]


class FakeEngine:
    def __init__(self, with_index):
        self.with_index, self.calls = with_index, []

    def rank(self, query, candidates):
        self.calls.append((query, list(candidates)))
        probs = [0.1 * (i + 1) for i in range(len(candidates))]
        order = sorted(range(len(candidates)), key=lambda i: -probs[i])
        return [{"rank": r + 1, "candidate": candidates[i], "prob": probs[i], **({"index": i} if self.with_index else {})}
                for r, i in enumerate(order)]


@pytest.mark.parametrize("with_index", [True, False])
def test_clm_maps_scores_back_one_call_per_query(with_index):
    eng = FakeEngine(with_index)
    rr = CLMReranker("http://x", engine=eng)
    p = UserProfile("q1", "", (), (), "80 cfm fan")
    c = [CandidateItem(f"i{k}", k, 1.0, f"text {k}") for k in range(4)]
    res = rr.rerank(p, c)
    assert res.scores == pytest.approx([0.1, 0.2, 0.3, 0.4]) and res.failures == 0
    assert len(eng.calls) == 1 and eng.calls[0][0] == "80 cfm fan"


def test_decision_models_get_search_framing(monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts"))
    import esci_score as score
    from recl2bench.rerankers.decision import build_state
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    args = type("A", (), {"scoring_mode": None, "seed": 0, "concurrency": None})()
    rr, cfg, tsha, mode, conc = score.build("jev", args, ROOT / "configs", "esci_v1")
    assert rr.state_labels == ("Search query", "Candidate product")
    assert "exactly match" in rr.question.question and tsha
    s = build_state(UserProfile("q", "", (), (), "80 cfm fan"), item=CandidateItem("i", 0, 1.0, "Fan"),
                    labels=rr.state_labels)
    assert s.startswith("Search query\n80 cfm fan") and "Candidate product\nFan" in s
