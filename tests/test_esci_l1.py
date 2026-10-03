import numpy as np
import pandas as pd
import pytest

from recl2bench.esci import l1 as L
from recl2bench.esci.l1 import normalize


def test_exact_topk_matches_brute_force_across_chunks():
    rng = np.random.default_rng(0)
    V = normalize(rng.normal(size=(1037, 16)).astype(np.float32))
    Q = normalize(rng.normal(size=(7, 16)).astype(np.float32))
    idx, sc = L.exact_topk(Q, V, 25, chunk=100)
    brute = np.argsort(-(Q @ V.T), axis=1)[:, :25]
    assert (idx == brute).all()
    assert np.all(np.diff(sc, axis=1) <= 1e-6)


def test_hnsw_close_to_exact():
    pytest.importorskip("faiss")
    rng = np.random.default_rng(1)
    V = normalize(rng.normal(size=(3000, 32)).astype(np.float32))
    Q = normalize(rng.normal(size=(20, 32)).astype(np.float32))
    assert L.overlap_at(L.hnsw_topk(Q, V, 50)[0], L.exact_topk(Q, V, 50)[0], 50) > 0.95


def test_query_prompt_format():
    assert L.qwen_query_prompt("Do X") == "Instruct: Do X\nQuery:"


def test_judged_pools_and_diagnostics():
    q = pd.DataFrame({"query_id": [1], "query": ["fan"]})
    qr = pd.DataFrame({"query_id": [1, 1, 1], "product_id": ["a", "b", "zz"], "label": ["E", "I", "E"]})
    V = np.eye(3, dtype=np.float32)
    Qv = np.array([[0.1, 0.9, 0.0]], dtype=np.float32)
    jp = L.judged_pools(q, qr, Qv, V, {"a": 0, "b": 1, "c": 2})
    assert jp.parent_asin.tolist() == ["b", "a"]            # zz not in catalog; order by cosine
    rp = L.retrieved_pools(q, np.array([[2, 0, 1]]), np.array([[0.9, 0.5, 0.1]]), ["a", "b", "c"])
    d = L.l1_diagnostics(rp, qr, ks=(1, 3))
    assert d["recall_E@3"] == pytest.approx(0.5) and d["judged@1"] == 0.0 and d["queries_with_E@3"] == 1.0
