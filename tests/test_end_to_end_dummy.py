"""Synthetic plumbing test: frozen pool -> dummy rerankers -> metrics -> bootstrap -> manifest."""
import json

import numpy as np
import pandas as pd
import pytest

from recl2bench.eval.bootstrap import compare_to_baseline
from recl2bench.eval.metrics import ndcg_at_k, oracle_order
from recl2bench import pools as P
from recl2bench.manifest import RunManifest
from recl2bench.rerankers.base import ProfileItem, UserProfile
from recl2bench.rerankers.dummy import L1OrderReranker, RandomReranker
from recl2bench.runner import run


@pytest.fixture
def world(tmp_path):
    rng = np.random.default_rng(0)
    users = [f"u{i:03d}" for i in range(60)]
    rows, pos, profiles = [], {}, {}
    for u in users:
        items = [f"{u}_i{j}" for j in range(100)]
        rows += [{"user_id": u, "query_time": pd.Timestamp("2022-07-16", tz="UTC"), "rank": j,
                  "parent_asin": it, "l1_score": 1 - j / 100} for j, it in enumerate(items)]
        # positives skew toward the top of L1, plus one L1 miss
        pos[u] = set(rng.choice(items[:30], 2, replace=False)) | {f"{u}_missed"}
        profiles[u] = UserProfile(u, "2022-07-16", (ProfileItem("x", "t", 5),), text="p")
    path = tmp_path / "pools_test.parquet"
    digest = P.freeze(pd.DataFrame(rows), path)
    df, _ = P.load(path)
    text = {r: "item" for r in df["parent_asin"]}
    return df, digest, profiles, text, pos


def test_pipeline(world, tmp_path):
    pools, digest, profiles, text, pos = world
    l1, _, t1 = run(L1OrderReranker(), pools, profiles, text, pos)
    rnd, scores, _ = run(RandomReranker(0), pools, profiles, text, pos)

    assert len(scores) == 60 * 100 and len(l1) == 60
    # L1 order must reproduce ANN-only exactly
    expect = [ndcg_at_k(pools[pools.user_id == u].sort_values("rank").parent_asin.tolist(), pos[u], 10)
              for u in sorted(pos)]
    assert np.allclose(l1["ndcg@10"].to_numpy(), expect)
    # oracle within pool is < 1 because of the L1 miss
    u = "u000"
    oracle = ndcg_at_k(oracle_order(pools[pools.user_id == u].parent_asin.tolist(), pos[u]), pos[u], 10)
    assert 0 < oracle < 1

    res = compare_to_baseline({"random": rnd["ndcg@10"].to_numpy()}, l1["ndcg@10"].to_numpy(),
                              n_resamples=2000)
    assert res[0].mean_diff < 0 and not res[0].significant

    m = RunManifest(run_id="test-run", track="A", model="l1_order", model_version="l1_order",
                    scoring_mode=None, pool_sha256=digest, template_sha256=None, budget_tokens=512,
                    failures=t1["failures"])
    saved = json.loads(m.write(tmp_path / "runs").read_text())
    assert saved["pool_sha256"] == digest and saved["budget_tokens"] == 512
