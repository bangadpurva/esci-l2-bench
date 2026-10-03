"""Score a frozen pool with one reranker and compute per-user metrics.

Usage from code:
    per_user, scores = run(reranker, pools, profiles, item_text, positives)
"""
from __future__ import annotations

import math
import time

import pandas as pd

from .eval.metrics import order_by_scores, user_metrics
from .rerankers.base import CandidateItem, Reranker, UserProfile


def run(reranker: Reranker, pools: pd.DataFrame, profiles: dict[str, UserProfile],
        item_text: dict[str, str], positives: dict[str, set], metric_fn=None):
    """metric_fn(ranked, user_id) -> dict overrides the default binary metrics (ESCI)."""
    rows, per_user = [], []
    totals = {"failures": 0, "retries": 0, "versions": set()}
    groups = pools.groupby("user_id", sort=True)
    try:
        from tqdm import tqdm
        groups = tqdm(groups, total=pools.user_id.nunique(), desc=getattr(reranker, "name", "rerank"),
                      unit="user", mininterval=5)
    except ImportError:
        pass
    for user_id, g in groups:
        g = g.sort_values("rank")
        cands = [CandidateItem(r.parent_asin, int(r.rank), float(r.l1_score), item_text[r.parent_asin])
                 for r in g.itertuples()]
        t0 = time.perf_counter()
        res = reranker.rerank(profiles[user_id], cands)
        dt = time.perf_counter() - t0
        totals["failures"] += res.failures
        totals["retries"] += res.retries
        if res.model_version:
            totals["versions"].add(res.model_version)
        # Failed candidates fall to the bottom, in L1 order; counted in failures.
        safe = [(-math.inf if math.isnan(s) else s) for s in res.scores]
        ranked = order_by_scores([c.item_id for c in cands], safe)
        m = metric_fn(ranked, user_id) if metric_fn else user_metrics(ranked, positives[user_id])
        per_user.append({"user_id": user_id, "latency_s": dt, "failures": res.failures, **m})
        rows += [{"user_id": user_id, "parent_asin": c.item_id, "l1_rank": c.l1_rank, "score": s}
                 for c, s in zip(cands, res.scores)]
    return pd.DataFrame(per_user), pd.DataFrame(rows), totals
