"""Analyses added in the peer-review revision (PREREGISTRATION.md, Amendment 5).

Everything here works from the files the main pipeline already writes (frozen pools, qrels, and each
run's per_user.parquet / scores.parquet), so the added analyses never touch the frozen lists.

- find_runs / latest:      locate runs across one or more runs directories
- ranked_ids:              rerank a run's saved scores exactly as runner.run does (NaN -> bottom)
- fuse_z / fuse_rrf:       score fusion of an L2 run with the L1 similarity
- evaluate_orders:         per-query metrics for any ranking function
- composition:             label mix of the top 10 (graded view of what a reranker promotes)
- diff_ci:                 paired bootstrap difference with percentile CI and bootstrap p
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np
import pandas as pd

from .eval.bootstrap import bootstrap_p_value, holm, paired_bootstrap
from .eval.metrics import order_by_scores
from .esci.metrics import query_metrics


def find_runs(runs_dirs: Iterable[str | Path], split: str, setting: str) -> dict[str, dict]:
    """Latest complete run per model name for (split, setting) across directories.

    Returns model -> {"dir", "manifest", "per_user"}. Smoke runs (limit_users) are skipped.
    """
    best: dict[str, dict] = {}
    for rd in runs_dirs:
        for d in sorted(Path(rd).glob("*/")):
            mf, pu = d / "manifest.json", d / "per_user.parquet"
            if not (mf.exists() and pu.exists()):
                continue
            m = json.loads(mf.read_text())
            u = pd.read_parquet(pu)
            if (m.get("setting") != setting or u["split"].iloc[0] != split
                    or int(u.get("limit_users", pd.Series([0])).iloc[0] or 0)):
                continue
            cur = best.get(m["model"])
            if cur is None or m["started_at"] > cur["manifest"]["started_at"]:
                best[m["model"]] = {"dir": d, "manifest": m, "per_user": u}
    return best


def ranked_ids(scores: pd.DataFrame) -> dict[str, list]:
    """query -> candidate ids best first, from a run's scores.parquet; NaN scores sink, ties keep L1 order."""
    out = {}
    for u, g in scores.sort_values(["user_id", "l1_rank"]).groupby("user_id", sort=True):
        s = [(-math.inf if (x != x) else float(x)) for x in g["score"]]
        out[str(u)] = order_by_scores(list(g["parent_asin"]), s)
    return out


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def _fill_nan(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    bad = np.isnan(x)
    if bad.all():
        return np.zeros_like(x)
    x[bad] = x[~bad].min()
    return x


def fuse_z(scores: pd.DataFrame, pools: pd.DataFrame, alpha: float) -> dict[str, list]:
    """Rank by alpha * z(L2 score) + (1 - alpha) * z(L1 similarity), z per query over its candidates.

    alpha = 1 is the L2 model alone, alpha = 0 is L1 order (ties break toward L1 order).
    """
    m = scores.merge(pools[["user_id", "parent_asin", "l1_score"]], on=["user_id", "parent_asin"], how="left")
    if m["l1_score"].isna().any():
        raise ValueError("scores contain candidates that are not in the frozen pool")
    out = {}
    for u, g in m.sort_values(["user_id", "l1_rank"]).groupby("user_id", sort=True):
        s = alpha * zscore(_fill_nan(g["score"].to_numpy())) + (1 - alpha) * zscore(g["l1_score"].to_numpy())
        out[str(u)] = order_by_scores(list(g["parent_asin"]), s.tolist())
    return out


def fuse_rrf(scores: pd.DataFrame, k: int = 60) -> dict[str, list]:
    """Reciprocal rank fusion (Cormack et al., 2009; k = 60, not tuned) of L1 order and the L2 order."""
    out = {}
    for u, g in scores.sort_values(["user_id", "l1_rank"]).groupby("user_id", sort=True):
        l1_rank = np.arange(1, len(g) + 1)
        s2 = _fill_nan(g["score"].to_numpy())
        l2_rank = np.empty(len(g), dtype=int)
        l2_rank[np.argsort(-s2, kind="stable")] = np.arange(1, len(g) + 1)
        s = 1.0 / (k + l1_rank) + 1.0 / (k + l2_rank)
        out[str(u)] = order_by_scores(list(g["parent_asin"]), s.tolist())
    return out


def evaluate_orders(orders: Mapping[str, list], labels: Mapping[str, Mapping]) -> pd.DataFrame:
    rows = [{"user_id": u, **query_metrics(r, labels.get(u, {}))} for u, r in orders.items()]
    return pd.DataFrame(rows).sort_values("user_id").reset_index(drop=True)


def select_alpha(valid_scores: pd.DataFrame, valid_pools: pd.DataFrame, valid_labels: Mapping,
                 grid: Iterable[float] = tuple(round(0.1 * i, 1) for i in range(0, 11)),
                 metric: str = "p@10") -> tuple[float, dict[float, float]]:
    """Pick the fusion weight on VALIDATION queries only. Ties go to the larger alpha (more L2)."""
    res = {a: float(evaluate_orders(fuse_z(valid_scores, valid_pools, a), valid_labels)[metric].mean())
           for a in grid}
    best = max(res.values())
    return max(a for a, v in res.items() if v >= best - 1e-12), res


def composition(orders: Mapping[str, list], labels: Mapping[str, Mapping], k: int = 10,
                classes: tuple[str, ...] = ("E", "S", "C", "I")) -> dict[str, float]:
    """Mean number of each label class (and unlabelled, "U") among each query's top k."""
    tot = {c: 0.0 for c in (*classes, "U")}
    for u, r in orders.items():
        lab = labels.get(u, {})
        for x in r[:k]:
            tot[lab.get(x, "U") if lab.get(x, "U") in tot else "U"] += 1
    n = max(len(orders), 1)
    return {c: v / n for c, v in tot.items()}


def diff_ci(a: np.ndarray, b: np.ndarray, n_resamples: int = 10_000, seed: int = 0) -> dict:
    """Mean of a - b with 95% percentile CI and two-sided bootstrap p (unadjusted)."""
    mean, boot = paired_bootstrap(np.asarray(a, float), np.asarray(b, float), n_resamples, seed)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return {"mean": mean, "lo": float(lo), "hi": float(hi), "p": bootstrap_p_value(boot)}


def diff_table(pairs: list[tuple[str, np.ndarray, np.ndarray]], n_resamples: int = 10_000,
               seed: int = 0) -> list[dict]:
    """Several paired comparisons with Holm-adjusted p across the list."""
    rows = [{"name": n, **diff_ci(a, b, n_resamples, seed + 10 * i)} for i, (n, a, b) in enumerate(pairs)]
    for r, p in zip(rows, holm([r["p"] for r in rows])):
        r["p_holm"] = p
    return rows


def align(per_user: Mapping[str, pd.DataFrame], column: str) -> dict[str, np.ndarray]:
    """Per-query vectors for one metric, aligned on the queries all runs share."""
    common = sorted(set.intersection(*(set(df.user_id) for df in per_user.values())))
    return {m: df.set_index("user_id").loc[common, column].to_numpy() for m, df in per_user.items()}


def has_label(labels: Mapping[str, Mapping], cls: str = "E") -> set[str]:
    return {u for u, d in labels.items() if any(v == cls for v in d.values())}
