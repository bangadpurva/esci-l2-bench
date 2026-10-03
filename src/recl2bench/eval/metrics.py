"""Per-user ranking metrics with binary relevance.

Conventions (pre-registered):
- `ranked` is the model's ordering of the candidate pool, best first.
- `positives` is the set of ALL positives in the user's outcome window, including
  ones L1 missed. Recall and IDCG use this full set, so an L1 miss costs every
  model equally and oracle NDCG within the pool can be < 1.
- A user with zero positives is undefined (the cohort requires >= 1); we raise.
"""
from __future__ import annotations

import math
from typing import Hashable, Iterable, Sequence

import numpy as np


def _check(positives: set, k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")
    if not positives:
        raise ValueError("user has no positives; cohort requires at least one")


def _hits(ranked: Sequence[Hashable], positives: set, k: int) -> list[int]:
    return [1 if item in positives else 0 for item in list(ranked)[:k]]


def dcg_at_k(rels: Sequence[int]) -> float:
    return float(sum(r / math.log2(i + 2) for i, r in enumerate(rels)))


def ndcg_at_k(ranked: Sequence[Hashable], positives: Iterable[Hashable], k: int) -> float:
    positives = set(positives)
    _check(positives, k)
    dcg = dcg_at_k(_hits(ranked, positives, k))
    idcg = dcg_at_k([1] * min(k, len(positives)))
    return dcg / idcg


def precision_at_k(ranked, positives, k: int) -> float:
    positives = set(positives)
    _check(positives, k)
    return sum(_hits(ranked, positives, k)) / k


def recall_at_k(ranked, positives, k: int) -> float:
    positives = set(positives)
    _check(positives, k)
    return sum(_hits(ranked, positives, k)) / len(positives)


def hit_rate_at_k(ranked, positives, k: int) -> float:
    positives = set(positives)
    _check(positives, k)
    return float(any(_hits(ranked, positives, k)))


def oracle_order(pool: Sequence[Hashable], positives: Iterable[Hashable]) -> list:
    """Ceiling within the pool: positives first, original order otherwise."""
    positives = set(positives)
    return [i for i in pool if i in positives] + [i for i in pool if i not in positives]


def order_by_scores(items: Sequence[Hashable], scores: Sequence[float]) -> list:
    """Descending by score; ties broken by original (L1) rank, stably."""
    scores = np.asarray(scores, dtype=float)
    if len(items) != len(scores):
        raise ValueError("items and scores differ in length")
    if np.isnan(scores).any():
        raise ValueError("NaN scores; log the failure instead of ranking it")
    idx = np.argsort(-scores, kind="stable")
    return [items[i] for i in idx]


PRIMARY = ("ndcg", 10)
SECONDARY = [
    ("precision", 10), ("precision", 30),
    ("ndcg", 30),
    ("recall", 10), ("recall", 30),
    ("hit_rate", 10), ("hit_rate", 30),
]
_FUNCS = {
    "ndcg": ndcg_at_k,
    "precision": precision_at_k,
    "recall": recall_at_k,
    "hit_rate": hit_rate_at_k,
}


def user_metrics(ranked, positives) -> dict[str, float]:
    out = {}
    for name, k in [PRIMARY, *SECONDARY]:
        out[f"{name}@{k}"] = _FUNCS[name](ranked, positives, k)
    return out
