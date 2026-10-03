"""Search metrics with graded, incomplete labels (ESCI).

Conventions (pre-registered in PREREGISTRATION.md):
- `labels` maps product_id -> "E" | "S" | "C" | "I" for the query; anything absent is
  UNJUDGED.
- Strict precision@k: relevant = E; unjudged counts as not relevant.
- Lenient precision@k: relevant = E or S; unjudged counts as not relevant.
- Judged-only ("condensed") precision@k: unjudged items are removed from the ranking
  first, then strict precision@k is computed on what remains (Sakai, 2007).
- judged@k: share of the top-k that carries any label.
- NDCG@10 with ESCI gains E=1, S=0.1, C=0.01, I=0; unjudged gain 0; ideal ranking
  built from all judged products of the query.
- Recall@k: share of the query's E products that appear in the top-k.
"""
from __future__ import annotations

import math
from typing import Hashable, Mapping, Sequence

from .data import GAINS

PRIMARY = "p@10"
COLUMNS = ["p@10", "p@5", "p@10_es", "p@5_es", "p@10_judged", "p@5_judged",
           "judged@10", "ndcg@10", "recall@100"]


def _dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def precision(ranked: Sequence[Hashable], labels: Mapping, k: int, rel=("E",)) -> float:
    return sum(1 for x in ranked[:k] if labels.get(x) in rel) / k


def judged_precision(ranked, labels, k, rel=("E",)) -> float:
    condensed = [x for x in ranked if x in labels]
    return precision(condensed, labels, k, rel) if condensed else 0.0


def judged_at(ranked, labels, k) -> float:
    return sum(1 for x in ranked[:k] if x in labels) / k


def ndcg(ranked, labels, k: int = 10) -> float:
    gains = [GAINS.get(labels.get(x), 0.0) for x in ranked[:k]]
    ideal = sorted((GAINS[v] for v in labels.values()), reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg > 0 else 0.0


def recall(ranked, labels, k: int, rel=("E",)) -> float:
    total = sum(1 for v in labels.values() if v in rel)
    return sum(1 for x in ranked[:k] if labels.get(x) in rel) / total if total else 0.0


def query_metrics(ranked: Sequence[Hashable], labels: Mapping) -> dict[str, float]:
    ranked = list(ranked)
    return {
        "p@10": precision(ranked, labels, 10), "p@5": precision(ranked, labels, 5),
        "p@10_es": precision(ranked, labels, 10, ("E", "S")),
        "p@5_es": precision(ranked, labels, 5, ("E", "S")),
        "p@10_judged": judged_precision(ranked, labels, 10),
        "p@5_judged": judged_precision(ranked, labels, 5),
        "judged@10": judged_at(ranked, labels, 10),
        "ndcg@10": ndcg(ranked, labels, 10),
        "recall@100": recall(ranked, labels, 100),
    }
