"""Graded scoring from several binary questions (added in revision, answers reviewer concerns on
single-question wording and graded relevance).

Score = sum_k P(yes | question_k). With nested questions ("exact match", "at least an acceptable
substitute") this is the expected relevance grade on a 0..2 scale, E[g] = sum_k P(g >= k), so it needs no
weights and nothing to tune. A candidate fails if any component fails.
"""
from __future__ import annotations

import math

from .base import Reranker, RerankResult


class SumReranker(Reranker):
    def __init__(self, parts: list[Reranker]):
        if len(parts) < 2:
            raise ValueError("SumReranker needs at least two components")
        self.parts = parts
        self.name = "+".join(getattr(p, "name", "part") for p in parts)

    def _score(self, user_profile, candidate_items):
        results = [p.rerank(user_profile, candidate_items) for p in self.parts]
        scores = [math.nan if any(math.isnan(r.scores[i]) for r in results)
                  else float(sum(r.scores[i] for r in results)) for i in range(len(candidate_items))]
        versions = sorted({r.model_version for r in results if r.model_version})
        return RerankResult(scores=scores, model_version=" + ".join(versions) or None,
                            failures=sum(math.isnan(s) for s in scores),
                            retries=sum(r.retries for r in results),
                            extra={"components": [r.extra for r in results]})
