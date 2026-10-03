"""Dummy rerankers for plumbing tests and floor baselines."""
from __future__ import annotations

import hashlib

import numpy as np

from .base import CandidateItem, Reranker, RerankResult, UserProfile


class L1OrderReranker(Reranker):
    """Returns L1 order unchanged: NDCG must equal ANN-only exactly."""
    name = "l1_order"

    def _score(self, user_profile, candidate_items):
        return RerankResult(scores=[float(-c.l1_rank) for c in candidate_items],
                            model_version="l1_order")


class RandomReranker(Reranker):
    """Random order, deterministic per (seed, user) so reruns match."""
    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def _score(self, user_profile: UserProfile, candidate_items: list[CandidateItem]):
        h = int(hashlib.sha256(f"{self.seed}:{user_profile.user_id}".encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(h)
        return RerankResult(scores=rng.random(len(candidate_items)).tolist(),
                            model_version=f"random-seed{self.seed}")
