"""The single contract every L2 model and baseline implements.

    rerank(user_profile, candidate_items) -> RerankResult

Rules the contract enforces:
- One score per candidate, aligned to the input order (no reordering, no drops).
- Scores are finite floats; a failed candidate is NaN plus a recorded failure,
  and the runner decides what to do with it (never silently ranked).
- Every model receives the same pre-truncated text (see text.py).
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProfileItem:
    item_id: str
    title: str
    rating: float
    attrs: tuple[str, ...] = ()


@dataclass(frozen=True)
class UserProfile:
    user_id: str
    query_time: str
    liked: tuple[ProfileItem, ...]
    disliked: tuple[ProfileItem, ...] = ()
    text: str | None = None          # budgeted rendering, filled by text.render_profile


@dataclass(frozen=True)
class CandidateItem:
    item_id: str
    l1_rank: int
    l1_score: float
    text: str                        # already cut to the item budget
    images: tuple[str, ...] = ()     # optional (Clef ablation)


@dataclass
class RerankResult:
    scores: list[float]
    model_version: str | None = None        # e.g. the `model` field each API response returns
    failures: int = 0
    retries: int = 0
    cache_hits: int = 0
    extra: dict = field(default_factory=dict)


class Reranker(ABC):
    name: str = "base"

    @abstractmethod
    def _score(self, user_profile: UserProfile,
               candidate_items: list[CandidateItem]) -> RerankResult: ...

    def rerank(self, user_profile: UserProfile,
               candidate_items: list[CandidateItem]) -> RerankResult:
        if not candidate_items:
            raise ValueError("empty candidate pool")
        result = self._score(user_profile, candidate_items)
        if len(result.scores) != len(candidate_items):
            raise ContractError(
                f"{self.name}: {len(result.scores)} scores for {len(candidate_items)} candidates")
        bad = [s for s in result.scores if not (isinstance(s, float) and (math.isfinite(s) or math.isnan(s)))]
        if bad:
            raise ContractError(f"{self.name}: non-float or infinite scores {bad[:3]}")
        nan = sum(math.isnan(s) for s in result.scores)
        if nan > result.failures:
            raise ContractError(f"{self.name}: {nan} NaN scores but only {result.failures} failures logged")
        return result


class ContractError(RuntimeError):
    pass
