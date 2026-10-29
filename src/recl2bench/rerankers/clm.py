"""CLM-v0.1-8B adapter (Contrastive-LM/CLM-v0.1-8B, `pip install contrastive-lm`).

Serving (GPU): vLLM serves Qwen/Qwen3-8B as a pooling model, and the CLM Engine scores a
query against its candidates in one call:

    vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b --runner pooling \
        --max-model-len 2048 --port 8090
    Engine(emb_url="http://127.0.0.1:8090/v1/embeddings").rank(query, candidates, instructions)
    -> [{"rank": 1, "candidate": "...", "prob": 0.99}, ...]

Probabilities are normalised over the candidate set, so CLM is excluded from calibration
metrics. One call per query: the model sees the whole list but scores each candidate
from its own encoding.
"""
from __future__ import annotations

from .base import Reranker, RerankResult


class CLMReranker(Reranker):
    name = "clm"

    def __init__(self, emb_url: str, engine=None, engine_kwargs: dict | None = None,
                 model_version: str = "CLM-v0.1-8B", instructions: str | None = None,
                 shuffle_seed: int | None = None):
        if engine is None:
            from clm import Engine
            engine = Engine(emb_url=emb_url, **(engine_kwargs or {}))
        self.engine, self.model_version, self.instructions = engine, model_version, instructions
        self.shuffle_seed = shuffle_seed          # order-sensitivity control (revision)

    def _score(self, user_profile, candidate_items):
        if self.shuffle_seed is None:
            return self._score_ordered(user_profile, candidate_items)
        import random
        perm = list(range(len(candidate_items)))
        random.Random(f"{self.shuffle_seed}:{user_profile.user_id}").shuffle(perm)
        res = self._score_ordered(user_profile, [candidate_items[i] for i in perm])
        scores = [float("nan")] * len(perm)
        for pos, i in enumerate(perm):
            scores[i] = res.scores[pos]
        res.scores = scores
        return res

    def _score_ordered(self, user_profile, candidate_items):
        texts = [c.text for c in candidate_items]
        out = self.engine.rank(user_profile.text, texts, instructions=self.instructions)
        if out and "index" in out[0]:
            by_index = {int(o["index"]): float(o["prob"]) for o in out}
            scores = [by_index.get(i, float("nan")) for i in range(len(texts))]
        else:   # map back by text; identical texts get identical scores
            by_text = {o["candidate"]: float(o["prob"]) for o in out}
            scores = [by_text.get(t, float("nan")) for t in texts]
        missing = sum(s != s for s in scores)
        return RerankResult(scores=scores, model_version=self.model_version, failures=missing)


class CLMPointwise(Reranker):
    """Interface control: put CLM through the same pointwise yes/no interface as Jev (revision).

    One call per (query, product): the state is the query plus the product, the two candidate
    answers are "yes" and "no", and the score is the probability CLM gives "yes". With two options the
    list normalisation is over {yes, no} only, so the score no longer depends on the other
    products in the list.
    """
    name = "clm_pointwise"

    def __init__(self, emb_url: str, question: str, engine=None, engine_kwargs: dict | None = None,
                 model_version: str = "CLM-v0.1-8B", state_labels=("Search query", "Candidate product")):
        if engine is None:
            from clm import Engine
            engine = Engine(emb_url=emb_url, **(engine_kwargs or {}))
        self.engine, self.model_version, self.question = engine, model_version, question
        self.state_labels = tuple(state_labels)

    def _score(self, user_profile, candidate_items):
        head, cand = self.state_labels
        scores, failures = [], 0
        for c in candidate_items:
            state = f"{head}\n{user_profile.text}\n\n{cand}\n{c.text}"
            try:
                out = self.engine.rank(state, ["yes", "no"], instructions=self.question)
                by_text = {str(o["candidate"]).strip().lower(): float(o["prob"]) for o in out}
                scores.append(by_text["yes"])
            except Exception:  # noqa: BLE001 - recorded as a failure, never ranked
                scores.append(float("nan"))
                failures += 1
        return RerankResult(scores=scores, model_version=self.model_version + " (pointwise yes/no)",
                            failures=failures)
