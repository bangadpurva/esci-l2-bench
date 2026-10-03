"""CLM-v0.1-8B adapter (sekkit/CLM-v0.1-8B, `pip install contrastive-lm`).

Serving (GPU): vLLM serves Qwen/Qwen3-8B as a pooling model, and the CLM Engine scores a
query against its candidates in one call:

    vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b --runner pooling \
        --max-model-len 2048 --port 8090
    Engine(emb_url="http://127.0.0.1:8090/v1/embeddings").rank(query, candidates)
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
                 model_version: str = "CLM-v0.1-8B"):
        if engine is None:
            from clm import Engine
            engine = Engine(emb_url=emb_url, **(engine_kwargs or {}))
        self.engine, self.model_version = engine, model_version

    def _score(self, user_profile, candidate_items):
        texts = [c.text for c in candidate_items]
        out = self.engine.rank(user_profile.text, texts)
        if out and "index" in out[0]:
            by_index = {int(o["index"]): float(o["prob"]) for o in out}
            scores = [by_index.get(i, float("nan")) for i in range(len(texts))]
        else:   # map back by text; identical texts get identical scores
            by_text = {o["candidate"]: float(o["prob"]) for o in out}
            scores = [by_text.get(t, float("nan")) for t in texts]
        missing = sum(s != s for s in scores)
        return RerankResult(scores=scores, model_version=self.model_version, failures=missing)
