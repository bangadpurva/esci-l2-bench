"""Decision-model family: Jev and Clef behind one adapter.

The adapter owns everything model-independent: building the state, the frozen
yes/no question, scoring modes, concurrency, retries on 429/529, and failure
accounting. A Backend owns only the HTTP call and response parsing.
"""
from __future__ import annotations

import json
import math
import random
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

from .base import CandidateItem, Reranker, RerankResult, UserProfile

ScoringMode = Literal["per_pair", "fan_out", "listwise"]


@dataclass(frozen=True)
class YesNoQuestion:
    id: str
    question: str
    true_means: str
    false_means: str


@dataclass
class BackendReply:
    probs: list[float]            # P(true) per question, in question order
    model_version: str | None     # the `model` field the response reports


class RetryableError(Exception):
    """Raise from a backend on HTTP 429/529 so the adapter retries."""


class Backend(ABC):
    name: str
    max_questions: int | None = None

    @abstractmethod
    def ask(self, state: str, questions: list[YesNoQuestion]) -> BackendReply: ...


def _ssl_context():
    """Use certifi's CA bundle when present (python.org macOS builds ship without one)."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


class SystemOneBackend(Backend):
    """Shared state-plus-typed-questions protocol (TypeSafe Jev, Cloudflare Clef).

    Request:  {"model", "state", "questions": {id: {"type": "noul", "instructions"}}}
    Response: {"model", "answers": {id: {"type": "noul", "noul": p}}, "usage": {...}},
              optionally wrapped in Cloudflare's {"result": ..., "success": ...}.
              Clef was verified live 2026-10-03; "probability" (seen in docs and
              SDKs) is also accepted.
    The noul type documents only `instructions`, so the true/false meanings are
    folded into the instruction text.
    """
    max_questions: int | None = None
    label = "systemone"

    def __init__(self, url: str, api_token: str, model: str, timeout: float = 60.0, transport=None):
        self.url, self.token, self.model, self.timeout = url, api_token, model, timeout
        self.transport = transport or self._post
        self.usage_input_tokens = 0

    def _post(self, url, headers, body):
        import urllib.error
        import urllib.request
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=_ssl_context()) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, {"error": e.read().decode(errors="replace")[:500]}
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            # network-level: handshake/connect/read timeouts, resets. Transient -> retry.
            raise RetryableError(f"network: {e}") from e

    @staticmethod
    def instructions(q: YesNoQuestion) -> str:
        return f"{q.question}\nAnswer true if: {q.true_means}\nAnswer false if: {q.false_means}"

    def ask(self, state, questions):
        cap = self.max_questions
        if not questions or (cap is not None and len(questions) > cap):
            raise ValueError(f"{self.label} takes 1-{cap or 'n'} questions, got {len(questions)}")
        keys = [f"q{i}" for i in range(len(questions))]   # ids safe for any item id
        body = {"model": self.model, "state": state,
                "questions": {k: {"type": "noul", "instructions": self.instructions(q)}
                              for k, q in zip(keys, questions)}}
        status, data = self.transport(self.url, {"Authorization": f"Bearer {self.token}",
                                                 "Content-Type": "application/json"}, body)
        if status in (429, 529) or status >= 500:
            raise RetryableError(f"HTTP {status}: {str(data)[:300]}")
        if status != 200:
            raise RuntimeError(f"{self.label} HTTP {status}: {data}")
        data = data.get("result", data)
        self.usage_input_tokens += int((data.get("usage") or {}).get("input_tokens", 0))
        probs = []
        for k in keys:
            a = data["answers"][k]
            v = a.get("noul", a.get("probability"))
            if v is None:
                raise ValueError(f"no 'noul' or 'probability' in {self.label} answer: {a}")
            probs.append(float(v))
        return BackendReply(probs, data.get("model"))


class ClefBackend(SystemOneBackend):
    """Cloudflare Workers AI: Clef / Clef Flash. Up to 64 questions per request."""
    max_questions = 64
    label = "Clef"

    def __init__(self, account_id: str, api_token: str, model_id: str = "@cf/cloudflare/clef-flash",
                 timeout: float = 60.0, transport=None):
        self.name = model_id.rsplit("/", 1)[-1]          # "clef-flash"
        super().__init__(f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_id}",
                         api_token, self.name, timeout, transport)


class JevBackend(SystemOneBackend):
    """TypeSafe Jev (docs.typesafe.ai): POST /v1/systemone, Bearer auth.

    Pin a versioned model (jev-1.13.0), never the jev-latest alias. Limits per the
    models page: 80 req/s, 100K tokens/s, 64K context per request.
    """
    label = "Jev"
    URL = "https://api.typesafe.ai/v1/systemone"

    def __init__(self, api_token: str, model: str = "jev-1.13.0", timeout: float = 60.0,
                 transport=None, url: str | None = None):
        if model.endswith("latest") or model.endswith("preview"):
            raise ValueError("pin a versioned Jev model, not an alias")
        self.name = model
        super().__init__(url or self.URL, api_token, model, timeout, transport)


def make_backend(backend: str, /, **cfg) -> Backend:
    """`cfg` is a whole model config (it may carry its own `name`, `family`, ...)."""
    name = backend
    if name in ("clef", "clef_flash"):
        import os
        return ClefBackend(os.environ[cfg.get("account_id_env", "CLOUDFLARE_ACCOUNT_ID")],
                           os.environ[cfg.get("api_key_env", "CLOUDFLARE_API_TOKEN")],
                           cfg.get("model_version", "@cf/cloudflare/clef-flash"))
    if name == "jev":
        import os
        return JevBackend(os.environ[cfg.get("api_key_env", "TYPESAFE_API_KEY")],
                          cfg.get("model_version", "jev-1.13.0"), url=cfg.get("endpoint"))
    raise ValueError(f"unknown decision backend {name!r}")


DEFAULT_STATE_LABELS = ("Shopper history", "Candidate product")


def build_state(profile: UserProfile, item: CandidateItem | None = None,
                items: list[CandidateItem] | None = None,
                labels: tuple[str, str] = DEFAULT_STATE_LABELS) -> str:
    head, cand = labels
    s = f"{head}\n{profile.text}\n"
    if item is not None:
        s += f"\n{cand}\n{item.text}\n"
    if items:
        s += f"\n{cand}s\n" + "\n\n".join(f"[{c.item_id}]\n{c.text}" for c in items) + "\n"
    return s


class DecisionReranker(Reranker):
    def __init__(self, backend: Backend, question: YesNoQuestion,
                 scoring_mode: ScoringMode = "per_pair", concurrency: int = 8,
                 max_retries: int = 6, shuffle_seed: int | None = None,
                 backoff_base: float = 1.0, backoff_cap: float = 30.0, sleep=None,
                 state_labels: tuple[str, str] = DEFAULT_STATE_LABELS):
        if scoring_mode == "listwise":
            raise NotImplementedError("listwise mode is a Track D variant; add after per_pair works")
        self.backend, self.question = backend, question
        self.scoring_mode, self.concurrency = scoring_mode, concurrency
        self.max_retries, self.shuffle_seed = max_retries, shuffle_seed
        self.backoff_base, self.backoff_cap = backoff_base, backoff_cap
        self.state_labels = tuple(state_labels)
        import time as _time
        self._sleep = sleep or _time.sleep
        self.retry_wait_s = 0.0          # total time spent backing off, reported separately
        self.name = f"{backend.name}:{scoring_mode}"
        self.error_samples: dict[str, int] = {}     # error text -> count, across the run

    def _record_error(self, e: Exception) -> None:
        import sys
        key = f"{type(e).__name__}: {str(e)[:300]}"
        if key not in self.error_samples:
            print(f"\n[{self.name}] backend error: {key}", file=sys.stderr, flush=True)
        self.error_samples[key] = self.error_samples.get(key, 0) + 1

    def _call(self, state, questions):
        retries = 0
        while True:
            try:
                return self.backend.ask(state, questions), retries
            except RetryableError:
                retries += 1
                if retries > self.max_retries:
                    raise
                # exponential backoff with full jitter: 429/529 means slow down, not retry now
                wait = random.uniform(0, min(self.backoff_cap, self.backoff_base * 2 ** (retries - 1)))
                self.retry_wait_s += wait
                self._sleep(wait)

    def _score(self, user_profile, candidate_items):
        if self.scoring_mode == "per_pair":
            return self._per_pair(user_profile, candidate_items)
        return self._fan_out(user_profile, candidate_items)

    def _per_pair(self, profile, items):
        def one(c):
            try:
                reply, r = self._call(build_state(profile, item=c, labels=self.state_labels), [self.question])
                return reply.probs[0], reply.model_version, r, 0
            except Exception as e:  # noqa: BLE001 - recorded as a failure, never ranked
                self._record_error(e)
                return math.nan, None, 0, 1

        with ThreadPoolExecutor(self.concurrency) as ex:
            out = list(ex.map(one, items))
        versions = {v for _, v, _, _ in out if v}
        return RerankResult(
            scores=[float(s) for s, *_ in out],
            model_version=",".join(sorted(versions)) or None,
            retries=sum(r for *_, r, _ in out), failures=sum(f for *_, f in out),
            extra={"model_versions_seen": sorted(versions)})

    def _fan_out(self, profile, items):
        cap = self.backend.max_questions or len(items)
        order = list(range(len(items)))
        if self.shuffle_seed is not None:      # order-sensitivity check
            random.Random(self.shuffle_seed).shuffle(order)
        scores = [math.nan] * len(items)
        retries = failures = 0
        versions = set()
        for start in range(0, len(order), cap):
            chunk = order[start:start + cap]
            chunk_items = [items[i] for i in chunk]
            qs = [YesNoQuestion(f"{self.question.id}:{c.item_id}",
                                f"For candidate [{c.item_id}]: {self.question.question}",
                                self.question.true_means, self.question.false_means)
                  for c in chunk_items]
            try:
                reply, r = self._call(build_state(profile, items=chunk_items, labels=self.state_labels), qs)
                retries += r
                versions.add(reply.model_version)
                for i, p in zip(chunk, reply.probs):
                    scores[i] = float(p)
            except Exception as e:  # noqa: BLE001
                self._record_error(e)
                failures += len(chunk)
        versions.discard(None)
        return RerankResult(scores=scores, model_version=",".join(sorted(versions)) or None,
                            retries=retries, failures=failures,
                            extra={"fan_out_order": order})
