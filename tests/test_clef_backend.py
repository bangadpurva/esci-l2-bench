import pytest

from recl2bench.rerankers.decision import (ClefBackend, DecisionReranker, RetryableError,
                                           YesNoQuestion)

Q = YesNoQuestion("yes_no_v1", "Will they buy it?", "matches", "doesn't")


class FakeHTTP:
    def __init__(self, statuses=(200,), envelope=True, field="noul"):
        self.statuses, self.envelope, self.bodies, self.field = list(statuses), envelope, [], field

    def __call__(self, url, headers, body):
        assert url.endswith("/ai/run/@cf/cloudflare/clef-flash")
        assert headers["Authorization"] == "Bearer tok"
        self.bodies.append(body)
        status = self.statuses.pop(0) if self.statuses else 200
        if status != 200:
            return status, {"error": "x"}
        ans = {k: {"type": "noul", self.field: 0.1 + 0.1 * i} for i, k in enumerate(body["questions"])}
        data = {"model": "clef-flash", "answers": ans, "usage": {"input_tokens": 100}}
        return 200, ({"result": data, "success": True} if self.envelope else data)


def test_request_shape_and_parsing():
    http = FakeHTTP(envelope=False)
    be = ClefBackend("acct", "tok", transport=http)
    reply = be.ask("state text", [Q, Q])
    body = http.bodies[0]
    assert body["model"] == "clef-flash" and body["state"] == "state text"
    assert body["questions"]["q0"]["type"] == "noul"
    assert "Answer true if: matches" in body["questions"]["q0"]["instructions"]
    assert reply.probs == pytest.approx([0.1, 0.2]) and reply.model_version == "clef-flash"
    assert be.usage_input_tokens == 100


def test_cloudflare_envelope_unwrapped():
    assert ClefBackend("acct", "tok", transport=FakeHTTP()).ask("s", [Q]).probs == pytest.approx([0.1])


def test_429_is_retryable_and_400_is_not():
    with pytest.raises(RetryableError):
        ClefBackend("acct", "tok", transport=FakeHTTP([429])).ask("s", [Q])
    with pytest.raises(RuntimeError):
        ClefBackend("acct", "tok", transport=FakeHTTP([400])).ask("s", [Q])


def test_question_cap():
    with pytest.raises(ValueError):
        ClefBackend("acct", "tok", transport=FakeHTTP()).ask("s", [Q] * 65)


def test_per_pair_through_adapter_with_retry(profile, candidates):
    be = ClefBackend("acct", "tok", transport=FakeHTTP([429, 529]))
    res = DecisionReranker(be, Q, "per_pair", concurrency=1, sleep=lambda s: None).rerank(profile, candidates)
    assert res.retries == 2 and res.failures == 0 and res.model_version == "clef-flash"


def test_score_builds_clef_from_its_config(monkeypatch):
    """The path esci_score.py takes: whole YAML config (with its own `name` key) -> backend."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import esci_score as score
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "tok")
    args = type("A", (), {"scoring_mode": None, "seed": 0, "concurrency": None})()
    rr, cfg, tsha, mode, conc = score.build("clef", args, root / "configs", "esci_v1")
    assert rr.backend.name == "clef-flash" and mode == "per_pair" and tsha
    assert rr.backend.url.endswith("/accounts/acct/ai/run/@cf/cloudflare/clef-flash")


def test_parses_live_response_shape_exactly():
    """Verbatim body from the first live call (2026-10-03)."""
    live = {"result": {"model": "clef-flash", "answers": {"q0": {"type": "noul", "noul": 0.6758}},
                       "usage": {"input_tokens": 166, "output_tokens": 0}},
            "success": True, "errors": [], "messages": []}
    be = ClefBackend("acct", "tok", transport=lambda url, h, body: (200, live))
    reply = be.ask("s", [Q])
    assert reply.probs == pytest.approx([0.6758]) and reply.model_version == "clef-flash"
    assert be.usage_input_tokens == 166


def test_docs_field_name_still_accepted():
    assert ClefBackend("acct", "tok", transport=FakeHTTP(field="probability")).ask("s", [Q]).probs == pytest.approx([0.1])


def test_missing_probability_field_is_an_error():
    bad = {"result": {"model": "clef-flash", "answers": {"q0": {"type": "noul"}}}}
    with pytest.raises(ValueError):
        ClefBackend("acct", "tok", transport=lambda u, h, b: (200, bad)).ask("s", [Q])


def test_backoff_waits_grow_and_are_capped(profile, candidates):
    waits = []
    be = ClefBackend("acct", "tok", transport=FakeHTTP([429] * 5))
    rr = DecisionReranker(be, Q, "per_pair", concurrency=1, sleep=waits.append,
                          backoff_base=1.0, backoff_cap=4.0)
    res = rr.rerank(profile, candidates[:1])
    assert res.retries == 5 and res.failures == 0 and len(waits) == 5
    caps = [1, 2, 4, 4, 4]
    assert all(0 <= w <= c for w, c in zip(waits, caps))
    assert rr.retry_wait_s == pytest.approx(sum(waits))


def test_gives_up_after_max_retries(profile, candidates):
    be = ClefBackend("acct", "tok", transport=FakeHTTP([429] * 10))
    rr = DecisionReranker(be, Q, "per_pair", concurrency=1, max_retries=2, sleep=lambda s: None)
    res = rr.rerank(profile, candidates[:1])
    assert res.failures == 1 and "RetryableError" in next(iter(rr.error_samples))


def test_jev_request_shape_and_versioned_model():
    from recl2bench.rerankers.decision import JevBackend
    seen = {}

    def http(url, headers, body):
        seen.update(url=url, auth=headers["Authorization"], body=body)
        return 200, {"model": "jev-1.13.0", "answers": {"q0": {"type": "noul", "noul": 0.42}},
                     "usage": {"input_tokens": 77}}
    be = JevBackend("key", transport=http)
    r = be.ask("state", [Q])
    assert seen["url"] == "https://api.typesafe.ai/v1/systemone" and seen["auth"] == "Bearer key"
    assert seen["body"]["model"] == "jev-1.13.0" and seen["body"]["questions"]["q0"]["type"] == "noul"
    assert r.probs == pytest.approx([0.42]) and r.model_version == "jev-1.13.0"
    assert be.usage_input_tokens == 77
    with pytest.raises(ValueError):
        JevBackend("key", model="jev-latest")


def test_score_builds_jev_from_its_config(monkeypatch):
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import esci_score as score
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    args = type("A", (), {"scoring_mode": None, "seed": 0, "concurrency": None})()
    rr, cfg, tsha, mode, conc = score.build("jev", args, root / "configs", "esci_v1")
    assert rr.backend.name == "jev-1.13.0" and conc == 32 and cfg["price_per_m_input"] == 0.042


def test_jev_parses_live_response_exactly():
    """Verbatim body from the first live Jev call (2026-10-03)."""
    from recl2bench.rerankers.decision import JevBackend
    live = {"model": "jev-1.13.0", "answers": {"q0": {"type": "noul", "noul": 0.67}},
            "usage": {"input_tokens": 295, "output_tokens": 21}}
    be = JevBackend("key", transport=lambda u, h, b: (200, live))
    r = be.ask("s", [Q])
    assert r.probs == pytest.approx([0.67]) and r.model_version == "jev-1.13.0"
    assert be.usage_input_tokens == 295


def test_network_timeouts_are_retryable(monkeypatch, profile, candidates):
    import urllib.error
    import urllib.request
    from recl2bench.rerankers.decision import JevBackend
    calls = {"n": 0}

    class Resp:
        status = 200
        def read(self):
            return b'{"model": "jev-1.13.0", "answers": {"q0": {"type": "noul", "noul": 0.5}}}'
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=None, context=None):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise urllib.error.URLError(TimeoutError("The handshake operation timed out"))
        return Resp()
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    rr = DecisionReranker(JevBackend("k"), Q, "per_pair", concurrency=1, sleep=lambda s: None)
    res = rr.rerank(profile, candidates[:1])
    assert res.failures == 0 and res.retries == 2 and res.scores == [0.5]
