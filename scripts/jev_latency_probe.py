"""Jev latency probe: per-call latency and per-request (100 candidates) wall time, client effects separated.

    python scripts/jev_latency_probe.py                # needs TYPESAFE_API_KEY (from .env)
    python scripts/jev_latency_probe.py --requests 10  # per concurrency level

The benchmark client opened a new HTTPS connection (and SSL context) for every call and reached
about 12 calls/s at any concurrency, well under Jev's documented 80 requests/s. This probe
measures, on the same state/question shape as the benchmark (ESCI search framing, ~570 input
tokens per call):

  1. cold   - the benchmark client as run (JevBackend.ask), sequential calls
  2. warm   - one kept-alive connection, shared SSL context, sequential calls
  3. burst  - one request = 100 calls in parallel on kept-alive connections, at each concurrency

Scores are not used; nothing here touches test data. Cost at the defaults: ~2,100 calls, ~1.2M
input tokens, about $0.05. Output: results/jev_latency_probe.json and a printed summary.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recl2bench.rerankers.decision import JevBackend, SystemOneBackend, YesNoQuestion, _ssl_context  # noqa: E402

QUERIES = ["t-shirt design", "usb c charger 65w", "queen size bed frame without headboard",
           "waterproof hiking boots women size 8", "stainless steel water bottle 32 oz"]
PRODUCT = ("Example product title with brand and model number\nBrand: Example\n"
           "Bullets: " + "durable material, easy to clean, fits most standard sizes, " * 18 +
           "\nDescription: " + "A versatile everyday item designed for comfort and long use. " * 14)


def load_env():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def summary(xs):
    return {"n": len(xs), "p50_ms": round(1000 * pct(xs, .5)), "p95_ms": round(1000 * pct(xs, .95)),
            "mean_ms": round(1000 * statistics.mean(xs))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="jev-1.13.0")
    ap.add_argument("--sequential", type=int, default=40)
    ap.add_argument("--requests", type=int, default=10, help="100-call requests per concurrency")
    ap.add_argument("--candidates", type=int, default=100)
    ap.add_argument("--concurrency", default="32,80")
    a = ap.parse_args(argv)
    load_env()
    key = os.environ["TYPESAFE_API_KEY"]
    t = yaml.safe_load((ROOT / "configs/templates/esci_v1.yaml").read_text())
    q = YesNoQuestion("esci_v1", t["question"], t["true_means"], t["false_means"])
    head, cand = t.get("state_labels", ["Search query", "Candidate product"])

    def state(i):
        return f"{head}\n{QUERIES[i % len(QUERIES)]}\n\n{cand}\n{PRODUCT} #{i}\n"

    body_of = lambda i: json.dumps({"model": a.model, "state": state(i), "questions": {
        "q0": {"type": "noul", "instructions": SystemOneBackend.instructions(q)}}}).encode()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    host, path = "api.typesafe.ai", "/v1/systemone"
    ctx = _ssl_context()
    local = threading.local()
    statuses: dict[int, int] = {}
    lock = threading.Lock()

    def warm_call(i):
        c = getattr(local, "c", None)
        if c is None:
            c = local.c = http.client.HTTPSConnection(host, context=ctx, timeout=60)
        t0 = time.perf_counter()
        try:
            c.request("POST", path, body=body_of(i), headers=headers)
            r = c.getresponse()
            r.read()
            st = r.status
        except (http.client.HTTPException, OSError):
            local.c = None
            st = -1
        dt = time.perf_counter() - t0
        with lock:
            statuses[st] = statuses.get(st, 0) + 1
        return dt

    out = {"model": a.model, "host": host, "client_note": "warm = kept-alive HTTPS, shared SSL context"}

    # 1. cold: the benchmark client exactly as run
    be = JevBackend(key, a.model)
    cold = []
    for i in range(a.sequential):
        t0 = time.perf_counter()
        be.ask(state(i), [q])
        cold.append(time.perf_counter() - t0)
    out["cold_sequential_per_call"] = summary(cold)

    # 2. warm: one kept-alive connection
    warm_call(0)                                   # open the connection (not timed)
    warm = [warm_call(i) for i in range(1, a.sequential + 1)]
    out["warm_sequential_per_call"] = summary(warm)

    # 3. burst: one request = `candidates` calls in parallel
    out["burst"] = {}
    for conc in [int(x) for x in a.concurrency.split(",")]:
        with ThreadPoolExecutor(conc) as ex:
            list(ex.map(warm_call, range(conc)))   # open `conc` connections (not timed)
            walls, calls = [], []
            for r in range(a.requests):
                t0 = time.perf_counter()
                lat = list(ex.map(warm_call, range(r * a.candidates, (r + 1) * a.candidates)))
                walls.append(time.perf_counter() - t0)
                calls += lat
        w = summary(walls)
        out["burst"][str(conc)] = {"per_request_s": {"n": w["n"], "p50": w["p50_ms"] / 1000,
                                                     "p95": w["p95_ms"] / 1000},
                                   "per_call": summary(calls),
                                   "calls_per_s": round(a.candidates * a.requests / sum(walls), 1)}
    out["http_status_counts"] = statuses
    dest = ROOT / "results/jev_latency_probe.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
