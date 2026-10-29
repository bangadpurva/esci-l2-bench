"""Cost-comparison sensitivity (Amendment 5; answers the editor's item 7).

No model calls and no new runs: it recomputes the per-1,000-request cost from the measured latencies
and billed tokens already in the paper, first reproducing the reported numbers, then varying each
assumption on its own and showing what happens to the Jev / Qwen3-Reranker cost ratio.

    python scripts/revision_cost.py --out paper_generated

Inputs (paper Table "Latency per request", Section "Latency and Cost"; override with flags):
  Qwen3-Reranker-0.6B p50/p95 s per 100-candidate request: ESCI 1.21/1.23, WANDS 1.58/1.70
  CLM-v0.1-8B                                           : ESCI 3.82/5.27, WANDS 4.11/5.59
  Jev input tokens per call 570 (ESCI) / 650 (WANDS), 100 calls per request, $0.042 per million
  input tokens, ~21 output tokens per call (output tokens are not billed by the vendor).
  GPU priced at AWS g6e.xlarge on-demand, $1.861 per hour.
"""
from __future__ import annotations

import argparse
from pathlib import Path

P50 = {"qwen3": {"ESCI": 1.21, "WANDS": 1.58}, "clm": {"ESCI": 3.82, "WANDS": 4.11}}
P95 = {"qwen3": {"ESCI": 1.23, "WANDS": 1.70}, "clm": {"ESCI": 5.27, "WANDS": 5.59}}
# Mean billed input tokens per call, backed out of the reported Jev costs (billed tokens; the paper's
# "about 570 / 650" are these rounded): cost per 1K requests = tokens * 100 calls * $0.042/M * 1000.
TOK = {"ESCI": 2.38 / (100 * 0.042e-6 * 1000), "WANDS": 2.75 / (100 * 0.042e-6 * 1000)}
OUT_TOK = 21
CALLS = 100
PRICE_IN = 0.042          # USD per million input tokens
RATE = 1.861              # USD per GPU hour, g6e.xlarge on demand
RESERVED_1Y = 1.1724      # g6e.xlarge, 1-year reserved, no upfront (aggregator snapshot, see paper)
SPOT = 0.8667             # lowest-AZ spot at one snapshot; spot prices move


def local_cost(secs: float, rate: float = RATE, util: float = 1.0, speedup: float = 1.0) -> float:
    """USD per 1,000 requests: busy GPU seconds per request x rate, divided by utilisation and speed-up."""
    return secs / speedup * rate / 3600 * 1000 / util


def jev_cost(ds: str, count_output: bool = False) -> float:
    toks = TOK[ds] + (OUT_TOK if count_output else 0)
    return toks * CALLS * PRICE_IN / 1e6 * 1000


def scenarios():
    """(label, jev cost fn(ds), qwen3 cost fn(ds))"""
    yield "Reported (p50 time, on-demand \\$1.861/h, 100\\% busy)", lambda d: jev_cost(d), lambda d: local_cost(P50["qwen3"][d])
    yield "Jev output tokens billed at the input rate", lambda d: jev_cost(d, True), lambda d: local_cost(P50["qwen3"][d])
    yield "p95 instead of p50 request time", lambda d: jev_cost(d), lambda d: local_cost(P95["qwen3"][d])
    yield "1-year reserved GPU (\\$1.172/h)", lambda d: jev_cost(d), lambda d: local_cost(P50["qwen3"][d], RESERVED_1Y)
    yield "Spot GPU (\\$0.867/h)", lambda d: jev_cost(d), lambda d: local_cost(P50["qwen3"][d], SPOT)
    yield "GPU 1.5$\\times$ faster than the A40", lambda d: jev_cost(d), lambda d: local_cost(P50["qwen3"][d], speedup=1.5)
    yield "GPU 2$\\times$ faster than the A40", lambda d: jev_cost(d), lambda d: local_cost(P50["qwen3"][d], speedup=2.0)
    yield "GPU busy 50\\% of the time", lambda d: jev_cost(d), lambda d: local_cost(P50["qwen3"][d], util=0.5)
    yield "GPU busy 25\\% of the time", lambda d: jev_cost(d), lambda d: local_cost(P50["qwen3"][d], util=0.25)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper_generated")
    a = ap.parse_args(argv)
    # reproduce the reported numbers first
    rep = {"qwen3": (0.63, 0.82), "clm": (1.97, 2.12), "jev": (2.38, 2.75)}
    got = {"qwen3": tuple(round(local_cost(P50["qwen3"][d]), 2) for d in ("ESCI", "WANDS")),
           "clm": tuple(round(local_cost(P50["clm"][d]), 2) for d in ("ESCI", "WANDS")),
           "jev": tuple(round(jev_cost(d), 2) for d in ("ESCI", "WANDS"))}
    for k in rep:
        assert all(abs(x - y) <= 0.0051 for x, y in zip(got[k], rep[k])), (k, got[k], rep[k])
    print("reproduced reported costs:", got)
    rows = []
    for label, jf, qf in scenarios():
        cells = []
        for d in ("ESCI", "WANDS"):
            j, q = jf(d), qf(d)
            cells += [f"{q:.2f}", f"{j:.2f}", f"{j / q:.1f}"]
        rows.append(" & ".join([label] + cells) + " \\\\")
    tab = "\n".join(["\\begin{tabular}{@{}lcccccc@{}}", "\\toprule",
                     " & \\multicolumn{3}{c}{ESCI} & \\multicolumn{3}{c}{WANDS} \\\\",
                     "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
                     "Assumption changed (one at a time) & Qwen3 & Jev & Ratio & Qwen3 & Jev & Ratio \\\\",
                     "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", ""])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tab_cost_sensitivity.tex").write_text(tab)
    print(tab)
    # the break-even utilisation, where both cost the same
    for d in ("ESCI", "WANDS"):
        u = local_cost(P50["qwen3"][d]) / jev_cost(d)
        print(f"{d}: Qwen3 on the GPU costs the same as Jev at {u:.0%} busy time")


if __name__ == "__main__":
    main()
