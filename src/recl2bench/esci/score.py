"""ESCI scoring helpers: labels, budgeted L2 text, graded oracle, results table."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..eval.bootstrap import bootstrap_p_value, compare_to_baseline, holm, paired_bootstrap
from ..rerankers.base import Reranker, RerankResult
from ..rerankers.text import cut
from .data import GAINS
from .metrics import COLUMNS, PRIMARY


def labels_by_query(qrels: pd.DataFrame) -> dict[str, dict[str, str]]:
    return {str(q): dict(zip(g.product_id, g.label)) for q, g in qrels.groupby("query_id")}


def budget_texts(texts: dict[str, str], tok, max_tokens: int) -> tuple[dict[str, str], int]:
    out, truncated = {}, 0
    for k, t in texts.items():
        c = cut(t, tok, max_tokens)
        out[k] = c.text
        truncated += c.truncated
    return out, truncated


class GradedOracle(Reranker):
    """Ceiling: orders by ESCI gain, then L1 order. Uses labels."""
    name = "oracle"

    def __init__(self, labels: dict[str, dict[str, str]]):
        self.labels = labels

    def _score(self, user_profile, candidate_items):
        lab = self.labels.get(user_profile.user_id, {})
        return RerankResult(scores=[float(GAINS.get(lab.get(c.item_id), 0.0)) for c in candidate_items],
                            model_version="oracle")


NO_TEST = {"l1_order", "oracle"}


def load_runs(runs_dir: Path, split: str, setting: str) -> dict[str, dict]:
    """Latest complete run per model for (split, setting)."""
    best: dict[str, dict] = {}
    for d in sorted(Path(runs_dir).glob("*/")):
        mf, pu = d / "manifest.json", d / "per_user.parquet"
        if not (mf.exists() and pu.exists()):
            continue
        m = json.loads(mf.read_text())
        u = pd.read_parquet(pu)
        if (m.get("setting") != setting or u["split"].iloc[0] != split
                or int(u.get("limit_users", pd.Series([0])).iloc[0] or 0)):
            continue
        if m["model"] not in best or m["started_at"] > best[m["model"]]["manifest"]["started_at"]:
            best[m["model"]] = {"manifest": m, "per_user": u}
    return best


def build_report(runs: dict[str, dict], split: str, setting: str, n_resamples: int = 10_000,
                 seed: int = 0, name: str = "ESCI") -> tuple[pd.DataFrame, str]:
    if "l1_order" not in runs:
        raise ValueError("score l1_order first; every comparison is against it")
    pools = {r["manifest"]["pool_sha256"] for r in runs.values()}
    if len(pools) != 1:
        raise ValueError(f"runs use different pools: {pools}")
    queries = sorted(runs["l1_order"]["per_user"].user_id)
    aligned = {}
    for m, r in runs.items():
        pu = r["per_user"].set_index("user_id")
        missing = set(queries) - set(pu.index)
        if missing:
            raise ValueError(f"{m} is missing {len(missing)} queries")
        aligned[m] = pu.loc[queries]
    base = aligned["l1_order"][PRIMARY].to_numpy()
    tested = {m: a[PRIMARY].to_numpy() for m, a in aligned.items() if m not in NO_TEST}
    stats = {s.name: s for s in compare_to_baseline(tested, base, n_resamples=n_resamples, seed=seed)} \
        if tested else {}
    rows = []
    for m, a in aligned.items():
        mf = runs[m]["manifest"]
        cost = mf.get("cost_usd")
        row = {"model": m, "version": mf.get("model_version"), **{c: a[c].mean() for c in COLUMNS},
               "latency_p50_s": a.latency_s.quantile(.5), "latency_p95_s": a.latency_s.quantile(.95),
               "usd_per_1k_queries": (cost / len(queries) * 1000) if cost is not None else np.nan,
               "failures": mf["failures"]}
        s = stats.get(m)
        if s:
            row.update({"d_p@10_vs_l1": s.mean_diff, "ci_low": s.ci_low, "ci_high": s.ci_high,
                        "p_holm": s.p_holm, "beats_l1": s.significant})
        rows.append(row)
    t = pd.DataFrame(rows).sort_values(PRIMARY, ascending=False).reset_index(drop=True)

    lines = [f"# {name} results: {split}, {setting} lists, {len(queries)} queries", "",
             f"Pool sha256 `{pools.pop()[:16]}…` · primary P@10 (relevant = Exact; unjudged = not relevant) · "
             f"paired bootstrap ({n_resamples:,}) vs L1 order, Holm-corrected", "",
             "| Model | P@10 | Δ vs L1 [CI] | p (Holm) | Beats L1 | P@5 | P@10 E+S | P@10 judged-only | judged@10 | NDCG@10 | p50 / p95 s | $ / 1K q |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in t.iterrows():
        if pd.notna(r.get("d_p@10_vs_l1", np.nan)):
            d = f"{r['d_p@10_vs_l1']:+.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}]"
            p, b = f"{r.p_holm:.4f}", ("yes" if r.beats_l1 else "no")
        else:
            d = p = b = "—"
        usd = "local" if pd.isna(r.usd_per_1k_queries) else f"{r.usd_per_1k_queries:.2f}"
        lines.append(f"| {r.model} | {r['p@10']:.4f} | {d} | {p} | {b} | {r['p@5']:.4f} | {r['p@10_es']:.4f} | "
                     f"{r['p@10_judged']:.4f} | {r['judged@10']:.3f} | {r['ndcg@10']:.4f} | "
                     f"{r.latency_p50_s:.3f} / {r.latency_p95_s:.3f} | {usd} |")
    lines += ["", "Oracle orders by ESCI gain (uses labels); L1 order is the baseline. "
              "Neither is in the Holm family. Low judged@10 means many top-10 items carry no label; "
              "compare P@10 with the judged-only column before drawing conclusions."]
    lines += head_to_head(aligned, n_resamples, seed)
    return t, "\n".join(lines) + "\n"


def head_to_head(aligned: dict[str, pd.DataFrame], n_resamples: int, seed: int,
                 metrics=("p@10", "p@5")) -> list[str]:
    """Paired bootstrap between every pair of rerankers (exploratory, added after results)."""
    models = sorted(m for m in aligned if m not in NO_TEST and m != "random")
    pairs = [(a, b) for i, a in enumerate(models) for b in models[i + 1:]]
    if not pairs:
        return []
    rows = []
    for j, (a, b) in enumerate(pairs):
        for k, met in enumerate(metrics):
            x, y = aligned[a][met].to_numpy(), aligned[b][met].to_numpy()
            if x.mean() < y.mean():
                a_, b_, x, y = b, a, y, x
            else:
                a_, b_ = a, b
            mean, boot = paired_bootstrap(x, y, n_resamples, seed + 100 + 10 * j + k)
            lo, hi = np.quantile(boot, [0.025, 0.975])
            rows.append([f"{a_} vs {b_}", met, mean, lo, hi, bootstrap_p_value(boot)])
    for r, p in zip(rows, holm([r[5] for r in rows])):
        r.append(p)
    out = ["", "## Head to head (exploratory)", "",
           f"Paired bootstrap ({n_resamples:,}) of the first model minus the second; 95% percentile CI; "
           f"Holm across all {len(rows)} comparisons in this table.", "",
           "| Comparison | Metric | Δ [95% CI] | p (Holm) | Significant |", "|---|---|---|---|---|"]
    for name, met, mean, lo, hi, _, ph in rows:
        out.append(f"| {name} | {met.upper()} | {mean:+.4f} [{lo:+.4f}, {hi:+.4f}] | {ph:.4f} | "
                   f"{'yes' if ph < 0.05 else 'no'} |")
    return out
