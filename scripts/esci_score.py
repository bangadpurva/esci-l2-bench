"""ESCI step 3: score one model on frozen lists, or build the results table.

    python scripts/esci_score.py --model l1_order --split test --setting retrieved
    python scripts/esci_score.py --model qwen3 --split test --setting judged
    python scripts/esci_score.py --report --split test --setting retrieved

Settings: `retrieved` = L1 top-100 from the whole catalog (end-to-end, incomplete labels);
`judged` = each query's ESCI-judged products (every candidate labelled).
"""
from __future__ import annotations

import argparse
import hashlib
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recl2bench.esci import metrics as EM  # noqa: E402
from recl2bench.esci import score as ES  # noqa: E402
from recl2bench import pools as P  # noqa: E402
from recl2bench.manifest import RunManifest  # noqa: E402
from recl2bench.rerankers.base import UserProfile  # noqa: E402
from recl2bench.runner import run  # noqa: E402
from recl2bench.tokenizer import get_tokenizer  # noqa: E402

MODELS = ["l1_order", "random", "oracle", "qwen3", "clef", "jev", "clm"]


def build(model: str, a, cfg_dir: Path, template: str):
    """Return (reranker, model_cfg, template_sha, scoring_mode, concurrency)."""
    from recl2bench.rerankers.decision import YesNoQuestion
    tpath = cfg_dir / "templates" / f"{template}.yaml"
    t = yaml.safe_load(tpath.read_text())
    tsha = hashlib.sha256(tpath.read_bytes()).hexdigest()
    mcfg = lambda m: yaml.safe_load((cfg_dir / f"models/{m}.yaml").read_text())  # noqa: E731
    if model == "l1_order":
        from recl2bench.rerankers.dummy import L1OrderReranker
        return L1OrderReranker(), {}, None, None, None
    if model == "random":
        from recl2bench.rerankers.dummy import RandomReranker
        return RandomReranker(a.seed), {}, None, None, None
    if model == "qwen3":
        from recl2bench.rerankers.cross_encoder import Qwen3Reranker
        c = mcfg("qwen3_reranker")
        return (Qwen3Reranker(t["instruction"], c["hf_model"], c["revision"], max_length=c["max_length"],
                              batch_size=c["batch_size"]), c, tsha, None, None)
    if model in ("clef", "jev"):
        from recl2bench.rerankers.decision import DecisionReranker, make_backend
        c = mcfg(model)
        q = YesNoQuestion(t["id"], t["question"], t["true_means"], t["false_means"])
        mode = a.scoring_mode or c["scoring_mode"]
        conc = a.concurrency or c["concurrency"]
        return (DecisionReranker(make_backend(c["backend"], **c), q, mode, concurrency=conc,
                                 state_labels=tuple(t["state_labels"])), c, tsha, mode, conc)
    if model == "clm":
        from recl2bench.rerankers.clm import CLMReranker
        c = mcfg("clm")
        return (CLMReranker(c["emb_url"], engine_kwargs=c.get("engine_kwargs") or {},
                            model_version=c["model_version"], instructions=t["question"]),
                c, tsha, None, None)
    raise ValueError(model)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--setting", default="retrieved", choices=["retrieved", "judged"])
    ap.add_argument("--limit-users", type=int, help="smoke test on the first N queries")
    ap.add_argument("--concurrency", type=int)
    ap.add_argument("--scoring-mode", choices=["per_pair", "fan_out"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", help="'whitespace' for tests")
    ap.add_argument("--config", default="configs/esci.yaml")
    ap.add_argument("--configs", default="configs")
    ap.add_argument("--proc", default="data/esci/processed")
    ap.add_argument("--pools-dir", default="data/esci/pools")
    ap.add_argument("--runs-dir", default="runs/esci")
    ap.add_argument("--results-dir", default="results/esci")
    ap.add_argument("--hardware", default=platform.platform())
    ap.add_argument("--resamples", type=int, default=10_000)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())

    if a.report:
        runs = ES.load_runs(Path(a.runs_dir), a.split, a.setting)
        table, md = ES.build_report(runs, a.split, a.setting, a.resamples)
        out = Path(a.results_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = out / f"{a.split}_{a.setting}"
        table.to_csv(stem.with_suffix(".csv"), index=False)
        stem.with_suffix(".md").write_text(md)
        print(md)
        return stem
    if not a.model:
        ap.error("--model or --report is required")

    proc, pdir = Path(a.proc), Path(a.pools_dir)
    pools, pool_sha = P.load(pdir / f"pools_{a.split}_{a.setting}.parquet")
    keep, cohort_sha = P.load_eval_cohort(pdir / f"eval_users_{a.split}_{a.setting}.parquet")
    pools = pools[pools.user_id.isin(set(keep))]
    users = sorted(pools.user_id.unique())
    if a.limit_users:
        users = users[:a.limit_users]
        pools = pools[pools.user_id.isin(set(users))]

    queries = pd.read_parquet(proc / f"queries_{a.split}.parquet")
    labels = ES.labels_by_query(pd.read_parquet(proc / f"qrels_{a.split}.parquet"))
    l2 = cfg["l2"]
    tok = get_tokenizer(a.tokenizer or cfg["l1"]["encoder"], None if a.tokenizer else cfg["l1"]["revision"])
    need = set(pools.parent_asin)
    prods = pd.read_parquet(proc / "products.parquet")
    prods = prods[prods.product_id.isin(need)]
    item_text, n_trunc = ES.budget_texts(dict(zip(prods.product_id, prods.text)), tok, l2["item_tokens"])
    qtext, _ = ES.budget_texts({str(k): v for k, v in zip(queries.query_id, queries["query"])}, tok,
                               l2["query_tokens"])
    profiles = {u: UserProfile(u, "", (), (), qtext[u]) for u in users}

    if a.model == "oracle":
        reranker, mcfg, tsha, mode, conc = ES.GradedOracle(labels), {}, None, None, None
    else:
        reranker, mcfg, tsha, mode, conc = build(a.model, a, Path(a.configs), l2["template"])
        if hasattr(reranker, "device"):
            a.hardware = f"{a.hardware} [{reranker.device}]"
            print(f"{a.model}: device={reranker.device}", flush=True)

    started = datetime.now(timezone.utc)
    per_user, scores, totals = run(reranker, pools, profiles, item_text, {},
                                   metric_fn=lambda ranked, u: EM.query_metrics(ranked, labels.get(u, {})))
    tokens = getattr(getattr(reranker, "backend", None), "usage_input_tokens", None)
    price = float((mcfg or {}).get("price_per_m_input", 0) or 0)
    cost = tokens * price / 1e6 if tokens is not None and price else None
    run_id = (f"{started:%Y%m%dT%H%M%S}_A_{a.model}_{a.split}_{a.setting}"
              + (f"_n{a.limit_users}" if a.limit_users else ""))
    out = Path(a.runs_dir) / run_id
    RunManifest(run_id=run_id, track="A", model=a.model,
                model_version=",".join(sorted(totals["versions"])) or (mcfg or {}).get("model_version"),
                scoring_mode=mode, pool_sha256=pool_sha, template_sha256=tsha,
                budget_tokens=l2["item_tokens"], hardware_or_region=a.hardware, concurrency=conc,
                truncated_pairs=n_trunc, retries=totals["retries"], failures=totals["failures"],
                started_at=started.isoformat(), model_versions_seen=sorted(totals["versions"]),
                eval_cohort=a.setting, eval_cohort_sha256=cohort_sha, n_users_scored=len(per_user),
                n_users_full_cohort=len(keep), setting=a.setting, cost_usd=cost, input_tokens=tokens,
                retry_wait_s=getattr(reranker, "retry_wait_s", None)).write(a.runs_dir)
    per_user.assign(split=a.split, setting=a.setting, limit_users=a.limit_users or 0
                    ).to_parquet(out / "per_user.parquet", index=False)
    scores.to_parquet(out / "scores.parquet", index=False)
    errs = getattr(reranker, "error_samples", {})
    if errs:
        print("backend errors (count: message):")
        for k_, v in sorted(errs.items(), key=lambda kv: -kv[1])[:5]:
            print(f"  {v}: {k_}")
    if tokens is not None:
        print(f"input tokens billed: {tokens:,}" + (f" (~${cost:.2f})" if cost is not None else ""))
    print(f"{run_id}: queries={len(per_user)} P@10={per_user['p@10'].mean():.4f} "
          f"P@5={per_user['p@5'].mean():.4f} NDCG@10={per_user['ndcg@10'].mean():.4f} "
          f"judged@10={per_user['judged@10'].mean():.3f} failures={totals['failures']} "
          f"p50/p95={per_user.latency_s.quantile(.5):.3f}/{per_user.latency_s.quantile(.95):.3f}s")
    return out


if __name__ == "__main__":
    main()
