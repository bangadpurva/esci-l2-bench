"""Diagnostic for CLM on validation queries (never test): is a near-random score real or wiring?

    python scripts/clm_probe.py --n 100

Compares, on the same judged validation lists: L1 order, CLM as pre-registered (reference head,
esci_v1 question as instructions), CLM without instructions, and the raw-encoder ablation
(clm-raw). Prints P@10 / P@5 / NDCG@10, the rank correlation with L1, and one ranked example.
These variants are diagnostics only; the test result stays the pre-registered configuration.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recl2bench import pools as P  # noqa: E402
from recl2bench.esci import metrics as EM  # noqa: E402
from recl2bench.esci import score as ES  # noqa: E402
from recl2bench.tokenizer import get_tokenizer  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--emb-url", default="http://127.0.0.1:8090/v1/embeddings")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path("configs/esci.yaml").read_text())
    question = yaml.safe_load(Path(f"configs/templates/{cfg['l2']['template']}.yaml").read_text())["question"]
    proc = Path("data/esci/processed")
    pools, _ = P.load("data/esci/pools/pools_valid_judged.parquet")
    users = sorted(pools.user_id.unique())[:a.n]
    labels = ES.labels_by_query(pd.read_parquet(proc / "qrels_valid.parquet"))
    q = pd.read_parquet(proc / "queries_valid.parquet")
    tok = get_tokenizer(cfg["l1"]["encoder"], cfg["l1"]["revision"])
    prods = pd.read_parquet(proc / "products.parquet")
    prods = prods[prods.product_id.isin(set(pools.parent_asin))]
    text, _ = ES.budget_texts(dict(zip(prods.product_id, prods.text)), tok, cfg["l2"]["item_tokens"])
    qtext, _ = ES.budget_texts({str(k): v for k, v in zip(q.query_id, q["query"])}, tok, cfg["l2"]["query_tokens"])

    from clm import Engine
    eng = Engine(emb_url=a.emb_url)
    print("models:", [m["name"] for m in eng.models()])
    variants = {"clm (pre-registered)": ("clm-latest", question),
                "clm, no instructions": ("clm-latest", None),
                "clm-raw (no head)": ("clm-raw", question)}
    rows, example = {k: [] for k in ["l1_order", *variants]}, None
    for u in users:
        g = pools[pools.user_id == u].sort_values("rank")
        ids = g.parent_asin.tolist()
        lab = labels.get(u, {})
        rows["l1_order"].append(EM.query_metrics(ids, lab) | {"rho": 1.0})
        for name, (model, instr) in variants.items():
            out = eng.rank(qtext[u], [text[i] for i in ids], instructions=instr, model=model)
            prob = {o["candidate"]: o["prob"] for o in out}
            s = np.array([prob[text[i]] for i in ids])
            ranked = [ids[j] for j in np.argsort(-s, kind="stable")]
            rho = pd.Series(s).corr(pd.Series(-np.arange(len(s))), method="spearman")
            rows[name].append(EM.query_metrics(ranked, lab) | {"rho": rho})
            if example is None and name == "clm (pre-registered)":
                example = (qtext[u], [(lab.get(i, "-"), round(float(s[ids.index(i)]), 4), text[i][:70])
                                      for i in ranked[:5]])
    print(f"\nvalidation, judged lists, {len(users)} queries")
    print(f"{'variant':24} {'P@10':>6} {'P@5':>6} {'NDCG@10':>8} {'rho vs L1':>9}")
    for k, v in rows.items():
        d = pd.DataFrame(v)
        print(f"{k:24} {d['p@10'].mean():6.3f} {d['p@5'].mean():6.3f} {d['ndcg@10'].mean():8.3f} {d['rho'].mean():9.2f}")
    print(f"\nexample query: {example[0]!r}  (label, prob, text)")
    for r in example[1]:
        print("  ", r)


if __name__ == "__main__":
    main()
