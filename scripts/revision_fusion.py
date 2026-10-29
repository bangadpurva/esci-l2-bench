"""Score fusion of an L2 model with the L1 embedding similarity (Amendment 5, offline, no model calls).

    python scripts/revision_fusion.py --models jev qwen3 --out paper_generated

For each setting (ESCI judged, ESCI retrieved, WANDS) and model:
  1. z-score fusion  alpha * z(L2) + (1 - alpha) * z(L1 similarity), z within each query's candidates.
     alpha is chosen on the VALIDATION queries (grid 0, 0.1, ..., 1; maximise P@10; ties -> larger alpha)
     and then applied unchanged to the test queries.
  2. reciprocal rank fusion (k = 60, Cormack et al. 2009), nothing tuned.
Needs validation and test runs (saved scores) of the model in the same setting, plus the L1 order run.
Reports P@10, P@5, NDCG@10 and paired-bootstrap differences (Holm across the comparisons per setting).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from recl2bench import revision as R  # noqa: E402
from revision_tables import MISSING, NAMES, NUMBERS, SETTINGS, Ctx, f3, fd, write  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["jev", "qwen3"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="paper_generated")
    ap.add_argument("--resamples", type=int, default=10_000)
    a = ap.parse_args(argv)
    root, out = Path(a.root), Path(a.out)
    rows, drows, chosen = [], [], {}
    for lab, ds, setting in SETTINGS:
        try:
            test, valid = Ctx(root, ds, setting, "test"), Ctx(root, ds, setting, "valid")
            if not (test.has("l1_order") and any(test.has(m) and valid.has(m) for m in a.models)):
                print(f"skip {lab}: no test+valid runs yet")
                continue
            t_pools, v_pools = test.pools(), valid.pools()
        except FileNotFoundError as e:
            print(f"skip {lab}: {e.filename}")
            continue
        for m in a.models:
            if not (test.has(m) and valid.has(m) and test.has("l1_order")):
                print(f"skip {lab}/{m}: need test+valid runs of {m} and the l1_order baseline")
                rows.append([lab, NAMES[m]] + [MISSING] * 6)
                continue
            ts = pd.read_parquet(test.runs[m]["dir"] / "scores.parquet")
            vs = pd.read_parquet(valid.runs[m]["dir"] / "scores.parquet")
            alpha, grid = R.select_alpha(vs, v_pools, valid.labels)
            chosen[f"{lab}/{m}"] = {"alpha": alpha, "valid_p@10_by_alpha": grid}
            ev = {"l1": R.evaluate_orders(R.fuse_z(ts, t_pools, 0.0), test.labels),
                  "l2": R.evaluate_orders(R.fuse_z(ts, t_pools, 1.0), test.labels),
                  "z": R.evaluate_orders(R.fuse_z(ts, t_pools, alpha), test.labels),
                  "rrf": R.evaluate_orders(R.fuse_rrf(ts), test.labels)}
            cmp_ = [("z-vs-l2", ev["z"], ev["l2"]), ("z-vs-l1", ev["z"], ev["l1"]),
                    ("rrf-vs-l2", ev["rrf"], ev["l2"]), ("rrf-vs-l1", ev["rrf"], ev["l1"])]
            res = R.diff_table([(n, x["p@10"].to_numpy(), y["p@10"].to_numpy()) for n, x, y in cmp_], a.resamples)
            for r in res:
                NUMBERS[f"fusion[{lab}][{m}][{r['name']}]"] = r
            rows.append([lab, NAMES[m], f3(ev["l1"]["p@10"].mean()), f3(ev["l2"]["p@10"].mean()),
                         f"{f3(ev['z']['p@10'].mean())} ($\\alpha$={alpha:g})", f3(ev["rrf"]["p@10"].mean()),
                         f3(ev["z"]["ndcg@10"].mean()), f3(ev["l2"]["ndcg@10"].mean())])
            drows.append([lab, NAMES[m]] + [fd(r) for r in res])
    if not rows:
        raise SystemExit("no fusion rows: run validation + test for the models first")
    write(out, "tab_fusion", ["Setting", "Model", "Embedding", "L2 alone", "z-fusion", "RRF",
                              "NDCG fusion", "NDCG L2"], rows, "@{}llcccccc")
    if drows:
        write(out, "tab_fusion_delta", ["Setting", "Model", "z-fusion $-$ L2", "z-fusion $-$ emb.",
                                        "RRF $-$ L2", "RRF $-$ emb."], drows, "@{}llcccc")
    out.mkdir(parents=True, exist_ok=True)
    (out / "fusion_alpha.json").write_text(json.dumps(chosen, indent=1))
    nf = out / "numbers.json"
    old = json.loads(nf.read_text()) if nf.exists() else {}
    old.update(NUMBERS)
    nf.write_text(json.dumps(old, indent=1, default=float))


if __name__ == "__main__":
    main()
