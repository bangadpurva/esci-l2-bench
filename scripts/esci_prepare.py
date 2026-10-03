"""ESCI step 1: sample queries, write graded qrels and product text.

    python scripts/esci_prepare.py                  # 1,000 test + 500 validation queries
    python scripts/esci_prepare.py --n-test 8956    # every test query

Outputs in data/esci/processed/: queries_{valid,test}.parquet, qrels_{valid,test}.parquet,
products.parquet (product_id, text for every US product), prepare_report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recl2bench.esci import data as D  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/esci/raw")
    ap.add_argument("--out", default="data/esci/processed")
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--n-valid", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--locale", default="us")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ex = D.load_examples(a.raw, a.locale)
    rep = {"locale": a.locale, "seed": a.seed, "examples": len(ex), "splits": {}}
    for split, source, n in (("test", "test", a.n_test), ("valid", "train", a.n_valid)):
        q = D.sample_queries(ex, source, n, a.seed)
        r = D.qrels(ex, q.query_id)
        q.to_parquet(out / f"queries_{split}.parquet", index=False)
        r.to_parquet(out / f"qrels_{split}.parquet", index=False)
        per_q = r.groupby("query_id")
        rep["splits"][split] = {
            "source_split": source, "queries": len(q), "judged_pairs": len(r),
            "judged_per_query_median": float(per_q.size().median()),
            "exact_per_query_mean": float((r.label == "E").groupby(r.query_id).sum().mean()),
            "label_share": r.label.value_counts(normalize=True).round(4).to_dict(),
        }
    # validation queries must never be used for fine-tuning later
    pd_ids = set(__import__("pandas").read_parquet(out / "queries_valid.parquet").query_id)
    rep["valid_query_ids_reserved"] = len(pd_ids)

    rep["products"] = D.write_products(a.raw, out / "products.parquet", a.locale)
    import pandas as pd
    ids = set(pd.read_parquet(out / "products.parquet", columns=["product_id"]).product_id)
    judged = set(pd.read_parquet(out / "qrels_test.parquet").product_id)
    rep["test_judged_products_in_catalog"] = len(judged & ids) / max(len(judged), 1)
    (out / "prepare_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
