"""WANDS (Wayfair ANnotation DataSet) in the shared ESCI layout.

Writes data/wands/processed/ with the same files the ESCI pipeline reads, so L1, freezing
and scoring are reused unchanged:
  queries_{valid,test}.parquet  query_id (int), query
  qrels_{valid,test}.parquet    query_id, product_id (str), label (E/S/I), gain
  products.parquet              product_id (str), text

Labels: Exact -> E, Partial -> S, Irrelevant -> I. Gains for NDCG: E 1.0, S 0.5, I 0
(equivalent to the 2/1/0 scale; NDCG is scale-invariant). Duplicate (query, product) labels
are resolved by majority vote, ties toward the less relevant label. Only queries with at
least one Exact label are used: for the others strict P@k is 0 for every ranking.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

LABEL_MAP = {"Exact": "E", "Partial": "S", "Irrelevant": "I"}
GAINS = {"E": 1.0, "S": 0.5, "I": 0.0}
_ORDER = {"I": 0, "S": 1, "E": 2}
_WS = re.compile(r"\s+")


def _clean(v) -> str:
    return "" if pd.isna(v) else _WS.sub(" ", str(v)).strip()


def product_text(row: dict) -> str:
    """Name first so truncation never drops it; then class, category, description, features."""
    parts = [_clean(row.get("product_name"))]
    for k, label in (("product_class", "Class"), ("category hierarchy", "Category"),
                     ("product_description", "Description")):
        v = _clean(row.get(k))
        if v:
            parts.append(f"{label}: {v}")
    feats = [f.strip() for f in _clean(row.get("product_features")).split("|") if f.strip()]
    if feats:
        parts.append("Features: " + "; ".join(feats))
    return "\n".join(p for p in parts if p)


def load_labels(raw: Path) -> pd.DataFrame:
    lab = pd.read_csv(raw / "label.csv", sep="\t")
    lab["label"] = lab["label"].map(LABEL_MAP)

    def vote(s: pd.Series) -> str:
        c = s.value_counts()
        top = c[c == c.max()].index
        return min(top, key=_ORDER.get)          # tie -> less relevant

    lab = lab.groupby(["query_id", "product_id"], as_index=False)["label"].agg(vote)
    lab["product_id"] = lab["product_id"].astype(str)
    return lab.assign(gain=lab["label"].map(GAINS))


def prepare(raw: str | Path, out: str | Path, n_valid: int, seed: int) -> dict:
    raw, out = Path(raw), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    q = pd.read_csv(raw / "query.csv", sep="\t")[["query_id", "query"]]
    q["query"] = q["query"].astype(str).str.strip()
    lab = load_labels(raw)
    has_e = set(lab.loc[lab.label == "E", "query_id"])
    eligible = q[q.query_id.isin(has_e)].sort_values("query_id")
    valid = eligible.sample(n=n_valid, random_state=seed).sort_values("query_id")
    test = eligible[~eligible.query_id.isin(valid.query_id)]
    rep = {"queries_total": len(q), "queries_without_exact": int(len(q) - len(eligible)),
           "seed": seed, "splits": {}}
    for split, qs in (("valid", valid), ("test", test)):
        r = lab[lab.query_id.isin(qs.query_id)].reset_index(drop=True)
        qs.reset_index(drop=True).to_parquet(out / f"queries_{split}.parquet", index=False)
        r.to_parquet(out / f"qrels_{split}.parquet", index=False)
        per_q = r.groupby("query_id")
        rep["splits"][split] = {
            "queries": len(qs), "judged_pairs": len(r),
            "judged_per_query_median": float(per_q.size().median()),
            "exact_per_query_median": float((r.label == "E").groupby(r.query_id).sum().median()),
            "label_share": r.label.value_counts(normalize=True).round(4).to_dict(),
        }
    p = pd.read_csv(raw / "product.csv", sep="\t")
    prods = pd.DataFrame({"product_id": p.product_id.astype(str),
                          "text": [product_text(r) for r in p.to_dict("records")]})
    prods.to_parquet(out / "products.parquet", index=False)
    n = prods.text.str.len()
    rep["products"] = {"count": len(prods), "text_chars_median": int(n.median()),
                       "text_chars_p95": int(n.quantile(.95)), "empty_text": int((n == 0).sum())}
    rep["test_judged_products_in_catalog"] = float(
        pd.read_parquet(out / "qrels_test.parquet").product_id.isin(set(prods.product_id)).mean())
    return rep
