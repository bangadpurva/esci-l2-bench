"""ESCI (Shopping Queries Dataset): query sampling, graded labels, product text.

Setting: US locale, `small_version == 1` (the official Task 1 ranking subset).
Test queries come from the `test` split; validation queries are sampled from `train`
(ESCI has no validation split) and are excluded from any later fine-tuning data.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ("E", "S", "C", "I")
# Official ESCI Task 1 gains (Reddy et al., 2022)
ESCI_GAINS = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
GAINS = dict(ESCI_GAINS)   # the active gains; esci_score.py sets them from the dataset config

EXAMPLES = "shopping_queries_dataset_examples.parquet"
PRODUCTS = "shopping_queries_dataset_products.parquet"
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def load_examples(raw: str | Path, locale: str = "us") -> pd.DataFrame:
    cols = ["query_id", "query", "product_id", "product_locale", "esci_label", "small_version", "split"]
    e = pd.read_parquet(Path(raw) / EXAMPLES, columns=cols)
    e = e[(e.small_version == 1) & (e.product_locale == locale)].drop(columns=["small_version"])
    e["query"] = e["query"].str.strip()
    return e.reset_index(drop=True)


def sample_queries(examples: pd.DataFrame, split: str, n: int | None, seed: int) -> pd.DataFrame:
    q = examples[examples.split == split][["query_id", "query"]].drop_duplicates("query_id")
    q = q.sort_values("query_id")
    if n is not None and n < len(q):
        q = q.sample(n=n, random_state=seed).sort_values("query_id")
    return q.reset_index(drop=True)


def qrels(examples: pd.DataFrame, query_ids) -> pd.DataFrame:
    r = examples[examples.query_id.isin(set(query_ids))][["query_id", "product_id", "esci_label"]]
    r = r.rename(columns={"esci_label": "label"}).drop_duplicates(["query_id", "product_id"])
    return r.assign(gain=r.label.map(GAINS)).reset_index(drop=True)


def clean(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    s = html.unescape(_TAG.sub(" ", str(v)))
    return _WS.sub(" ", s).strip()


def product_text(row: dict) -> str:
    """Title and brand first so truncation never drops them; then color, bullets, description."""
    parts = [clean(row.get("product_title"))]
    for k, label in (("product_brand", "Brand"), ("product_color", "Color")):
        v = clean(row.get(k))
        if v:
            parts.append(f"{label}: {v}")
    bullets = row.get("product_bullet_point")
    if isinstance(bullets, str) and bullets.strip():
        parts.append("Features: " + "; ".join(clean(b) for b in bullets.split("\n") if clean(b)))
    d = clean(row.get("product_description"))
    if d:
        parts.append("Description: " + d)
    return "\n".join(p for p in parts if p)


def load_products(raw: str | Path, locale: str = "us", ids: set | None = None) -> pd.DataFrame:
    """product_id, text. Reads only the locale (and optionally only `ids`)."""
    import pyarrow.parquet as pq
    filters = [("product_locale", "=", locale)]
    if ids is not None:
        filters.append(("product_id", "in", sorted(ids)))
    t = pq.read_table(Path(raw) / PRODUCTS, filters=filters,
                      columns=["product_id", "product_title", "product_description",
                               "product_bullet_point", "product_brand", "product_color"])
    df = t.to_pandas()
    out = pd.DataFrame({"product_id": df.product_id,
                        "text": [product_text(r) for r in df.to_dict("records")]})
    return out.drop_duplicates("product_id").reset_index(drop=True)


def write_products(raw: str | Path, out_path: str | Path, locale: str = "us",
                   batch_size: int = 20_000) -> dict:
    """Stream products -> parquet(product_id, text) without holding the catalog in memory."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    cols = ["product_id", "product_title", "product_description", "product_bullet_point",
            "product_brand", "product_color", "product_locale"]
    pf = pq.ParquetFile(Path(raw) / PRODUCTS)
    schema = pa.schema([("product_id", pa.string()), ("text", pa.string())])
    seen, n, n_chars, empty = set(), 0, [], 0
    with pq.ParquetWriter(out_path, schema) as w:
        for b in pf.iter_batches(batch_size=batch_size, columns=cols):
            df = b.to_pandas()
            df = df[(df.product_locale == locale) & ~df.product_id.isin(seen)]
            if df.empty:
                continue
            seen.update(df.product_id)
            texts = [product_text(r) for r in df.to_dict("records")]
            w.write_table(pa.table({"product_id": df.product_id.tolist(), "text": texts}, schema=schema))
            n += len(texts)
            lens = [len(t) for t in texts]
            n_chars.extend(lens[::10])                   # 10% sample for length stats
            empty += sum(1 for L in lens if L == 0)
    s = pd.Series(n_chars) if n_chars else pd.Series([0])
    return {"count": n, "text_chars_median": int(s.median()), "text_chars_p95": int(s.quantile(.95)),
            "empty_text": empty}
