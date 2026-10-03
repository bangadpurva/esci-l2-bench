"""ESCI L1: content-based dense retrieval over the product catalog."""
from __future__ import annotations

import numpy as np
import pandas as pd


def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)

QUERY_INSTRUCTION = "Given a product search query, retrieve products that match what the shopper asks for"


def qwen_query_prompt(instruction: str = QUERY_INSTRUCTION) -> str:
    """Qwen3-Embedding query format from its model card; documents get no prompt."""
    return f"Instruct: {instruction}\nQuery:"


def exact_topk(Q: np.ndarray, V: np.ndarray, k: int, chunk: int = 100_000):
    """Exact cosine top-k (inputs L2-normalised), streamed over item chunks."""
    n = Q.shape[0]
    best_s = np.full((n, k), -np.inf, dtype=np.float32)
    best_i = np.full((n, k), -1, dtype=np.int64)
    for s in range(0, V.shape[0], chunk):
        S = (Q @ V[s:s + chunk].T).astype(np.float32)
        kk = min(k, S.shape[1])
        part = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
        cand_s = np.concatenate([best_s, np.take_along_axis(S, part, 1)], axis=1)
        cand_i = np.concatenate([best_i, part + s], axis=1)
        keep = np.argpartition(-cand_s, k - 1, axis=1)[:, :k]
        best_s = np.take_along_axis(cand_s, keep, 1)
        best_i = np.take_along_axis(cand_i, keep, 1)
    order = np.argsort(-best_s, axis=1, kind="stable")
    return np.take_along_axis(best_i, order, 1), np.take_along_axis(best_s, order, 1)


def hnsw_topk(Q: np.ndarray, V: np.ndarray, k: int, M: int = 32, ef_construction: int = 200,
              ef_search: int = 512):
    import faiss
    index = faiss.IndexHNSWFlat(V.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    index.add(np.ascontiguousarray(V, dtype=np.float32))
    index.hnsw.efSearch = max(ef_search, k)
    D, I = index.search(np.ascontiguousarray(Q, dtype=np.float32), k)
    return I.astype(np.int64), D.astype(np.float32)


def overlap_at(a: np.ndarray, b: np.ndarray, k: int) -> float:
    return float(np.mean([len(set(x[:k]) & set(y[:k])) / k for x, y in zip(a, b)]))


def retrieved_pools(queries: pd.DataFrame, idx: np.ndarray, sc: np.ndarray, product_ids) -> pd.DataFrame:
    rows = []
    for qi, qid in enumerate(queries.query_id):
        for r, (j, s) in enumerate(zip(idx[qi], sc[qi])):
            if j >= 0:
                rows.append((str(qid), pd.NaT, r, product_ids[j], float(s), "dense"))
    return pd.DataFrame(rows, columns=["user_id", "query_time", "rank", "parent_asin", "l1_score", "source"])


def judged_pools(queries: pd.DataFrame, qrels: pd.DataFrame, Qv: np.ndarray, V: np.ndarray,
                 pid_index: dict) -> pd.DataFrame:
    """Each query's judged products, ordered by the same embedding similarity (the L1 order)."""
    rows = []
    by_q = qrels.groupby("query_id")["product_id"].apply(list).to_dict()
    for qi, qid in enumerate(queries.query_id):
        prods = [p for p in by_q.get(qid, []) if p in pid_index]
        if not prods:
            continue
        s = V[[pid_index[p] for p in prods]] @ Qv[qi]
        for r, j in enumerate(np.argsort(-s, kind="stable")):
            rows.append((str(qid), pd.NaT, r, prods[j], float(s[j]), "judged"))
    return pd.DataFrame(rows, columns=["user_id", "query_time", "rank", "parent_asin", "l1_score", "source"])


def l1_diagnostics(pools: pd.DataFrame, qrels: pd.DataFrame, ks=(10, 50, 100)) -> dict:
    lab = qrels.assign(user_id=qrels.query_id.astype(str)).set_index(["user_id", "product_id"])["label"]
    p = pools.join(lab, on=["user_id", "parent_asin"])
    n_e = (qrels.label == "E").groupby(qrels.query_id.astype(str)).sum()
    out = {"queries": int(pools.user_id.nunique())}
    for k in ks:
        top = p[p["rank"] < k]
        got_e = (top.label == "E").groupby(top.user_id).sum().reindex(n_e.index, fill_value=0)
        out[f"recall_E@{k}"] = float((got_e / n_e.where(n_e > 0)).mean())
        out[f"judged@{k}"] = float(top.label.notna().mean())
        out[f"queries_with_E@{k}"] = float((got_e > 0).mean())
    return out
