"""ESCI step 2: embed the catalog and queries, retrieve top-100, gate, freeze.

    python scripts/esci_l1.py                       # dry run: embed, retrieve, report
    python scripts/esci_l1.py --freeze              # write + hash pools (irreversible)

Writes (with --freeze) data/esci/pools/pools_{split}_{retrieved,judged}.parquet + .sha256
and eval_users_{split}_{kind}.parquet (all sampled queries), plus l1_report.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recl2bench.esci import l1 as L  # noqa: E402
from recl2bench.esci.l1 import normalize  # noqa: E402
from recl2bench import pools as P  # noqa: E402


class QwenEncoder:
    def __init__(self, model: str, revision: str | None, max_len: int, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer
        self.m = SentenceTransformer(model, revision=revision)
        self.m.max_seq_length = max_len
        self.name = f"{model}@{revision}|max_len={max_len}"
        self.batch_size = batch_size

    def encode(self, texts, prompt=None):
        return self.m.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=True, prompt=prompt).astype(np.float32)


class FakeEncoder:
    """Test-only: deterministic bag-of-words hashing, so queries match product text."""
    name = "fake"

    def encode(self, texts, prompt=None):
        dim = 64
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in str(t).lower().split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % dim] += 1
        return normalize(out + 1e-6)


def item_vectors(products: pd.DataFrame, enc, cache: Path) -> np.ndarray:
    key = hashlib.sha256((enc.name + "\0" + "\0".join(products.product_id)).encode()).hexdigest()[:16]
    f = cache / f"product_emb_{key}.npy"
    if f.exists():
        return np.load(f).astype(np.float32)
    V = enc.encode(products.text.tolist())
    cache.mkdir(parents=True, exist_ok=True)
    np.save(f, V.astype(np.float16))
    return V


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/esci.yaml")
    ap.add_argument("--proc", default="data/esci/processed")
    ap.add_argument("--pools-dir", default="data/esci/pools")
    ap.add_argument("--index", choices=["exact", "hnsw"])
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--encoder", help="'fake' for tests")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    l1c = cfg["l1"]
    proc, k = Path(a.proc), l1c["top_m"]
    index = a.index or l1c["index"]

    products = pd.read_parquet(proc / "products.parquet")
    pids = products.product_id.tolist()
    pid_index = {p: i for i, p in enumerate(pids)}
    enc = FakeEncoder() if a.encoder == "fake" else QwenEncoder(
        l1c["encoder"], l1c["revision"], l1c["max_item_tokens"], l1c.get("batch_size", 64))
    V = item_vectors(products, enc, proc)
    prompt = None if a.encoder == "fake" else L.qwen_query_prompt(l1c["query_instruction"])

    report = {"encoder": enc.name, "index": index, "top_m": k, "catalog": len(pids), "splits": {}}
    for split in ("valid", "test"):
        q = pd.read_parquet(proc / f"queries_{split}.parquet")
        qr = pd.read_parquet(proc / f"qrels_{split}.parquet")
        Qv = enc.encode(q["query"].tolist(), prompt=prompt)
        ex_i, ex_s = L.exact_topk(Qv, V, k)
        info = {}
        if index == "hnsw":
            h_i, h_s = L.hnsw_topk(Qv, V, k)
            info["ann_overlap@100"] = L.overlap_at(h_i, ex_i, k)
            info["gate_passed"] = info["ann_overlap@100"] >= l1c["min_ann_overlap"]
            idx, sc = h_i, h_s
            if split == "valid" and not info["gate_passed"]:
                raise SystemExit(f"ANN gate failed: overlap@100 = {info['ann_overlap@100']:.3f}")
        else:
            idx, sc = ex_i, ex_s
        retrieved = L.retrieved_pools(q, idx, sc, pids)
        judged = L.judged_pools(q, qr, Qv, V, pid_index)
        info["retrieved"] = L.l1_diagnostics(retrieved, qr)
        info["judged_pool_sizes"] = judged.groupby("user_id").size().describe().round(1).to_dict()
        info["judged_products_missing_from_catalog"] = int((~qr.product_id.isin(pid_index)).sum())
        if a.freeze:
            for kind, pools in (("retrieved", retrieved), ("judged", judged)):
                sha = P.freeze(pools, Path(a.pools_dir) / f"pools_{split}_{kind}.parquet")
                users = Path(a.pools_dir) / f"eval_users_{split}_{kind}.parquet"
                pd.DataFrame({"user_id": sorted(pools.user_id.unique())}).to_parquet(users, index=False)
                csha = P.sha256_file(users)
                users.with_suffix(users.suffix + ".sha256").write_text(f"{csha}  {users.name}\n")
                info[f"{kind}_pool_sha256"], info[f"{kind}_eval_sha256"] = sha, csha
        report["splits"][split] = info
    out = Path(a.pools_dir if a.freeze else proc) / "l1_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return report


if __name__ == "__main__":
    main()
