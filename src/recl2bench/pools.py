"""Freeze L1 pools: one Parquet per split plus a .sha256, verified on every load."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

POOL_COLUMNS = ["user_id", "query_time", "rank", "parent_asin", "l1_score"]
OPTIONAL_COLUMNS: list[str] = []


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze(pools: pd.DataFrame, path: str | Path, overwrite: bool = False) -> str:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} is frozen; refusing to overwrite")
    missing = set(POOL_COLUMNS) - set(pools.columns)
    if missing:
        raise ValueError(f"pool table missing columns {missing}")
    cols = POOL_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in pools.columns]
    df = pools[cols].sort_values(["user_id", "rank"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n")
    return digest


def load(path: str | Path, top_m: int | None = None) -> tuple[pd.DataFrame, str]:
    """Load and verify the hash. top_m keeps only the first top_m ranks."""
    path = Path(path)
    expected = path.with_suffix(path.suffix + ".sha256").read_text().split()[0]
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"pool hash mismatch for {path}: {actual} != {expected}")
    df = pd.read_parquet(path)
    if top_m is not None:
        df = df[df["rank"] < top_m].reset_index(drop=True)
    return df, actual


def load_eval_cohort(path: str | Path) -> tuple[list[str], str]:
    path = Path(path)
    expected = path.with_suffix(path.suffix + ".sha256").read_text().split()[0]
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"eval cohort hash mismatch for {path}")
    return pd.read_parquet(path).user_id.tolist(), actual
