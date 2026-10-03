"""ESCI end to end on synthetic files in the official format."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import esci_l1  # noqa: E402
import esci_prepare  # noqa: E402
import esci_score  # noqa: E402
from recl2bench.esci import data as D  # noqa: E402

WORDS = ["fan", "tire", "fence", "laptop", "case", "lamp", "drill", "guitar", "mat", "kettle"]


@pytest.fixture
def esci_raw(tmp_path):
    rng = np.random.default_rng(0)
    prods, ex = [], []
    for i in range(600):
        w = WORDS[i % len(WORDS)]
        prods.append({"product_id": f"P{i:04d}", "product_title": f"{w} model {i % 7} deluxe",
                      "product_description": "<p>Great&nbsp;item</p>" if i % 2 else None,
                      "product_bullet_point": f"{w} feature\nsize {i % 5}", "product_brand": f"B{i % 9}",
                      "product_color": None, "product_locale": "us"})
    prods.append({**prods[0], "product_id": "JP1", "product_locale": "jp"})
    qid = 0
    for split, n in (("train", 40), ("test", 50)):
        for _ in range(n):
            w = WORDS[qid % len(WORDS)]
            cand = [p for p in prods[:600] if p["product_title"].startswith(w)]
            pick = rng.choice(len(cand), 16, replace=False)
            for j, c in enumerate(pick):
                lab = "E" if j < 6 else ("S" if j < 10 else ("C" if j < 11 else "I"))
                ex.append({"example_id": len(ex), "query": f" {w} model {qid % 7}", "query_id": qid,
                           "product_id": cand[c]["product_id"], "product_locale": "us", "esci_label": lab,
                           "small_version": 1, "large_version": 1, "split": split})
            qid += 1
    ex.append({**ex[0], "query_id": 999, "small_version": 0})      # dropped: not small_version
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame(ex).to_parquet(raw / D.EXAMPLES)
    pd.DataFrame(prods).to_parquet(raw / D.PRODUCTS)
    return tmp_path


def test_esci_end_to_end(esci_raw):
    t = esci_raw
    proc, pools, runs, res = t / "processed", t / "pools", t / "runs", t / "results"
    esci_prepare.main(["--raw", str(t / "raw"), "--out", str(proc), "--n-test", "30", "--n-valid", "20"])
    rep = json.loads((proc / "prepare_report.json").read_text())
    assert rep["splits"]["test"]["queries"] == 30 and rep["products"]["count"] == 600   # JP excluded
    assert "<p>" not in pd.read_parquet(proc / "products.parquet").text.str.cat()

    common = ["--config", str(ROOT / "configs/esci.yaml"), "--proc", str(proc), "--pools-dir", str(pools)]
    esci_l1.main(common + ["--encoder", "fake", "--index", "exact", "--freeze"])
    l1 = json.loads((pools / "l1_report.json").read_text())["splits"]["test"]
    assert l1["retrieved"]["queries"] == 30 and 0 <= l1["retrieved"]["recall_E@100"] <= 1
    with pytest.raises(FileExistsError):
        esci_l1.main(common + ["--encoder", "fake", "--index", "exact", "--freeze"])

    sc = ["--config", str(ROOT / "configs/esci.yaml"), "--configs", str(ROOT / "configs"), "--proc", str(proc),
          "--pools-dir", str(pools), "--runs-dir", str(runs), "--results-dir", str(res), "--tokenizer", "whitespace",
          "--split", "test", "--resamples", "300"]
    for setting in ("retrieved", "judged"):
        for m in ("l1_order", "random", "oracle"):
            esci_score.main(sc + ["--model", m, "--setting", setting])
        esci_score.main(sc + ["--model", "random", "--setting", setting, "--limit-users", "3", "--seed", "4"])
        stem = esci_score.main(sc + ["--report", "--setting", setting])
        tab = pd.read_csv(stem.with_suffix(".csv")).set_index("model")
        assert set(tab.index) == {"l1_order", "random", "oracle"}          # smoke run excluded
        assert tab.loc["oracle", "p@10"] >= tab["p@10"].max() - 1e-12
        assert pd.notna(tab.loc["random", "p_holm"]) and pd.isna(tab.loc["l1_order", "p_holm"])
        if setting == "judged":
            assert tab.loc["l1_order", "judged@10"] == pytest.approx(1.0)  # 16 judged items per query
            assert tab.loc["oracle", "ndcg@10"] == pytest.approx(1.0)


def test_run_esci_stages(esci_raw, tmp_path):
    """Drive the bash stages in a copy of the repo (fake encoder, whitespace tokenizer)."""
    import os
    import shutil
    import subprocess
    repo = tmp_path / "repo"
    top = {".git", "data", "runs", "results", "logs", "backups", ".hf_cache", ".venv"}
    shutil.copytree(ROOT, repo, ignore=lambda d, n: [x for x in n if x in {"__pycache__", ".pytest_cache"}
                                                     or (Path(d) == ROOT and x in top)])
    shutil.copytree(esci_raw / "raw", repo / "data/esci/raw")
    env = {**os.environ, "FORCE": "1"}

    def stage(name, extra=""):
        p = subprocess.run(["bash", "scripts/run_esci.sh", name], cwd=repo, capture_output=True, text=True,
                           env={**env, "ESCI_ARGS": extra})
        assert p.returncode == 0, f"{name}:\n{p.stdout[-1500:]}\n{p.stderr[-1500:]}"
        return p.stdout

    stage("prepare", "--n-test 20 --n-valid 10")
    stage("l1-dryrun", "--encoder fake --index exact")
    stage("l1-freeze", "--encoder fake --index exact")
    stage("baselines", "--tokenizer whitespace --resamples 200")
    out = stage("report", "")
    assert "| oracle |" in out and "P@10 judged-only" in out
    stage("backup")
    assert list((repo / "backups").glob("esci_*.tar.gz"))
