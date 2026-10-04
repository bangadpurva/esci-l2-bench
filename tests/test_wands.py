"""WANDS on synthetic files in the official format, end to end through the shared pipeline."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import esci_l1  # noqa: E402
import esci_score  # noqa: E402
import wands_prepare  # noqa: E402
from recl2bench import wands as W  # noqa: E402
from recl2bench.esci import data as D  # noqa: E402

WORDS = ["sofa", "lamp", "rug", "desk", "chair", "bed", "mirror", "shelf"]


@pytest.fixture
def wands_raw(tmp_path):
    rng = np.random.default_rng(0)
    raw = tmp_path / "raw"
    raw.mkdir()
    prods = [{"product_id": i, "product_name": f"{WORDS[i % 8]} model {i % 5}", "product_class": WORDS[i % 8].title(),
              "category hierarchy": "Furniture / Living", "product_description": None if i % 3 else "solid wood",
              "product_features": "color : grey|material:oak", "rating_count": 1, "average_rating": 4.0,
              "review_count": 1} for i in range(400)]
    queries = [{"query_id": q, "query": f"{WORDS[q % 8]} model {q % 5}", "query_class": "x"} for q in range(60)]
    labels = []
    for q in range(60):
        cand = [p["product_id"] for p in prods if p["product_name"].startswith(WORDS[q % 8])]
        for j, c in enumerate(rng.choice(cand, 20, replace=False)):
            lab = "Partial" if q >= 50 else ("Exact" if j < 5 else ("Partial" if j < 12 else "Irrelevant"))
            labels.append({"query_id": q, "product_id": int(c), "label": lab})
    labels += [{"query_id": 0, "product_id": labels[0]["product_id"], "label": "Irrelevant"}] * 2   # vote: 2 I vs 1 E
    for name, rows in (("product", prods), ("query", queries), ("label", labels)):
        df = pd.DataFrame(rows)
        if name == "label":
            df.insert(0, "id", range(len(df)))
        df.to_csv(raw / f"{name}.csv", sep="\t", index=False)
    return tmp_path


def test_labels_vote_and_text(wands_raw):
    lab = W.load_labels(wands_raw / "raw")
    assert set(lab.label) <= {"E", "S", "I"} and not lab.duplicated(["query_id", "product_id"]).any()
    first = lab[(lab.query_id == 0) & (lab.product_id == str(pd.read_csv(wands_raw / "raw/label.csv", sep="\t").product_id[0]))]
    assert first.label.item() == "I" and first.gain.item() == 0.0
    t = W.product_text({"product_name": "oak desk", "product_class": "Desks", "category hierarchy": None,
                        "product_description": "  big  desk ", "product_features": "a : 1|b:2|"})
    assert t.splitlines() == ["oak desk", "Class: Desks", "Description: big desk", "Features: a : 1; b:2"]


def test_wands_end_to_end(wands_raw):
    t = wands_raw
    proc, pools, runs, res = t / "processed", t / "pools", t / "runs", t / "results"
    rep = wands_prepare.main(["--raw", str(t / "raw"), "--out", str(proc), "--n-valid", "10"])
    assert rep["queries_without_exact"] == 10                       # queries 50-59 have no Exact
    assert rep["splits"]["valid"]["queries"] == 10 and rep["splits"]["test"]["queries"] == 40
    assert rep["test_judged_products_in_catalog"] == 1.0

    cfg = str(ROOT / "configs/wands.yaml")
    esci_l1.main(["--config", cfg, "--proc", str(proc), "--pools-dir", str(pools), "--encoder", "fake", "--freeze"])
    l1 = json.loads((pools / "l1_report.json").read_text())
    assert l1["index"] == "exact" and l1["splits"]["test"]["retrieved"]["queries"] == 40

    sc = ["--config", cfg, "--configs", str(ROOT / "configs"), "--proc", str(proc), "--pools-dir", str(pools),
          "--runs-dir", str(runs), "--results-dir", str(res), "--tokenizer", "whitespace", "--split", "test",
          "--resamples", "300"]
    for setting in ("retrieved", "judged"):
        for m in ("l1_order", "random", "oracle"):
            esci_score.main(sc + ["--model", m, "--setting", setting])
        stem = esci_score.main(sc + ["--report", "--setting", setting])
        assert stem.with_suffix(".md").read_text().startswith("# WANDS results")
        tab = pd.read_csv(stem.with_suffix(".csv")).set_index("model")
        assert tab.loc["oracle", "p@10"] >= tab["p@10"].max() - 1e-12
        if setting == "judged":
            assert tab.loc["oracle", "ndcg@10"] == pytest.approx(1.0)
    assert D.GAINS == {"E": 1.0, "S": 0.5, "I": 0.0}
    esci_score.main(sc[:0] + ["--config", str(ROOT / "configs/esci.yaml"), "--report", "--runs-dir", str(runs),
                              "--results-dir", str(res), "--resamples", "50"])
    assert D.GAINS == D.ESCI_GAINS                                  # gains reset per run
