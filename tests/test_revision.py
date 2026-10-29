"""Tests for the revision analyses (Amendment 5) on small synthetic data.

The end-to-end test builds a fake ESCI-shaped workspace (config, qrels, frozen pools, runs with saved
scores) and checks the generated numbers against values computed independently here.
"""
from __future__ import annotations

import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from recl2bench import pools as P  # noqa: E402
from recl2bench import revision as R  # noqa: E402
from recl2bench.esci import data as D  # noqa: E402
from recl2bench.esci.metrics import query_metrics  # noqa: E402
from recl2bench.rerankers.base import CandidateItem, UserProfile  # noqa: E402
from recl2bench.rerankers.clm import CLMPointwise, CLMReranker  # noqa: E402
from recl2bench.rerankers.composite import SumReranker  # noqa: E402
from recl2bench.rerankers.decision import SystemOneBackend, YesNoQuestion  # noqa: E402

GAIN = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
N_Q, N_C = 30, 20


def _world(seed=0, n_q=N_Q, n_c=N_C, split="test"):
    rng = np.random.default_rng(seed)
    pools, qrels, truth = [], [], {}
    for q in range(n_q):
        uid = f"{split[0]}{q}"
        lab = rng.choice(list("ESCI"), size=n_c, p=[0.3, 0.3, 0.1, 0.3])
        sim = np.sort(rng.normal(size=n_c))[::-1]            # L1 similarity, best first
        for r in range(n_c):
            pid = f"{uid}-p{r}"
            pools.append((uid, "", r, pid, float(sim[r])))
            if r % 5 != 4:                                   # one in five candidates is unlabelled
                qrels.append((uid, pid, lab[r]))
        truth[uid] = {f"{uid}-p{r}": lab[r] for r in range(n_c) if r % 5 != 4}
    pools = pd.DataFrame(pools, columns=P.POOL_COLUMNS)
    qrels = pd.DataFrame(qrels, columns=["query_id", "product_id", "label"])
    return pools, qrels, truth


def _scores(pools, truth, kind, rng):
    s = []
    for r in pools.itertuples():
        g = GAIN.get(truth[r.user_id].get(r.parent_asin), 0.0)
        s.append({"l1": -float(r.rank), "oracle": g + 1e-6 * (-r.rank), "good": g + rng.normal(0, 0.25),
                  "weak": g + rng.normal(0, 0.9), "random": rng.normal()}[kind])
    return np.asarray(s)


def _write_run(runs_dir, model, split, setting, pools, truth, kind, rng, started="2026-10-29T10:00:00+00:00"):
    sc = pools[["user_id", "parent_asin", "rank"]].rename(columns={"rank": "l1_rank"}).copy()
    sc["score"] = _scores(pools, truth, kind, rng)
    ranked = R.ranked_ids(sc)
    pu = R.evaluate_orders(ranked, truth)
    pu["latency_s"], pu["failures"] = 0.1, 0
    pu = pu.assign(split=split, setting=setting, limit_users=0)
    d = Path(runs_dir) / f"{started}_{model}_{split}_{setting}".replace(":", "")
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"model": model, "setting": setting, "started_at": started}))
    pu.to_parquet(d / "per_user.parquet", index=False)
    sc.to_parquet(d / "scores.parquet", index=False)
    return d


@pytest.fixture()
def ws(tmp_path):
    (tmp_path / "configs").mkdir()
    shutil.copy(ROOT / "configs" / "esci.yaml", tmp_path / "configs" / "esci.yaml")
    rng = np.random.default_rng(1)
    proc, pdir = tmp_path / "data/esci/processed", tmp_path / "data/esci/pools"
    proc.mkdir(parents=True)
    pdir.mkdir(parents=True)
    worlds = {}
    for split in ("test", "valid"):
        pools, qrels, truth = _world(seed=3 if split == "test" else 4, split=split)
        qrels.to_parquet(proc / f"qrels_{split}.parquet", index=False)
        P.freeze(pools, pdir / f"pools_{split}_judged.parquet")
        worlds[split] = (pools, truth)
        for model, kind in [("l1_order", "l1"), ("oracle", "oracle"), ("jev", "good"), ("qwen3", "weak"),
                            ("random", "random"), ("clm", "random")]:
            _write_run(tmp_path / "runs/esci", model, split, "judged", pools, truth, kind, rng)
    return tmp_path, worlds


def test_fusion_endpoints_and_rrf():
    pools, _, truth = _world()
    rng = np.random.default_rng(0)
    sc = pools[["user_id", "parent_asin", "rank"]].rename(columns={"rank": "l1_rank"}).copy()
    sc["score"] = _scores(pools, truth, "good", rng)
    assert R.fuse_z(sc, pools, 0.0) == R.ranked_ids(sc.assign(score=-sc.l1_rank.astype(float)))
    assert R.fuse_z(sc, pools, 1.0) == R.ranked_ids(sc)
    same = sc.assign(score=-sc.l1_rank.astype(float))
    assert R.fuse_rrf(same) == R.ranked_ids(same)           # fusing L1 with itself changes nothing


def test_fusion_nan_scores_sink_and_do_not_crash():
    pools, _, truth = _world(n_q=3)
    sc = pools[["user_id", "parent_asin", "rank"]].rename(columns={"rank": "l1_rank"}).copy()
    sc["score"] = np.arange(len(sc), dtype=float)
    sc.loc[sc.index[:2], "score"] = np.nan
    for orders in (R.fuse_z(sc, pools, 0.7), R.fuse_rrf(sc), R.ranked_ids(sc)):
        assert all(len(set(v)) == N_C for v in orders.values())


def test_select_alpha_uses_validation_and_prefers_better_l2():
    pools, _, truth = _world(split="valid")
    rng = np.random.default_rng(0)
    sc = pools[["user_id", "parent_asin", "rank"]].rename(columns={"rank": "l1_rank"}).copy()
    sc["score"] = _scores(pools, truth, "oracle", rng)
    a, grid = R.select_alpha(sc, pools, truth)
    assert a >= 0.5 and grid[a] >= grid[0.0]
    # contract: the choice attains the validation maximum, ties go to the larger alpha
    sc["score"] = _scores(pools, truth, "random", rng)
    a2, grid2 = R.select_alpha(sc, pools, truth)
    best = max(grid2.values())
    assert grid2[a2] == pytest.approx(best)
    assert all(not (abs(v - best) < 1e-12 and k > a2) for k, v in grid2.items())


def test_composition_sums_to_k():
    pools, _, truth = _world()
    sc = pools[["user_id", "parent_asin", "rank"]].rename(columns={"rank": "l1_rank"}).copy()
    sc["score"] = -sc.l1_rank.astype(float)
    comp = R.composition(R.ranked_ids(sc), truth, 10)
    assert sum(comp.values()) == pytest.approx(10.0)


def test_tables_end_to_end(ws, tmp_path_factory):
    root, worlds = ws
    import revision_tables as T
    out = root / "out"
    T.NUMBERS.clear()
    T.WARN.clear()
    T.main(["ndcg", "composition", "exact_subset", "--root", str(root), "--out", str(out), "--resamples", "500"])
    tex = (out / "tab_ndcg.tex").read_text()
    assert tex.startswith("\\begin{tabular}") and "Jev" in tex and "\\bottomrule" in tex
    # independent NDCG@10 for Jev on the judged test setting
    pools, truth = worlds["test"]
    pu = pd.read_parquet(next((root / "runs/esci").glob("*_jev_test_judged")) / "per_user.parquet")
    sc = pd.read_parquet(next((root / "runs/esci").glob("*_jev_test_judged")) / "scores.parquet")
    D.GAINS.clear(); D.GAINS.update(D.ESCI_GAINS)
    expect = np.mean([query_metrics(r, truth[u])["ndcg@10"] for u, r in R.ranked_ids(sc).items()])
    assert f"{expect:.3f}" in tex
    assert pu["ndcg@10"].mean() == pytest.approx(expect)
    nums = json.loads((out / "numbers.json").read_text())
    assert "ndcg_diff[jev-qwen3][ESCI judged]" in nums
    assert "p_holm" in nums["ndcg_diff[jev-l1_order][ESCI judged]"]
    delta = (out / "tab_ndcg_delta.tex").read_text()
    big = nums["ndcg_diff[jev-l1_order][ESCI judged]"]
    assert ("$^{*}$" in delta) == (big["p_holm"] < 0.05)        # stars exactly where the caption says
    comp = nums["composition[jev][ESCI judged]"]
    assert sum(comp.values()) == pytest.approx(10.0)
    assert (out / "tab_exact_subset.tex").exists()
    # missing ESCI retrieved / WANDS cells are flagged, never silently dropped
    assert "n/a" in tex and any("tab_ndcg" in w for w in T.WARN)


def test_tables_refuse_when_no_data_and_strict(tmp_path):
    (tmp_path / "configs").mkdir()
    shutil.copy(ROOT / "configs" / "esci.yaml", tmp_path / "configs" / "esci.yaml")
    import revision_tables as T
    T.WARN.clear()
    with pytest.raises(SystemExit):
        T.main(["ndcg", "--root", str(tmp_path), "--out", str(tmp_path / "o"), "--strict"])


def test_fusion_script_end_to_end(ws):
    root, _ = ws
    # the fusion script needs the same settings; only ESCI judged exists, the others are skipped
    import revision_fusion as F
    out = root / "outf"
    F.main(["--models", "jev", "qwen3", "--root", str(root), "--out", str(out), "--resamples", "300"])
    tex = (out / "tab_fusion.tex").read_text()
    assert "Jev" in tex and "$\\alpha$=" in tex
    chosen = json.loads((out / "fusion_alpha.json").read_text())
    assert set(chosen) == {"ESCI judged/jev", "ESCI judged/qwen3"}
    assert all(0.0 <= v["alpha"] <= 1.0 for v in chosen.values())


# --- model-side additions ------------------------------------------------------------------
class _FakeEngine:
    """Scores a candidate from its own text only (as the CLM adapter documents), softmax over the list."""
    def rank(self, query, candidates, instructions=None):
        raw = [(len(c) % 7) + (1.5 if c == "yes" and "good" in query else 0.0) for c in candidates]
        z = sum(math.exp(r) for r in raw)
        out = [{"index": i, "candidate": c, "prob": math.exp(r) / z} for i, (c, r) in enumerate(zip(candidates, raw))]
        return sorted(out, key=lambda o: -o["prob"])


def _cands(n=12):
    return [CandidateItem(f"c{i}", i, -float(i), "x" * (3 + i * 5 % 11)) for i in range(n)]


def test_clm_shuffle_keeps_alignment_for_candidate_local_scores():
    prof = UserProfile("u1", "", (), (), "a query")
    base = CLMReranker("", engine=_FakeEngine()).rerank(prof, _cands())
    shuf = CLMReranker("", engine=_FakeEngine(), shuffle_seed=7).rerank(prof, _cands())
    # the fake engine's list normaliser is a common factor over different list orders: same set -> same scores
    assert shuf.scores == pytest.approx(base.scores)


def test_clm_pointwise_uses_yes_probability():
    prof = UserProfile("u1", "", (), (), "good query")
    r = CLMPointwise("", "Does it match?", engine=_FakeEngine()).rerank(prof, _cands(5))
    assert len(r.scores) == 5 and all(0 < s < 1 for s in r.scores) and r.failures == 0


class _Const:
    name = "c"
    def __init__(self, v): self.v = v
    def rerank(self, p, c):
        from recl2bench.rerankers.base import RerankResult
        return RerankResult([self.v] * len(c), "m")


def test_sum_reranker_sums_and_propagates_nan():
    from recl2bench.rerankers.base import RerankResult
    prof = UserProfile("u", "", (), (), "q")
    s = SumReranker([_Const(0.25), _Const(0.5)])
    assert s.rerank(prof, _cands(3)).scores == [0.75] * 3
    nan = SumReranker([_Const(0.25), _Const(float("nan"))])
    assert nan.rerank(prof, _cands(3)).failures == 3
    with pytest.raises(ValueError):
        SumReranker([_Const(1.0)])


def test_question_only_instructions_and_default_unchanged():
    q_full = YesNoQuestion("a", "Q?", "yes means", "no means")
    q_terse = YesNoQuestion("b", "Q?", "", "")
    assert SystemOneBackend.instructions(q_full) == "Q?\nAnswer true if: yes means\nAnswer false if: no means"
    assert SystemOneBackend.instructions(q_terse) == "Q?"


def test_variant_templates_load_and_have_required_fields():
    import yaml
    for t in ("esci_v1", "esci_v1p", "esci_v2", "esci_v3", "esci_v4", "esci_v5"):
        d = yaml.safe_load((ROOT / "configs/templates" / f"{t}.yaml").read_text())
        assert {"id", "question", "true_means", "false_means", "instruction", "state_labels"} <= set(d), t
        assert d["id"] == t
        assert d["question"].strip() in d["instruction"].strip().replace("\n", " ") or t == "esci_v1"
    main = yaml.safe_load((ROOT / "configs/templates/esci_v1.yaml").read_text())
    assert main["question"].strip() == "Does this product exactly match what the shopper's search query asks for?"


def test_run_name_tags_only_revision_variants():
    spec = importlib.util.spec_from_file_location("esci_score", ROOT / "scripts/esci_score.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    A = lambda **k: type("A", (), {"model": "jev", "template": None, "graded_with": None,
                                   "clm_shuffle_seed": None, **k})()
    assert m.run_name(A(), "esci_v1") == "jev"                                   # main runs keep their names
    assert m.run_name(A(template="esci_v2"), "esci_v2") == "jev@esci_v2"
    assert m.run_name(A(template="esci_v1p", graded_with="esci_v5"), "esci_v1p") == "jev@esci_v1p+esci_v5"
    assert m.run_name(A(model="clm", clm_shuffle_seed=1), "esci_v1") == "clm#shuffle1"


def test_ablation_table_holm_stars_and_missing_cells(ws):
    root, worlds = ws
    import revision_tables as T
    pools, truth = worlds["valid"]
    rng = np.random.default_rng(5)
    for suffix in ("@esci_v2", "@esci_v4"):
        _write_run(root / "runs/esci_rev", "jev" + suffix, "valid", "judged", pools, truth, "good", rng)
        _write_run(root / "runs/esci_rev", "qwen3" + suffix, "valid", "judged", pools, truth, "weak", rng)
    T.NUMBERS.clear(); T.WARN.clear()
    out = root / "out_abl"
    T.main(["ablation", "--root", str(root), "--out", str(out), "--resamples", "500"])
    tex = (out / "tab_ablation.tex").read_text()
    r = T.NUMBERS["ablation[Paraphrase (v2)][ESCI judged]"]
    assert "p_holm" in r and r["p_holm"] >= r["p"] - 1e-12          # Holm never lowers a p-value
    assert ("$^{*}$" in tex) == any(v["p_holm"] < 0.05 for k, v in T.NUMBERS.items() if k.startswith("ablation["))
    assert "n/a" in tex                                              # wordings without runs are flagged, not dropped
