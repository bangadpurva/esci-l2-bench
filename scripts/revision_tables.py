"""Revision tables (Amendment 5): turns finished runs into LaTeX tabulars for the manuscript.

    python scripts/revision_tables.py all --out paper_generated           # whatever runs exist
    python scripts/revision_tables.py ndcg baselines --strict             # fail if a needed run is missing

Reads only files the pipeline already wrote (runs/<dataset>[_rev]/, data/<dataset>/{processed,pools}).
Each table is written as `<name>.tex` (a bare tabular, the manuscript supplies caption and label) plus
`<name>.csv`; key numbers also go to numbers.json. A table is written only if its core runs exist, and any
missing cell is printed as "n/a" so scripts/check_slots.py can refuse to ship it.

Tables:  ndcg  composition  exact_subset  baselines  ablation  clm_controls
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recl2bench import pools as P  # noqa: E402
from recl2bench import revision as R  # noqa: E402
from recl2bench.esci import data as D  # noqa: E402
from recl2bench.esci import score as ES  # noqa: E402

SETTINGS = [("ESCI judged", "esci", "judged"), ("ESCI retr.", "esci", "retrieved"),
            ("WANDS", "wands", "retrieved")]
NAMES = {"oracle": "Oracle (ceiling)", "jev": "Jev", "qwen3": "Qwen3-Reranker-0.6B",
         "l1_order": "Embedding order", "clm": "CLM", "random": "Random",
         "qwen3_4b": "Qwen3-Reranker-4B", "qwen3_8b": "Qwen3-Reranker-8B",
         "llm_qwen3_8b": "Qwen3-8B, prompted", "bge_m3": "bge-reranker-v2-m3",
         "clm_pointwise": "CLM, pointwise yes/no"}
MISSING = "n/a"
NUMBERS: dict = {}
WARN: list[str] = []


def f3(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return MISSING
    return f"{x:.3f}" if x >= 0 else f"$-${abs(x):.3f}"


def fs(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return MISSING
    return f"+{x:.3f}" if x >= 0 else f"$-${abs(x):.3f}"


def fd(r) -> str:
    if r is None:
        return MISSING
    star = "$^{*}$" if r.get("p_holm", 1) < 0.05 else ""
    return f"{fs(r['mean'])} [{fs(r['lo'])}, {fs(r['hi'])}]{star}"


class Ctx:
    """Everything needed to read one (dataset, setting, split)."""

    def __init__(self, root: Path, ds: str, setting: str, split: str):
        self.root, self.ds, self.setting, self.split = root, ds, setting, split
        cfg = yaml.safe_load((root / "configs" / f"{ds}.yaml").read_text())
        D.GAINS.clear()
        D.GAINS.update(cfg.get("metrics", {}).get("gains") or D.ESCI_GAINS)
        proc = root / "data" / ds / "processed"
        self.labels = ES.labels_by_query(pd.read_parquet(proc / f"qrels_{split}.parquet"))
        self.runs = R.find_runs([root / "runs" / ds, root / "runs" / f"{ds}_rev"], split, setting)
        self.pools_path = root / "data" / ds / "pools" / f"pools_{split}_{setting}.parquet"

    def pools(self):
        return P.load(self.pools_path)[0]

    def has(self, m: str) -> bool:
        return m in self.runs

    def per_user(self, m: str):
        return self.runs[m]["per_user"] if m in self.runs else None

    def orders(self, m: str):
        return R.ranked_ids(pd.read_parquet(self.runs[m]["dir"] / "scores.parquet"))


def ctxs(root, split, only_ok=True):
    out = {}
    for label, ds, setting in SETTINGS:
        try:
            out[label] = Ctx(root, ds, setting, split)
        except FileNotFoundError as e:
            WARN.append(f"{label}/{split}: {e.filename}")
    return out


def write(out: Path, name: str, header: list[str], rows: list[list[str]], spec: str, df: pd.DataFrame | None = None,
          groups: list[tuple[str, int]] | None = None):
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule"]
    if groups:
        lines.append(" & " + " & ".join(f"\\multicolumn{{{n}}}{{c}}{{{g}}}" for g, n in groups) + " \\\\")
        c = 2
        cm = []
        for _, n in groups:
            cm.append(f"\\cmidrule(lr){{{c}-{c + n - 1}}}")
            c += n
        lines.append("".join(cm))
    lines += [" & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        lines.append("\\midrule" if r == ["---"] else " & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    (out / f"{name}.tex").write_text("\n".join(lines))
    pd.DataFrame(rows, columns=None).to_csv(out / f"{name}.csv", index=False, header=False)
    n_missing = sum(c == MISSING for r in rows for c in r)
    print(f"wrote {name}.tex ({len(rows)} rows, {n_missing} missing cells)")
    if n_missing:
        WARN.append(f"{name}: {n_missing} missing cells")


def need(strict: bool, what: str):
    WARN.append(f"missing: {what}")
    if strict:
        raise SystemExit(f"--strict: missing {what}")


# ---------------------------------------------------------------- tables
def t_ndcg(root, out, strict, n):
    cs = ctxs(root, "test")
    order = ["oracle", "jev", "qwen3", "l1_order", "clm", "random"]
    rows = [[NAMES[m]] + [f3(c.per_user(m)["ndcg@10"].mean()) if c.has(m) else MISSING for c in cs.values()]
            for m in order]
    if not any(c.has("jev") for c in cs.values()):
        return need(strict, "test runs for ndcg table")
    write(out, "tab_ndcg", ["Model"] + list(cs), rows, "@{}l" + "c" * len(cs))
    comps = [("Jev $-$ embedding order", "jev", "l1_order"),
             ("Qwen3-Reranker $-$ embedding order", "qwen3", "l1_order"),
             ("Jev $-$ Qwen3-Reranker", "jev", "qwen3")]
    cell = {}
    for lab, c in cs.items():
        have = [(nm, a, b) for nm, a, b in comps if c.has(a) and c.has(b)]
        if not have:
            continue
        vecs = [(nm, R.align({a: c.per_user(a), b: c.per_user(b)}, "ndcg@10"), a, b) for nm, a, b in have]
        res = R.diff_table([(nm, v[a], v[b]) for nm, v, a, b in vecs], n)   # Holm across the comparisons in this setting
        for (nm, v, a, b), r in zip(vecs, res):
            NUMBERS[f"ndcg_diff[{a}-{b}][{lab}]"] = r
            cell[(nm, lab)] = r
    drows = [[nm] + [fd(cell.get((nm, lab))) for lab in cs] for nm, _, _ in comps]
    write(out, "tab_ndcg_delta", ["Difference in NDCG@10 [95\\% CI]"] + list(cs), drows, "@{}l" + "c" * len(cs))


def t_composition(root, out, strict, n):
    cs = ctxs(root, "test")
    models = ["oracle", "jev", "qwen3", "l1_order", "clm"]
    rows = []
    for lab, c in cs.items():
        have = [m for m in models if c.has(m)]
        if not have:
            continue
        rows.append(["---"]) if rows else None
        rows.append([f"\\textit{{{lab}}}", "", "", "", "", ""])
        for m in have:
            comp = R.composition(c.orders(m), c.labels, 10, ("E", "S", "C", "I"))
            NUMBERS[f"composition[{m}][{lab}]"] = comp
            rows.append([NAMES[m]] + [f"{comp[k]:.2f}" if (k != "C" or c.ds == "esci") else "--"
                                      for k in ("E", "S", "C", "I", "U")])
    if not rows:
        return need(strict, "test runs for composition table")
    write(out, "tab_composition", ["Ranking", "Exact", "Subst./Partial", "Compl.", "Irrel.", "Unlabelled"],
          rows, "@{}lccccc")


def t_exact_subset(root, out, strict, n):
    rows = []
    for lab, c in ctxs(root, "test").items():
        if c.ds != "esci" or not c.has("jev") or not c.has("qwen3") or not c.has("l1_order"):
            continue
        keep = sorted(R.has_label(c.labels, "E") & set(c.per_user("jev").user_id))
        sub = {m: c.per_user(m)[c.per_user(m).user_id.isin(keep)] for m in ("l1_order", "qwen3", "jev")}
        v = R.align(sub, "p@10")
        r = R.diff_ci(v["jev"], v["qwen3"], n)
        NUMBERS[f"exact_subset[{lab}]"] = {"n": len(keep), **{m: float(x.mean()) for m, x in v.items()}, "jev-qwen3": r}
        rows.append([lab, str(len(keep)), f3(v["l1_order"].mean()), f3(v["qwen3"].mean()),
                     f3(v["jev"].mean()), fd(r)])
    if not rows:
        return need(strict, "ESCI test runs for exact_subset table")
    write(out, "tab_exact_subset", ["Setting", "Queries", "Embedding", "Qwen3-0.6B", "Jev", "Jev $-$ Qwen3 [95\\% CI]"],
          rows, "@{}lrcccc")


def t_baselines(root, out, strict, n):
    cs = ctxs(root, "test")
    models = ["l1_order", "qwen3", "qwen3_4b", "qwen3_8b", "llm_qwen3_8b", "bge_m3", "jev"]
    models = [m for m in models if any(c.has(m) for c in cs.values())]      # a baseline that was not run at all is omitted, not shown as n/a
    added = [m for m in models if m not in ("l1_order", "jev", "qwen3")]
    if not any(c.has(m) for c in cs.values() for m in added):
        return need(strict, "test runs of the added baselines")
    rows = []
    for m in models:
        row = [NAMES[m]]
        for c in cs.values():
            for col in ("p@10", "p@5", "ndcg@10"):
                row.append(f3(c.per_user(m)[col].mean()) if c.has(m) else MISSING)
        rows.append(row)
    write(out, "tab_baselines", ["Model"] + ["P@10", "P@5", "NDCG"] * len(cs), rows,
          "@{}l" + "ccc" * len(cs), groups=[(g, 3) for g in cs])
    drows = [[NAMES[m]] + [MISSING] * len(cs) for m in models if m not in ("l1_order", "jev")]
    # Holm across the compared baselines within each setting
    for j, (lab, c) in enumerate(cs.items()):
        pairs = [(m, c.per_user("jev"), c.per_user(m)) for m in models if m not in ("l1_order", "jev") and c.has(m) and c.has("jev")]
        if not pairs:
            continue
        v = [(m, R.align({"a": a, "b": b}, "p@10")) for m, a, b in pairs]
        res = R.diff_table([(m, x["a"], x["b"]) for m, x in v], n)
        for r in res:
            NUMBERS[f"jev_minus[{r['name']}][{lab}]"] = r
            i = [x for x in models if x not in ("l1_order", "jev")].index(r["name"])
            drows[i][1 + j] = fd(r)
    write(out, "tab_baselines_delta", ["Jev $-$ model, P@10 [95\\% CI]"] + list(cs), drows, "@{}l" + "c" * len(cs))


VARIANTS = [("Main wording (esci\\_v1)", ""), ("Same text for both models (v1p)", "@esci_v1p"),
            ("Paraphrase (v2)", "@esci_v2"), ("Terse, question only (v3)", "@esci_v3"),
            ("Broad relevance (v4, control)", "@esci_v4"), ("Graded: exact + substitute (v1p+v5)", "@esci_v1p+esci_v5")]


def t_ablation(root, out, strict, n):
    cs = {k: v for k, v in ctxs(root, "valid").items() if k in ("ESCI judged", "WANDS")}
    rows = []
    for lab, c in cs.items():
        if c.has("l1_order"):
            NUMBERS[f"ablation_l1[{lab}]"] = float(c.per_user("l1_order")["p@10"].mean())
    if not cs or not any(c.has("jev") for c in cs.values()):
        return need(strict, "validation runs for the prompt ablation")
    res = {}
    for lab, c in cs.items():
        pairs, means = [], {}
        for vlabel, suffix in VARIANTS:
            j, q = c.runs.get("jev" + suffix), c.runs.get("qwen3" + suffix)
            if j is None or q is None:
                continue
            v = R.align({"j": j["per_user"], "q": q["per_user"]}, "p@10")
            pairs.append((vlabel, v["j"], v["q"]))
            means[vlabel] = (float(v["j"].mean()), float(v["q"].mean()))
        for r in R.diff_table(pairs, n):                                   # Holm across the wordings in this dataset
            res[(lab, r["name"])] = {**r, "jev": means[r["name"]][0], "qwen3": means[r["name"]][1]}
            NUMBERS[f"ablation[{r['name']}][{lab}]"] = res[(lab, r["name"])]
    for vlabel, _ in VARIANTS:
        row = [vlabel]
        for lab in cs:
            r = res.get((lab, vlabel))
            row += [MISSING] * 3 if r is None else [f3(r["jev"]), f3(r["qwen3"]), fd(r)]
        rows.append(row)
    ref = ["Embedding order"] + sum(([f3(c.per_user("l1_order")["p@10"].mean()) if c.has("l1_order") else MISSING,
                                      "", ""] for c in cs.values()), [])
    rows += [["---"], ref]
    write(out, "tab_ablation", ["Wording"] + ["Jev", "Qwen3", "Jev $-$ Qwen3"] * len(cs), rows,
          "@{}l" + "ccc" * len(cs), groups=[(f"{g} (validation, P@10)", 3) for g in cs])


def t_clm(root, out, strict, n):
    c = ctxs(root, "valid").get("ESCI judged")
    names = [("l1_order", "Embedding order"), ("random", "Random"), ("clm", "CLM, list scoring (as on test)"),
             ("clm#shuffle1", "CLM, list scoring, candidates shuffled (seed 1)"),
             ("clm#shuffle2", "CLM, list scoring, candidates shuffled (seed 2)"),
             ("clm_pointwise", "CLM, one product per call, yes/no")]
    if c is None or not c.has("clm_pointwise"):
        return need(strict, "validation runs for CLM interface controls")
    rows = []
    for m, label in names:
        rows.append([label] + [f3(c.per_user(m)[col].mean()) if c.has(m) else MISSING
                               for col in ("p@10", "p@5", "ndcg@10")])
    write(out, "tab_clm_controls", ["Variant", "P@10", "P@5", "NDCG@10"], rows, "@{}lccc")


TABLES = {"ndcg": t_ndcg, "composition": t_composition, "exact_subset": t_exact_subset,
          "baselines": t_baselines, "ablation": t_ablation, "clm_controls": t_clm}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("tables", nargs="+", choices=[*TABLES, "all"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="paper_generated")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--resamples", type=int, default=10_000)
    a = ap.parse_args(argv)
    root, out = Path(a.root), Path(a.out)
    for t in (TABLES if "all" in a.tables else a.tables):
        TABLES[t](root, out, a.strict, a.resamples)
    out.mkdir(parents=True, exist_ok=True)
    nf = out / "numbers.json"
    old = json.loads(nf.read_text()) if nf.exists() else {}
    old.update(NUMBERS)
    nf.write_text(json.dumps(old, indent=1, default=float))
    if WARN:
        print("\nWARNINGS:\n  " + "\n  ".join(WARN))


if __name__ == "__main__":
    main()
