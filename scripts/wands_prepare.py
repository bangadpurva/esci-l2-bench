"""WANDS step 1: queries with an Exact label, split into validation and test; labels; product text.

    python scripts/wands_prepare.py            # 80 validation queries, the rest test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recl2bench import wands  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/wands/raw")
    ap.add_argument("--out", default="data/wands/processed")
    ap.add_argument("--n-valid", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20261003)
    a = ap.parse_args(argv)
    rep = wands.prepare(a.raw, a.out, a.n_valid, a.seed)
    (Path(a.out) / "prepare_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return rep


if __name__ == "__main__":
    main()
