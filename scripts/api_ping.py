"""One raw call to a decision-model API; prints HTTP status and the response body.

    python scripts/api_ping.py --model jev
    python scripts/api_ping.py --model clef

Reads keys from .env. Never prints the key itself.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from recl2bench.rerankers.decision import YesNoQuestion, make_backend  # noqa: E402


def load_env():
    env = root / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["clef", "jev"], default="jev")
    a = ap.parse_args(argv)
    load_env()
    cfg = yaml.safe_load((root / f"configs/models/{a.model}.yaml").read_text())
    needed = [cfg.get("api_key_env")] + ([cfg["account_id_env"]] if "account_id_env" in cfg else [])
    for k in needed:
        v = os.environ.get(k, "")
        print(f"{k}: {'set, ' + str(len(v)) + ' chars' if v else 'MISSING'}")
    if not all(os.environ.get(k) for k in needed):
        sys.exit("fill the missing keys in .env (no quotes, no spaces)")
    be = make_backend(cfg["backend"], **cfg)
    q = YesNoQuestion("ping", "Would this shopper buy guitar strings?", "yes", "no")
    body = {"model": be.model, "state": "Shopper bought an acoustic guitar and a capo.",
            "questions": {"q0": {"type": "noul", "instructions": be.instructions(q)}}}
    status, data = be._post(be.url, {"Authorization": f"Bearer {be.token}", "Content-Type": "application/json"}, body)
    shown = be.url
    for k in needed:
        shown = shown.replace(os.environ[k], "<" + k + ">")
    print(f"POST {shown}\nHTTP {status}")
    print(json.dumps(data, indent=2)[:3000])


if __name__ == "__main__":
    main()
