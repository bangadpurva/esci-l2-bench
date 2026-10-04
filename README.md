# esci-l2-bench

L2 rerankers for product search on Amazon ESCI: query → Qwen3-Embedding content ANN
top-100 → rerank with Qwen3-Reranker-0.6B, Jev, Clef Flash and CLM-v0.1-8B, compared on
P@10 / P@5, latency and cost. Design and decision rule: [`PREREGISTRATION.md`](PREREGISTRATION.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[models,api,dev]"
cp .env.example .env          # then fill in the keys you have
python -m pytest -q
```

## Running

Every stage is Every stage is `bash scripts/run_esci.sh <stage>`; output is saved to `logs/`.

## Where to run what

| Stages | Machine | Why |
|---|---|---|
| download, prepare | anywhere (about 2 GB RAM) | CPU only |
| l1-dryrun, l1-freeze | **GPU** (A40 or similar) | embeds 1.2M products; hours on a laptop |
| qwen3 | GPU | cross-encoder |
| clm-setup, clm-serve, clm | GPU, 24 GB+ (A40 is fine) | vLLM serves Qwen3-8B |
| clef, jev | anywhere with the API keys in `.env` | API calls |
| baselines, report | anywhere | CPU only |

Freeze on **one** machine only. Scoring elsewhere needs `data/esci/processed` (including
`products.parquet`) and `data/esci/pools` copied across; hashes are checked on load.

## Order

```bash
bash scripts/run_esci.sh download
bash scripts/run_esci.sh prepare                    # STOP: send prepare_report.json
bash scripts/run_esci.sh l1-dryrun                  # GPU. STOP: send the JSON report
bash scripts/run_esci.sh l1-freeze                  # irreversible, asks for FREEZE
bash scripts/run_esci.sh baselines
bash scripts/run_esci.sh qwen3
bash scripts/run_esci.sh clm-setup
bash scripts/run_esci.sh clm-serve
bash scripts/run_esci.sh clm
bash scripts/run_esci.sh jev-smoke                  # STOP
bash scripts/run_esci.sh jev
bash scripts/run_esci.sh clef-smoke                 # STOP
bash scripts/run_esci.sh clef
bash scripts/run_esci.sh report
bash scripts/run_esci.sh backup
```

Then the same model stages and `report` with `SETTING=judged` (the clean comparison).
Validation runs (for fusion later): prefix any model stage with `SPLIT=valid`.

## Rough cost and time (1,000 test queries × 100 candidates = 100K pairs)

| Model | Time | Cost |
|---|---|---|
| Embedding the catalog (once) | ~30–60 min on an A40 | GPU time |
| Qwen3-Reranker | ~10–30 min on an A40 | GPU time |
| CLM | minutes (one call per query) + ~16 GB model download | GPU time |
| Jev | ~2.5 h (API caps ~12 req/s) | ~$1–2 |
| Clef Flash | depends on Cloudflare limits | ~$5–8 |

The judged setting is about 6× smaller (median 16 candidates per query).

## Second dataset: WANDS (Wayfair)

Same pipeline, near-complete labels (about 220 labelled products per query). Design:
[`PREREGISTRATION_WANDS.md`](PREREGISTRATION_WANDS.md). Prefix every stage with `DATASET=wands`;
only the retrieved setting is scored.

```bash
DATASET=wands bash scripts/run_esci.sh download
DATASET=wands bash scripts/run_esci.sh prepare
DATASET=wands bash scripts/run_esci.sh l1-dryrun     # GPU, a few minutes (43K products). STOP
DATASET=wands bash scripts/run_esci.sh l1-freeze
DATASET=wands bash scripts/run_esci.sh baselines
DATASET=wands bash scripts/run_esci.sh qwen3
DATASET=wands bash scripts/run_esci.sh jev           # also clef, clm (after clm-serve)
DATASET=wands bash scripts/run_esci.sh report
```

299 test queries × 100 candidates = about 30K pairs: Jev about 45 min and $0.75.
