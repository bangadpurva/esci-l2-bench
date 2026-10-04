#!/usr/bin/env bash
# ESCI benchmark: query -> content ANN top-100 -> L2 rerankers. One command per stage.
#
#   bash scripts/run_esci.sh <stage>
#
# Environment (defaults in brackets):
#   SPLIT        test | valid                         [test]
#   SETTING      retrieved (L1 top-100) | judged       [retrieved]
#   CONCURRENCY  parallel API calls for clef / jev     [model config]
#
# Stages (STOP = send the output for review before continuing):
#   check        GPU, packages, tests
#   download     ESCI parquet files from the official repo (sha256-checked)
#   prepare      sample 1,000 test / 500 valid queries, labels, product text      STOP
#   l1-dryrun    embed catalog + queries, top-100, ANN gate, diagnostics (GPU)     STOP
#   l1-freeze    freeze both list settings (asks you to type FREEZE)
#   baselines    l1_order, random, oracle on SPLIT/SETTING
#   qwen3        Qwen3-Reranker cross-encoder (GPU)
#   clef-smoke | jev-smoke   20 validation queries                                 STOP
#   clef | jev   decision-model APIs
#   clm-setup    pip install contrastive-lm + vllm (GPU box)
#   clm-serve    start vLLM (Qwen3-8B pooling) in the background, wait until ready
#   clm          CLM-v0.1-8B (needs clm-serve)
#   report       results table -> results/esci/<split>_<setting>.md
#   backup       archive lists, labels, runs, results, logs (not raw data or embeddings)
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE="${1:-}"
[ -n "$STAGE" ] || { sed -n '2,30p' "$0"; exit 1; }
if [ -f .env ]; then set -a; . ./.env; set +a; fi
SPLIT="${SPLIT:-test}"
SETTING="${SETTING:-retrieved}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
mkdir -p logs
LOG="logs/esci_${STAGE}_${SPLIT}_${SETTING}_$(date +%Y%m%dT%H%M%S).log"
HW="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
HW="${HW:-$(uname -m)} $(hostname)"
run() { echo "+ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }
read -r -a EXTRA <<< "${ESCI_ARGS:-}"
SC=(python scripts/esci_score.py --split "$SPLIT" --setting "$SETTING" --hardware "$HW"
    ${CONCURRENCY:+--concurrency "$CONCURRENCY"} ${EXTRA[@]+"${EXTRA[@]}"})

echo "stage=$STAGE SPLIT=$SPLIT SETTING=$SETTING log=$LOG" | tee -a "$LOG"
case "$STAGE" in
  check)
    run nvidia-smi || true
    run python -m pytest -q ;;
  download)
    run bash scripts/esci_download.sh data/esci/raw ;;
  prepare)
    run python scripts/esci_prepare.py ${EXTRA[@]+"${EXTRA[@]}"} ;;
  l1-dryrun)
    run python scripts/esci_l1.py ${EXTRA[@]+"${EXTRA[@]}"} ;;
  l1-freeze)
    echo "Freezing ESCI lists (retrieved + judged, valid + test). This cannot be undone."
    if [ "${FORCE:-}" != "1" ]; then read -r -p "Type FREEZE to continue: " ans; [ "$ans" = "FREEZE" ] || exit 1; fi
    run python scripts/esci_l1.py --freeze ${EXTRA[@]+"${EXTRA[@]}"} ;;
  baselines)
    for m in l1_order random oracle; do run "${SC[@]}" --model "$m"; done ;;
  qwen3|clm)
    run "${SC[@]}" --model "$STAGE" ;;
  clef|jev)
    run "${SC[@]}" --model "$STAGE" ;;
  clef-smoke|jev-smoke)
    m="${STAGE%-smoke}"
    run python scripts/esci_score.py --split valid --setting "$SETTING" --hardware "$HW" --model "$m" --limit-users 20 \
        ${CONCURRENCY:+--concurrency "$CONCURRENCY"} ;;
  clm-setup)
    run python -m pip install -q contrastive-lm vllm
    run clm-download ;;          # the 75 MB CLM-v0.1-8B head; Engine does not fetch it itself
  clm-serve)
    if curl -sf http://127.0.0.1:8090/v1/models >/dev/null; then echo "vLLM already serving on :8090"; exit 0; fi
    nohup vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b --runner pooling \
        --max-model-len 2048 --port 8090 > logs/vllm.log 2>&1 &
    echo "vLLM starting (log: logs/vllm.log); first start downloads ~16 GB"
    for i in $(seq 1 180); do
      if curl -sf http://127.0.0.1:8090/v1/models >/dev/null; then echo "ready after $((i*10))s"; exit 0; fi
      sleep 10
    done
    echo "vLLM not ready after 30 min; see logs/vllm.log"; exit 1 ;;
  report)
    run python scripts/esci_score.py --report --split "$SPLIT" --setting "$SETTING" ;;
  backup)
    mkdir -p backups
    OUT="backups/esci_$(date +%Y%m%dT%H%M%S).tar.gz"
    run tar czf "$OUT" --exclude='*.npy' --exclude='products.parquet' \
        data/esci/processed data/esci/pools $( [ -d runs/esci ] && echo runs/esci ) \
        $( [ -d results/esci ] && echo results/esci ) logs
    ls -lh "$OUT"; echo "Copy it off this machine before destroying the instance." ;;
  *)
    echo "unknown stage: $STAGE"; sed -n '2,30p' "$0"; exit 1 ;;
esac
echo "done: $STAGE (log: $LOG)"
