#!/usr/bin/env bash
# Revision experiments (PREREGISTRATION.md, Amendment 5). Writes to runs/<dataset>_rev; the frozen lists and the
# main runs in runs/<dataset> are never touched.
#
#   bash scripts/run_revision.sh <phase>
#
#   baselines-valid   l1_order / random / oracle on the validation queries (CPU)
#   ablation          prompt wording + graded formulation, Qwen3-0.6B and Jev, VALIDATION only (run fusion-inputs first:
#                       it provides the main-wording column)
#                       ESCI judged (500 q) and WANDS (80 q).  Jev: ~170K calls, ~$4, ~4 h at 12 calls/s
#   fusion-inputs     Qwen3-0.6B and Jev on validation (all three settings) so fusion weights can be chosen
#                       (the test runs already exist from the main study).  Jev: ~58K calls
#   clm-controls      CLM order-shuffle (seeds 1, 2) and pointwise yes/no, ESCI judged validation (needs clm-serve)
#   test-baselines    Qwen3-Reranker-4B / 8B, prompted Qwen3-8B, bge-reranker-v2-m3 on the TEST lists (GPU 48 GB)
#   tables            revision_cost + revision_fusion + revision_tables  ->  paper_generated/
#   all               every phase in order
#
# MODELS="jev" or MODELS="qwen3" limits ablation / fusion-inputs to one model (default: both), so Jev (network-bound) can
# run on a laptop while Qwen3 (needs a GPU to be fast) runs on a pod. Merge the two runs_rev directories before `tables`.
#
# Needs the same environment as run_esci.sh (.env with TYPESAFE_API_KEY, the frozen lists, GPU for local models).
set -euo pipefail
cd "$(dirname "$0")/.."
PHASE="${1:-}"
[ -n "$PHASE" ] || { sed -n '2,23p' "$0"; exit 1; }
go() { # DATASET SPLIT SETTING STAGE [ESCI_ARGS]
  echo "=== $1 $2 $3 $4 ${5:-}"
  DATASET="$1" SPLIT="$2" SETTING="$3" RUNS_DIR="runs/$1_rev" ESCI_ARGS="${5:-}" bash scripts/run_esci.sh "$4"
}
SETTINGS=("esci judged" "esci retrieved" "wands retrieved")
TEMPLATES=(esci_v1p esci_v2 esci_v3 esci_v4)

baselines_valid() { for s in "${SETTINGS[@]}"; do set -- $s; go "$1" valid "$2" baselines; done; }
ablation() {
  for s in "esci judged" "wands retrieved"; do set -- $s
    for m in ${MODELS:-qwen3 jev}; do
      # (the main-wording validation runs come from fusion-inputs, which `all` runs first)
      for t in "${TEMPLATES[@]}"; do go "$1" valid "$2" "$m" "--template $t"; done
      go "$1" valid "$2" "$m" "--template esci_v1p --graded-with esci_v5"  # graded: P(exact) + P(acceptable)
    done
  done
}
fusion_inputs() { for s in "${SETTINGS[@]}"; do set -- $s; for m in ${MODELS:-qwen3 jev}; do go "$1" valid "$2" "$m"; done; done; }
clm_controls() {
  go esci valid judged clm
  go esci valid judged clm "--clm-shuffle-seed 1"
  go esci valid judged clm "--clm-shuffle-seed 2"
  go esci valid judged clm_pointwise
}
test_baselines() { for s in "${SETTINGS[@]}"; do set -- $s; for m in qwen3_4b qwen3_8b llm_qwen3_8b bge_m3; do go "$1" test "$2" "$m"; done; done; }
tables() {
  python scripts/revision_cost.py --out paper_generated
  python scripts/revision_fusion.py --models jev qwen3 --out paper_generated
  python scripts/revision_tables.py all --out paper_generated
  python scripts/check_slots.py paper_generated || true
}
case "$PHASE" in
  baselines-valid) baselines_valid ;;
  ablation) ablation ;;
  fusion-inputs) fusion_inputs ;;
  clm-controls) clm_controls ;;
  test-baselines) test_baselines ;;
  tables) tables ;;
  all) baselines_valid; fusion_inputs; ablation; clm_controls; test_baselines; tables ;;
  *) echo "unknown phase: $PHASE"; sed -n '2,23p' "$0"; exit 1 ;;
esac
