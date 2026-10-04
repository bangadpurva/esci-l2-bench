#!/usr/bin/env bash
# Wayfair WANDS (github.com/wayfair/WANDS), checked against the hashes of the files we
# profiled on 2026-10-03.
#   bash scripts/wands_download.sh [out_dir]
set -euo pipefail
OUT="${1:-data/wands/raw}"
mkdir -p "$OUT"
BASE=https://raw.githubusercontent.com/wayfair/WANDS/main/dataset
declare -A SHA=(
  [product.csv]=d993926254572e6eba96c8fd87cc549a17fb91ad3748308036eee4cf92b10ac6
  [query.csv]=63b61660560fecc33ec490804c7e2b81402ee3e7c31a9cbb5e03736639f68e95
  [label.csv]=c11fe81ad62f17f56f316b0ec9630ebe8fbe1393578cb0ca4f05c17253a180ef
)
for f in "${!SHA[@]}"; do
  if [ ! -s "$OUT/$f" ]; then curl -fL --retry 3 -o "$OUT/$f" "$BASE/$f"; fi
  got=$( (sha256sum "$OUT/$f" 2>/dev/null || shasum -a 256 "$OUT/$f") | cut -d' ' -f1)
  [ "$got" = "${SHA[$f]}" ] || { echo "checksum mismatch for $f"; exit 1; }
  echo "ok $f"
done
ls -lh "$OUT"
