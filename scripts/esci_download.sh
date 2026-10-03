#!/usr/bin/env bash
# Amazon ESCI (Shopping Queries Dataset) from the official repo's Git LFS storage,
# checked against the LFS pointer hashes.
#   bash scripts/esci_download.sh [out_dir]
set -euo pipefail
OUT="${1:-data/esci/raw}"
mkdir -p "$OUT"
BASE=https://media.githubusercontent.com/media/amazon-science/esci-data/main/shopping_queries_dataset
declare -A SHA=(
  [shopping_queries_dataset_examples.parquet]=4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a
  [shopping_queries_dataset_products.parquet]=25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265
)
for f in "${!SHA[@]}"; do
  if [ ! -s "$OUT/$f" ]; then curl -fL --retry 3 -o "$OUT/$f" "$BASE/$f"; fi
  got=$( (sha256sum "$OUT/$f" 2>/dev/null || shasum -a 256 "$OUT/$f") | cut -d' ' -f1)
  [ "$got" = "${SHA[$f]}" ] || { echo "checksum mismatch for $f"; exit 1; }
  echo "ok $f"
done
ls -lh "$OUT"
