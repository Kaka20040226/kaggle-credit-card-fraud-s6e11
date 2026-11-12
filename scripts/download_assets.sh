#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT/data"
PUBLIC_DIR="$ROOT/public_inputs/part12"

mkdir -p "$DATA_DIR" "$PUBLIC_DIR"

kaggle competitions download playground-series-s6e11 \
  --path "$DATA_DIR" --unzip --quiet

for file in \
  wide_lm_stack_nested.npy \
  wide_lm_stack_test.npy \
  part9_test.npy \
  wide_lm_stack_members.csv
do
  kaggle datasets download goodpjw2008/s6e11-fraud-series-artifacts \
    --file "$file" --path "$PUBLIC_DIR" --unzip --quiet
done

kaggle datasets download megayak/s6e11-target-first-lm \
  --file lm_oof.npy --path "$PUBLIC_DIR" --unzip --quiet

echo "Competition data and public OOF artifacts are ready."
