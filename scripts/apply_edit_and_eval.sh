#!/usr/bin/env bash
set -euo pipefail
. .venv/bin/activate || true

python -m clip_causal_repair.feature_edit apply \
  --arch ViT-B-32 \
  --pretrained laion2b_s34b_b79k \
  --root data/wilds \
  --direction edits/water_direction.npy \
  --layer visual \
  --alpha 1.0 \
  --eval-split test \
  --out outputs/after_edit.csv
