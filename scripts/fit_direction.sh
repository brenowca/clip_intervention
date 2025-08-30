#!/usr/bin/env bash
set -euo pipefail
. .venv/bin/activate || true

python -m clip_causal_repair.feature_edit fit-direction \
  --arch ViT-B-32 \
  --pretrained laion2b_s34b_b79k \
  --backend openclip \
  --root data/wilds \
  --split train \
  --max-samples 2000 \
  --out edits/water_direction.npy