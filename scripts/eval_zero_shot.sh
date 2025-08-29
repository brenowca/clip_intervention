#!/usr/bin/env bash
set -euo pipefail
. .venv/bin/activate || true
python -m clip_causal_repair.zero_shot   --arch ViT-B-32   --pretrained laion2b_s34b_b79k   --root data/wilds   --batch-size 64   --split test   --out outputs/zero_shot.csv
