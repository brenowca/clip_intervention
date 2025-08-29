#!/usr/bin/env bash
set -euo pipefail
. .venv/bin/activate || true
python -m clip_causal_repair.causal_localize   --arch ViT-B-32   --pretrained laion2b_s34b_b79k   --root data/wilds   --batch-size 32   --subset-size 256   --out outputs/scan_effects.csv
