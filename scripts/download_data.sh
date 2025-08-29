#!/usr/bin/env bash
set -euo pipefail
. .venv/bin/activate || true
python -m clip_causal_repair.waterbirds --download --root data/wilds
