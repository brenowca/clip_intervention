# Causal Repair of a Spurious Background Concept in CLIP (Waterbirds)

**TL;DR**: CLIP sometimes “cheats” on Waterbirds by using **background water/land** as a shortcut. This repo aims to:
1) **Elicit** the failure clearly (zero-shot CLIP on Waterbirds),
2) **Causally localize** the internal components that carry the water-background signal,
3) Apply a **surgical edit** that minimally targets that signal, and
4) **Evaluate** whether the fix helps the failure case **without breaking** performance elsewhere (overall & worst-group accuracy).

---

## Why this matters
Zero-shot CLIP is widely used. But robustness issues from **spurious correlations** (like backgrounds) can cause brittle behavior. We take an *applied interpretability* approach: localize → minimally edit → measure side-effects with credible, small-footprint evals.

---

## What you’ll find here
- **Zero-shot baseline** for CLIP on Waterbirds (overall & worst-group metrics)
- **Causal localization scan** (ablating/patching visual heads / MLP blocks → effect on logits)
- **Feature-level surgical edit** (project-out a learned “water-background direction” in the image encoder)
- **Side-effect budgeting** (compare before vs. after on groups and aggregate metrics)

> ⚠️ This is a small, time-bounded project scaffold. It aims for clarity & minimalism over framework bloat.

---

## Setup
### 0) Requirements
- Python ≥ 3.10
- CUDA GPU recommended (but CPU works for small runs)

### 1) Create & activate a virtualenv
```bash
bash scripts/setup_env.sh
```
This will create `.venv` and install the package in editable mode.

### 2) Download Waterbirds (via `wilds`)
```bash
bash scripts/download_data.sh
```
Data will go under `./data/wilds/` by default.

> If download fails (e.g., firewalls), manually place the WILDS Waterbirds data under `data/wilds/` and re-run steps.

---

## Quickstart
### Baseline: Zero-shot evaluation
```bash
bash scripts/eval_zero_shot.sh
```
Outputs: overall accuracy, per-group accuracy, and **worst-group accuracy**.

### Causal localization (scan)
```bash
bash scripts/scan_layers.sh
```
Outputs a CSV with the **logit effect** from ablating specific attention heads / MLP blocks.

### Fit a background direction & apply a surgical edit
```bash
# 1) learn a "water vs land background" direction from CLIP image features
bash scripts/fit_direction.sh

# 2) apply a projection-out edit at a chosen layer and re-evaluate
bash scripts/apply_edit_and_eval.sh
```

## Per-layer edits
The editing API also allows targeting internal layers. The `scripts/edit_directional.py` script can
fit a background direction and apply it at any module:

```bash
# 1) fit the direction
python scripts/edit_directional.py fit-direction --arch ViT-B-32 --root data/wilds \
    --out edits/water_direction.npy

# 2) sweep α when applying to a specific layer
for a in 0.0 0.05 0.1 0.2; do
  python scripts/edit_directional.py apply --direction edits/water_direction.npy \
    --layer "visual.transformer.resblocks.6.mlp" --alpha $a \
    --out outputs/layer6_a${a}.csv
done
```

Higher `α` removes more of the spurious component but risks **majority collapse**.
Hooks on an `nn.Parameter` (e.g., OpenCLIP's `visual.proj`) must be wrapped in a
`LinearProxy` module before registering edits.

See the example config at `configs/per_layer_edit.yaml` and the helper scripts in
`scripts/` for end-to-end runs.

---

## Design sketch
- **Model**: OpenCLIP ViT-B/32 (default `laion2b_s34b_b79k`, configurable)
- **Data**: WILDS Waterbirds (labels for bird type; metadata with background attr)
- **Baseline**: Zero-shot with simple prompts: "a photo of a waterbird", "a photo of a landbird"
- **Localization**: Forward hooks to selectively ablate head/block outputs → measure Δ(logit margin)
- **Surgical edit**: Estimate a background direction `u` from features; at a chosen layer, project features as `(I - uuᵀ)x` (tiny, targeted)
- **Side effects**: Report overall accuracy, per-group metrics, and **worst-group** accuracy before/after.

---

## Repro & logging
- Deterministic seeds where possible (`--seed 123`)
- CSV outputs under `./outputs/`
- Minimal dependencies; no external logging service required

---

## Ethical & licensing notes
- **Datasets may contain biases**. Use responsibly.
- This repo is MIT-licensed (see `LICENSE`). OpenCLIP and WILDS are separate projects with their own licenses.

---

## Citation
If this scaffold was useful, please cite the repo and relevant WILDS/CLIP works as appropriate.
