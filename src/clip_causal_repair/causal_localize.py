from __future__ import annotations
import argparse
from pathlib import Path
import csv
import torch
from tqdm import tqdm

from .clip_loader import load_clip, encode_text, encode_images
from .waterbirds import get_waterbirds, make_loader


def ablate_head_outputs(block, head_idx: int):
    """Zero out one attention head's output (per-token) inside a CLIP ViT block."""
    attn = block.attn
    orig_forward = attn.forward

    def patched(x, **kwargs):
        # WARNING: This is a simplified illustrative patch for a scan.
        B, N, C = x.shape
        qkv = attn.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(B, N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(B, N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(B, N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
        attn_scores = (q @ k.transpose(-2, -1)) * attn.scale
        attn_probs = attn_scores.softmax(dim=-1)
        out = attn_probs @ v  # [B, heads, N, head_dim]
        out[:, head_idx, :, :] = 0.0
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = attn.proj(out)
        return out

    attn.forward = patched
    return orig_forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=str, default="ViT-B-32")
    ap.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    ap.add_argument("--root", type=str, default="data/wilds")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--subset-size", type=int, default=256, help="eval on a small subset for speed")
    ap.add_argument("--out", type=str, default="outputs/scan_effects.csv")
    args = ap.parse_args()

    model, preprocess, tokenizer, device = load_clip(args.arch, args.pretrained)
    prompts = ["a photo of a landbird", "a photo of a waterbird"]
    text_feats = encode_text(model, tokenizer, prompts, device)

    subset, _ = get_waterbirds(args.root, "test", transform=preprocess)
    if args.subset_size and len(subset) > args.subset_size:
        subset = torch.utils.data.Subset(subset, list(range(args.subset_size)))
    loader = make_loader(subset, batch_size=args.batch_size, shuffle=False)

    # baseline logits
    base_logits, base_targets = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            feats = encode_images(model, x, device)
            base_logits.append((100.0 * feats @ text_feats.T).cpu())
            base_targets.append(y)
    base_logits = torch.cat(base_logits)
    base_targets = torch.cat(base_targets)

    # scan each block/head
    vis = model.visual
    blocks = vis.transformer.resblocks
    num_blocks = len(blocks)
    num_heads = blocks[0].attn.num_heads

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block", "head", "delta_margin_mean"])

        for b in range(num_blocks):
            block = blocks[b]
            for h in range(num_heads):
                orig = ablate_head_outputs(block, h)
                ab_logits = []
                with torch.no_grad():
                    for x, y, _ in loader:
                        x = x.to(device)
                        feats = encode_images(model, x, device)
                        ab_logits.append((100.0 * feats @ text_feats.T).cpu())
                # restore
                block.attn.forward = orig

                ab_logits = torch.cat(ab_logits)
                margin_base = base_logits[:, 1] - base_logits[:, 0]
                margin_ab = ab_logits[:, 1] - ab_logits[:, 0]
                delta = (margin_ab - margin_base).mean().item()
                w.writerow([b, h, f"{delta:.6f}"])

    print(f"✅ Saved scan to {args.out}")


if __name__ == "__main__":
    main()
