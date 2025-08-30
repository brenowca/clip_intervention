from __future__ import annotations
import argparse
from pathlib import Path
import csv
import torch

from .clip_loader import load_clip, encode_text, encode_images
from .waterbirds import get_waterbirds, make_loader
from .metrics import aggregate_accuracy_metrics


def ablate_head_outputs(block: torch.nn.Module, head_idx: int):
    """
    Wraps a CLIP ViT encoder block so that the output of the head
    with index `head_idx` is zeroed out for every token.

    Parameters
    ----------
    block : nn.Module
        The encoder block from `CLIPVisionModelWithProjection`.  
        It must expose `block.attention` (the Multi‑Head Self‑Attention
        module) and `block.num_heads` (the number of heads).
    head_idx : int
        Zero‑based index of the head to ablate (must be < block.num_heads).

    Returns
    -------
    nn.Module
        The original attention head's forward method of the block instance, i.e. `block.attn.forward`.
    """
    attn = block.attn
    orig_forward = attn.forward

    def patched_attn_forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        """
        Calls the original attention forward, then zeros the selected
        head's contribution before projecting back to `embed_dim`.
        """
        # Original attention returns a tuple: (attn_output, attn_weights)
        attn_output, attn_weights = orig_forward(hidden_states, *args, **kwargs)

        # Reshape to isolate heads: [batch, seq_len, num_heads, head_dim]
        batch, seq_len, embed_dim = attn_output.shape
        head_dim = embed_dim // self.num_heads
        attn_output = attn_output.view(batch, seq_len, self.num_heads, head_dim)

        # Zero out the chosen head (per token)
        attn_output[:, :, head_idx, :] = 0.0

        # Merge heads back to the original shape
        attn_output = attn_output.view(batch, seq_len, embed_dim)

        return attn_output, attn_weights

    attn.forward = patched_attn_forward.__get__(block.attn, block.attn.__class__)
    return orig_forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=str, default="ViT-B-32")
    ap.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    ap.add_argument("--root", type=str, default="data/wilds")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--subset-size", type=int, default=256, help="eval on a small subset for speed")
    ap.add_argument("--prompts", type=str, default=["a photo of a landbird", "a photo of a waterbird"], nargs="+")
    ap.add_argument("--out", type=str, default="outputs/scan_effects.csv")
    args = ap.parse_args()
    
    print(args)

    model, preprocess, tokenizer, device = load_clip(args.arch, args.pretrained)
    text_feats = encode_text(model, tokenizer, args.prompts, device)

    subset, metadata_fields = get_waterbirds(args.root, "test", transform=preprocess)
    if args.subset_size and len(subset) > args.subset_size:
        subset = torch.utils.data.Subset(subset, list(range(args.subset_size)))
    loader = make_loader(subset, batch_size=args.batch_size, shuffle=False)

    print("Get baseline logits")
    base_logits, base_targets, base_meta = [], [], []

    with torch.no_grad():
        for x, y, meta in loader:
            x = x.to(device)
            feats = encode_images(model, x, device)
            base_logits.append((100.0 * feats @ text_feats.T).cpu())
            base_targets.append(y)
            base_meta.append(meta)
    base_logits = torch.cat(base_logits)
    base_targets = torch.cat(base_targets)
    base_meta = torch.cat(base_meta)
    base_pred = base_logits.argmax(dim=-1)

    # scan each block/head
    vis = model.visual
    blocks = vis.transformer.resblocks
    num_blocks = len(blocks)
    num_heads = blocks[0].attn.num_heads

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["block", "head", "delta_margin_mean", "delta_overall_acc", "delta_worst_acc"])

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
                # restore attention head
                block.attn.forward = orig

                ab_logits = torch.cat(ab_logits)
                ab_pred = ab_logits.argmax(dim=-1)
                margin_base = base_logits[:, 1] - base_logits[:, 0]
                margin_ab = ab_logits[:, 1] - ab_logits[:, 0]
                delta = (margin_ab - margin_base).mean().item()
                metrics, _ = aggregate_accuracy_metrics(
                    base_targets.numpy(),
                    base_pred.numpy(),
                    ab_pred.numpy(),
                    base_meta.numpy(),
                    metadata_fields,
                )
                w.writerow(
                    [
                        b,
                        h,
                        f"{delta:.6f}",
                        f"{metrics['delta_overall']:.6f}",
                        f"{metrics['delta_worst']:.6f}",
                    ]
                )

    print(f"✅ Saved scan to {args.out}")


if __name__ == "__main__":
    main()
