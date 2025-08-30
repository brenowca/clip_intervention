from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .clip_loader import load_clip, encode_text, encode_images
from .waterbirds import get_waterbirds, make_loader
from .group_metrics import group_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=str, default="ViT-B-32")
    ap.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    ap.add_argument("--backend", type=str, default="openclip", choices=["openclip", "hf"])
    ap.add_argument("--root", type=str, default="data/wilds")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--prompts", type=str, default=["a photo of a landbird", "a photo of a waterbird"], nargs="+")
    ap.add_argument("--out", type=str, default="outputs/zero_shot.csv")
    args = ap.parse_args()

    model, preprocess, tokenizer, device = load_clip(
        args.arch, args.pretrained, backend=args.backend
    )

    text_feats = encode_text(model, tokenizer, args.prompts, device)

    subset, metadata_fields = get_waterbirds(args.root, args.split, transform=preprocess, download=False)
    loader = make_loader(subset, batch_size=args.batch_size, shuffle=False)

    ys, preds, metas = [], [], []
    with torch.no_grad():
        for x, y, meta in tqdm(loader, desc="eval"):
            x = x.to(device)
            image_feats = encode_images(model, x, device)
            logits = 100.0 * image_feats @ text_feats.T
            pred = logits.argmax(dim=-1).cpu()
            ys.append(y)
            preds.append(pred)
            metas.append(meta)

    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(preds).numpy()
    meta = torch.cat(metas).numpy()

    overall, worst, df = group_report(y_true, y_pred, meta, metadata_fields)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    print(f"Overall acc: {overall:.3f}\nWorst-group acc: {worst:.3f}\nSaved per-group to: {args.out}")


if __name__ == "__main__":
    main()
