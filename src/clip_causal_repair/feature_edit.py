from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from .clip_loader import load_clip, encode_text, encode_images
from .waterbirds import get_waterbirds, make_loader
from .group_metrics import group_report


@torch.no_grad()
def extract_image_features(model, loader, device):
    feats, ys, metas = [], [], []
    for x, y, meta in tqdm(loader, desc="feats"):
        x = x.to(device)
        f = model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
        ys.append(y)
        metas.append(meta)
    return torch.cat(feats).numpy(), torch.cat(ys).numpy(), torch.cat(metas).numpy()


def fit_direction_from_metadata(features: np.ndarray, metadata: np.ndarray, metadata_fields: list[str]) -> np.ndarray:
    # Find background attribute index
    bg_idx = None
    for name in ["place", "background", "env", "environment"]:
        if name in metadata_fields:
            bg_idx = metadata_fields.index(name)
            break
    if bg_idx is None:
        raise RuntimeError("Could not find background attribute in metadata fields")

    y_bg = metadata[:, bg_idx].astype(int)
    clf = LogisticRegression(max_iter=200, solver="lbfgs")
    clf.fit(features, y_bg)
    w = clf.coef_.reshape(-1)
    w = w / (np.linalg.norm(w) + 1e-12)
    return w.astype(np.float32)


class ProjectionEdit:
    def __init__(self, direction: np.ndarray, layer: str = "final", alpha: float = 1.0):
        self.u = torch.tensor(direction, dtype=torch.float32)
        self.layer = layer
        self.alpha = alpha
        self.hook = None

    def _hook_fn(self, module, inputs, output):
        # output is [B, D]
        u = self.u.to(output.device)
        proj = (output @ u)[:, None] * u[None, :]
        return output - self.alpha * proj

    def apply(self, model):
        # simple variant: edit the final pooled features
        if self.layer == "final":
            # OpenCLIP: model.visual.layernorm_post (just before projection)
            target = model.visual
            if hasattr(target, "ln_post"):
                mod = target.ln_post
            elif hasattr(target, "layernorm_post"):
                mod = target.layernorm_post
            else:
                raise RuntimeError("Could not find final layer norm to hook")
            self.hook = mod.register_forward_hook(lambda m, x, y: self._hook_fn(m, x, y))
        else:
            raise NotImplementedError("Only layer=final is implemented in this minimal scaffold")

    def remove(self):
        if self.hook is not None:
            self.hook.remove()
            self.hook = None


def cmd_fit_direction(args):
    model, preprocess, tokenizer, device = load_clip(args.arch, args.pretrained)
    subset, metadata_fields = get_waterbirds(args.root, args.split, transform=preprocess)
    loader = make_loader(subset, batch_size=args.batch_size, shuffle=True)

    if args.max_samples:
        idx = list(range(min(args.max_samples, len(subset))))
        subset = torch.utils.data.Subset(subset, idx)
        loader = make_loader(subset, batch_size=args.batch_size, shuffle=True)

    feats, ys, metas = extract_image_features(model, loader, device)
    direction = fit_direction_from_metadata(feats, metas, metadata_fields)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, direction)
    print(f"✅ Saved direction to {args.out}")


def cmd_apply(args):
    model, preprocess, tokenizer, device = load_clip(args.arch, args.pretrained)
    direction = np.load(args.direction)

    edit = ProjectionEdit(direction, layer=args.layer, alpha=args.alpha)
    edit.apply(model)

    prompts = ["a photo of a landbird", "a photo of a waterbird"]
    text_feats = encode_text(model, tokenizer, prompts, device)

    subset, metadata_fields = get_waterbirds(args.root, args.eval_split, transform=preprocess)
    loader = make_loader(subset, batch_size=args.batch_size, shuffle=False)

    ys, preds, metas = [], [], []
    with torch.no_grad():
        for x, y, meta in tqdm(loader, desc="eval-after-edit"):
            x = x.to(device)
            img = encode_images(model, x, device)
            logits = 100.0 * img @ text_feats.T
            pred = logits.argmax(dim=-1).cpu()
            ys.append(y)
            preds.append(pred)
            metas.append(meta)

    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(preds).numpy()
    meta = torch.cat(metas).numpy()

    overall, worst, df = group_report(y_true, y_pred, meta, metadata_fields)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df.to_csv(args.out, index=False)
    print(f"Overall acc (after edit): {overall:.3f}\nWorst-group acc (after edit): {worst:.3f}\nSaved per-group to: {args.out}")

    edit.remove()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    fit = sub.add_parser("fit-direction")
    fit.add_argument("--arch", type=str, default="ViT-B-32")
    fit.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    fit.add_argument("--root", type=str, default="data/wilds")
    fit.add_argument("--split", type=str, default="train")
    fit.add_argument("--batch-size", type=int, default=64)
    fit.add_argument("--max-samples", type=int, default=2000)
    fit.add_argument("--out", type=str, default="edits/water_direction.npy")

    app = sub.add_parser("apply")
    app.add_argument("--arch", type=str, default="ViT-B-32")
    app.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    app.add_argument("--root", type=str, default="data/wilds")
    app.add_argument("--direction", type=str, required=True)
    app.add_argument("--layer", type=str, default="final")
    app.add_argument("--alpha", type=float, default=1.0)
    app.add_argument("--eval-split", type=str, default="test")
    app.add_argument("--batch-size", type=int, default=64)
    app.add_argument("--out", type=str, default="outputs/after_edit.csv")

    args = ap.parse_args()

    if args.cmd == "fit-direction":
        cmd_fit_direction(args)
    elif args.cmd == "apply":
        cmd_apply(args)


if __name__ == "__main__":
    main()
