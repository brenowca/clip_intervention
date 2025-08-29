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


def cmd_fit_direction(arch, pretrained, root, split, max_samples, batch_size, out):
    model, preprocess, tokenizer, device = load_clip(arch, pretrained)
    subset, metadata_fields = get_waterbirds(root, split, transform=preprocess)
    loader = make_loader(subset, batch_size=batch_size, shuffle=True)

    if max_samples:
        idx = list(range(min(max_samples, len(subset))))
        subset = torch.utils.data.Subset(subset, idx)
        loader = make_loader(subset, batch_size=batch_size, shuffle=True)

    feats, ys, metas = extract_image_features(model, loader, device)
    direction = fit_direction_from_metadata(feats, metas, metadata_fields)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.save(out, direction)
    print(f"✅ Saved direction to {out}")


def cmd_apply(arch, pretrained, root, direction, layer, alpha, eval_split, batch_size, out):
    """
    Apply a directional edit to the CLIP model and evaluate its performance on a specified dataset split.

    Parameters
    ----------
    - arch (str): Architecture of the CLIP model.
    - pretrained (str): Pretrained weights identifier.
    - root (str): Path to the dataset root.
    - direction (str): Filepath to the saved direction vector.
    - layer (str): Layer to apply the edit to.
    - alpha (float): Scaling factor for the edit.
    - eval_split (str): Dataset split for evaluation.
    - batch_size (int): Batch size for data loading.
    - out (str): Output CSV file path for evaluation results.

    Steps
    -----
    1. Load the CLIP model and related components.
    2. Load the directional edit vector.
    3. Apply the edit to the specified layer of the model.
    4. Encode text prompts for classification.
    5. Load evaluation data and compute predictions.
    6. Generate evaluation metrics and save results.
    7. Remove the applied edit from the model.

    Outputs
    -------
    Saves a CSV file with per-group evaluation metrics and prints overall and worst-group accuracies.
    """
    model, preprocess, tokenizer, device = load_clip(arch, pretrained)
    direction = np.load(direction)

    edit = ProjectionEdit(direction, layer=layer, alpha=alpha)
    edit.apply(model)

    prompts = ["a photo of a landbird", "a photo of a waterbird"]
    text_feats = encode_text(model, tokenizer, prompts, device)

    subset, metadata_fields = get_waterbirds(root, eval_split, transform=preprocess)
    loader = make_loader(subset, batch_size=batch_size, shuffle=False)

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
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df.to_csv(out, index=False)
    print(f"Overall acc (after edit): {overall:.3f}\nWorst-group acc (after edit): {worst:.3f}\nSaved per-group to: {out}")

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
        cmd_fit_direction(args.arch, args.pretrained, args.root, args.split, args.max_samples, args.batch_size, args.out)
    elif args.cmd == "apply":
        cmd_apply(args.arch, args.pretrained, args.root, np.load(args.direction), args.layer, args.alpha, args.eval_split, args.batch_size, args.out)


if __name__ == "__main__":
    main()
