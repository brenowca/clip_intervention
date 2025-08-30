from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import matplotlib.pyplot as plt

from .clip_loader import load_clip, encode_text, encode_images
from .waterbirds import get_waterbirds, make_loader
from .group_metrics import group_report, _get_bg_idx


def parse_alpha_values(alpha_str: str) -> list[float]:
    return [float(a) for a in alpha_str.split(',') if a]


@torch.no_grad()
def extract_embeddings(model, loader, device):
    feats, ys, metas = [], [], []
    for x, y, meta in tqdm(loader, desc="embed"):
        x = x.to(device)
        f = encode_images(model, x, device)
        feats.append(f.cpu())
        ys.append(y)
        metas.append(meta)
    return (
        torch.cat(feats).numpy(),
        torch.cat(ys).numpy(),
        torch.cat(metas).numpy(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", type=str, default="ViT-B-32")
    ap.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    ap.add_argument("--root", type=str, default="data/wilds")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--split-train", type=str, default="val")
    ap.add_argument("--split-test", type=str, default="test")
    ap.add_argument("--alpha-values", type=str, default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--out-dir", type=str, default="outputs/inlp")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, preprocess, tokenizer, device = load_clip(args.arch, args.pretrained)
    model.eval()
    model.requires_grad_(False)

    prompts = ["a photo of a landbird", "a photo of a waterbird"]
    text_feats = encode_text(model, tokenizer, prompts, device)

    # ---- Train background classifier ----
    train_subset, metadata_fields = get_waterbirds(
        args.root, args.split_train, transform=preprocess
    )
    train_loader = make_loader(train_subset, batch_size=args.batch_size, shuffle=False)
    feats, _, metas = extract_embeddings(model, train_loader, device)

    bg_idx = _get_bg_idx(metadata_fields)
    y_bg = metas[:, bg_idx].astype(int)
    X_tr, X_val, y_tr, y_val = train_test_split(
        feats, y_bg, test_size=0.3, stratify=y_bg, random_state=0
    )

    clf = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, solver="lbfgs")
    )
    clf.fit(X_tr, y_tr)
    val_acc = clf.score(X_val, y_val)
    print(f"BG val acc: {val_acc:.3f}")

    scaler = clf.named_steps["standardscaler"]
    lr = clf.named_steps["logisticregression"]
    w = (lr.coef_[0] / scaler.scale_).astype(np.float32)
    np.save(out_dir / "w.npy", w)

    # ---- Extract test embeddings ----
    test_subset, _ = get_waterbirds(
        args.root, args.split_test, transform=preprocess
    )
    test_loader = make_loader(test_subset, batch_size=args.batch_size, shuffle=False)
    test_feats, y_true, test_meta = extract_embeddings(model, test_loader, device)

    alphas = parse_alpha_values(args.alpha_values)
    rows = []
    w_norm_sq = float(np.dot(w, w)) + 1e-12
    for alpha in alphas:
        proj = test_feats - alpha * np.outer(test_feats @ w, w) / w_norm_sq
        proj = proj / np.linalg.norm(proj, axis=1, keepdims=True)
        cos = (test_feats * proj).sum(axis=1)
        mean_cos = float(cos.mean())

        img = torch.from_numpy(proj).to(device)
        logits = 100.0 * img @ text_feats.T
        preds = logits.argmax(dim=-1).cpu().numpy()

        overall, worst, _ = group_report(y_true, preds, test_meta, metadata_fields)
        rows.append(
            {
                "alpha": alpha,
                "overall": overall,
                "worst_group": worst,
                "mean_cosine": mean_cos,
            }
        )
        print(f"alpha={alpha:.2f} overall={overall:.3f} worst={worst:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics.csv", index=False)

    plt.figure()
    plt.plot(df["alpha"], df["overall"], label="overall")
    plt.plot(df["alpha"], df["worst_group"], label="worst-group")
    plt.xlabel("alpha")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(out_dir / "accuracy_vs_alpha.png")


if __name__ == "__main__":
    main()
