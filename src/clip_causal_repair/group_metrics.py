from __future__ import annotations
import numpy as np
import pandas as pd


def _get_bg_idx(metadata_fields: list[str]) -> int:
    # Common names in WILDS Waterbirds meta
    for name in ["place", "background", "env", "environment"]:
        if name in metadata_fields:
            return metadata_fields.index(name)
    # Fallback: assume second field is background if present
    return 1 if len(metadata_fields) > 1 else 0


def group_report(y_true: np.ndarray, y_pred: np.ndarray, metadata: np.ndarray, metadata_fields: list[str]):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    bg_idx = _get_bg_idx(metadata_fields)
    bg_attr = metadata[:, bg_idx].astype(int)

    # group = (label, background)
    groups = list(set((int(y), int(b)) for y, b in zip(y_true, bg_attr)))
    rows = []
    for (y, b) in sorted(groups):
        mask = (y_true == y) & (bg_attr == b)
        acc = (y_pred[mask] == y_true[mask]).mean() if mask.any() else float("nan")
        rows.append({"label": y, "background": b, "n": int(mask.sum()), "acc": float(acc)})

    overall = float((y_pred == y_true).mean())
    worst_group = float(np.nanmin([r["acc"] for r in rows]))
    df = pd.DataFrame(rows)
    return overall, worst_group, df
