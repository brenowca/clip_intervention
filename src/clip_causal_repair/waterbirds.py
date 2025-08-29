from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from wilds import get_dataset


def get_waterbirds(root: str, split: str, transform=None, download: bool = False):
    ds = get_dataset(dataset="waterbirds", root_dir=root, download=download)
    subset = ds.get_subset(split, transform=transform)
    return subset, ds.metadata_fields


def make_loader(subset, batch_size: int = 64, num_workers: int = 4, shuffle: bool = False):
    return DataLoader(subset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--root", type=str, default="data/wilds")
    args = ap.parse_args()
    if args.download:
        Path(args.root).mkdir(parents=True, exist_ok=True)
        _ = get_dataset(dataset="waterbirds", root_dir=args.root, download=True)
        print("✅ Waterbirds downloaded to", args.root)


if __name__ == "__main__":
    main()
