from __future__ import annotations

import torch
import open_clip
from transformers import CLIPModel, CLIPProcessor


def load_clip(
    arch: str = "ViT-B-32",
    pretrained: str = "laion2b_s34b_b79k",
    device: str | None = None,
    backend: str = "openclip",
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if backend == "openclip":
        model, _, preprocess = open_clip.create_model_and_transforms(
            arch, pretrained=pretrained
        )
        tokenizer = open_clip.get_tokenizer(arch)
    elif backend == "hf":
        model = CLIPModel.from_pretrained(pretrained)
        processor = CLIPProcessor.from_pretrained(pretrained)
        preprocess = processor.image_processor
        tokenizer = processor.tokenizer
    else:
        raise ValueError("backend must be 'openclip' or 'hf'")

    model.eval().to(device)
    return model, preprocess, tokenizer, torch.device(device)


@torch.no_grad()
def encode_text(model, tokenizer, texts: list[str], device):
    tokens = tokenizer(texts)
    if isinstance(tokens, dict):
        tokens = {k: v.to(device) for k, v in tokens.items()}
    else:
        tokens = tokens.to(device)

    if hasattr(model, "encode_text"):
        text_features = model.encode_text(tokens)
    else:
        text_features = model.get_text_features(**tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


@torch.no_grad()
def encode_images(model, images, device):
    model_device = next(model.parameters()).device
    assert model_device.type == device.type, \
        f"Model device ({next(model.parameters()).device}) does not match target device ({device})"

    if isinstance(images, dict):
        images = images["pixel_values"].to(device)
    else:
        images = images.to(device)

    if hasattr(model, "encode_image"):
        image_features = model.encode_image(images)
    else:
        image_features = model.get_image_features(pixel_values=images)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features
