from __future__ import annotations
import torch
import open_clip


def load_clip(arch: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k", device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(arch)
    return model, preprocess, tokenizer, torch.device(device)


@torch.no_grad()
def encode_text(model, tokenizer, texts: list[str], device):
    tokens = tokenizer(texts).to(device)
    text_features = model.encode_text(tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    return text_features


@torch.no_grad()
def encode_images(model, images, device):
    assert next(model.parameters()).device == device, \
        f"Model device ({next(model.parameters()).device}) does not match target device ({device})"
    images = images.to(device)
    image_features = model.encode_image(images)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features
