"""Zero-shot CLIP-family labelers (E4): cheap, non-chat, error profile unlike the chat VLMs.

For each location, every image is scored against positive / negative text prompts; per-image score =
softmax over the prompt ensemble. The location score aggregates over dates (max over NAIP dates and the
high-res image by default: a gully visible on any date counts). Thresholds / abstain band are tuned on dev
by the caller (ws/confidence_lfs.py); this module only produces continuous scores.
"""
from __future__ import annotations

import numpy as np

POS = [
    "an aerial image of an ephemeral gully, a narrow erosion channel cutting through a farm field",
    "a satellite image showing gully erosion in agricultural land",
    "an aerial photo of a concentrated flow channel with eroded bare soil",
]
NEG = [
    "an aerial image of a smooth agricultural field with no erosion",
    "a satellite image of farmland without gullies",
    "an aerial photo of a uniform crop field",
]


def location_scores(img_logits: np.ndarray, n_pos: int, agg: str = "max") -> np.ndarray:
    """img_logits: (N, 8, n_pos + n_neg) image-text logits. Returns P(gully) per location (N,).

    Per image: softmax over [mean(pos prompts), mean(neg prompts)] logits -> p_img; then aggregate over the
    8 images with `max`, `mean`, or `top2` (mean of the two largest).
    """
    pos = img_logits[..., :n_pos].mean(-1)
    neg = img_logits[..., n_pos:].mean(-1)
    p_img = 1 / (1 + np.exp(-(pos - neg)))
    if agg == "max":
        return p_img.max(1)
    if agg == "mean":
        return p_img.mean(1)
    if agg == "top2":
        return np.sort(p_img, 1)[:, -2:].mean(1)
    raise ValueError(agg)


def encode_text(model_id: str, prompts: list[str], device: str):
    """Text embeddings (L2-normalized) from an open_clip / HF CLIP model id."""
    import torch
    from transformers import CLIPModel, CLIPTokenizer

    tok, model = CLIPTokenizer.from_pretrained(model_id), CLIPModel.from_pretrained(model_id).to(device).eval()
    with torch.no_grad():
        t = model.get_text_features(**tok(prompts, padding=True, return_tensors="pt").to(device))
    return model, torch.nn.functional.normalize(t, dim=-1)
