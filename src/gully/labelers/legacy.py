"""Ingest the old labeler outputs (Gully_Detection/weak-supervision/large_dataset/results) as LFs.

Old files are {legacy_id: label} or {legacy_id: {"class_label", "response", ["questions"]}}, one
directory for the pool (ids -> U-) and one for the expert-labeled tiles (ids -> L-). Both use bare
integers, which collide (audit B13), hence the prefixes. Missing locations are written as abstain.
"""
from __future__ import annotations

import json
from pathlib import Path

# labeler name -> (file in results/, model, prompt_id)
LEGACY = {
    "legacy_qwen2.5-vl-72b_sq": ("qwen2.5vl-72b-labels.json", "Qwen2.5-VL-72B", "legacy_sq"),
    "legacy_llama3.2-v-90b_sq": ("llama3.2-vision-90b-labels.json", "Llama-3.2-Vision-90B", "legacy_sq"),
    "legacy_llama3.2-v-90b_mq": (
        "llama3.2-vision-90b-llama3.2-vision-90b-multi-question-labels.json",
        "Llama-3.2-Vision-90B (aggregator Llama-3.2-Vision-90B)", "legacy_mq"),
    "legacy_llama3.2-v-90b_mq_scoutagg": (
        "llama3.2-vision-90b-llama4-scout-multi-question-labels.json",
        "Llama-3.2-Vision-90B (aggregator Llama-4-Scout)", "legacy_mq"),
    "legacy_qwen2.5-vl-7b_sq": ("qwen2.5vl:7b_labels.json", "Qwen2.5-VL-7B", "legacy_sq"),
    "legacy_llama4-scout_sq": ("llama4-scout-labels.json", "Llama-4-Scout", "legacy_sq"),
}
# the test files are named slightly differently for the 7B model
_TEST_ALIASES = {"qwen2.5vl:7b_labels.json": "qwen2.5vl-7b-labels.json"}


def _parse(v) -> tuple[int, str]:
    """(label in {0,1,-1}, raw text)."""
    if v is None:
        return -1, ""
    if isinstance(v, dict):
        raw = str(v.get("response", ""))
        cl = v.get("class_label")
        return (int(cl) if cl in (0, 1, "0", "1") else -1), raw
    return (int(v) if v in (0, 1) else -1), ""


def iter_records(results_dir: str | Path, name: str, split_dir: str, prefix: str, ids: list[str] | None):
    """Yield spec-format JSONL records for one labeler on one location set."""
    fname, model, prompt_id = LEGACY[name]
    if split_dir == "test_data":
        fname = _TEST_ALIASES.get(fname, fname)
    with open(Path(results_dir) / split_dir / fname if split_dir else Path(results_dir) / fname) as f:
        data = json.load(f)
    keys = ids if ids is not None else sorted(data, key=int)
    for k in keys:
        label, raw = _parse(data.get(k))
        yield {
            "loc_id": f"{prefix}-{int(k):06d}", "labeler": name, "model": model, "revision": None,
            "prompt_id": prompt_id, "image_mode": "collage", "raw": raw, "label": label,
            "p_yes": None, "abstain": label == -1, "latency_s": None, "ts": None,
        }
