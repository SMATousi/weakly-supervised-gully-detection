"""VLM labeler on vLLM: multi-image or collage input, JSON-constrained answer, P(yes) from logprobs.

Pure helpers (messages, parsing, p_yes, resume) are importable without vLLM; `run` needs the `vlm` extra.
Output: one JSONL record per location (spec/02-architecture.md), appended and flushed per chunk so a
killed job resumes where it stopped.
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

YEARS = ["2010", "2012", "2014", "2016", "2018", "2020", "2022", "2023"]
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string", "enum": ["yes", "no", "unsure"]},
                   "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
    "required": ["answer", "confidence"],
}


def prep_images(arrays: list[np.ndarray], naip_size: int, highres_size: int) -> list[Image.Image]:
    """8 uint8 RGB arrays (7 NAIP + high-res, in role order) -> PIL images at the model-facing sizes."""
    out = []
    for i, a in enumerate(arrays):
        s = highres_size if i == 7 else naip_size
        out.append(Image.fromarray(a).convert("RGB").resize((s, s), Image.BICUBIC if i < 7 else Image.LANCZOS))
    return out


def make_collage(images: list[Image.Image], cell: int = 300, cols: int = 4) -> Image.Image:
    """The legacy 4x2 collage with year labels (kept as an `image_mode` for comparison, E2)."""
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, im in enumerate(images):
        x, y = (i % cols) * cell, (i // cols) * cell
        canvas.paste(im.resize((cell, cell), Image.LANCZOS), (x, y))
        draw.rectangle([x, y, x + len(YEARS[i]) * 8, y + 20], fill=(0, 0, 0))
        draw.text((x + 2, y + 4), YEARS[i], fill=(255, 255, 255))
    return canvas


def build_messages(images: list[Image.Image], prompt: dict, mode: str) -> list[dict]:
    """OpenAI-style chat messages; images are PIL objects (vLLM `image_pil` parts)."""
    if mode == "collage":
        content = [{"type": "image_pil", "image_pil": make_collage(images)}]
    elif mode == "multi_image":
        content = []
        for i, im in enumerate(images):
            label = f"Image {i + 1}: " + (f"NAIP {YEARS[i]} (about 1 m/pixel)" if i < 7
                                            else "very high resolution (about 15 cm/pixel) 2023")
            content += [{"type": "text", "text": label}, {"type": "image_pil", "image_pil": im}]
    else:
        raise ValueError(mode)
    content.append({"type": "text", "text": prompt["user"]})
    return [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": content}]


def parse_answer(text: str) -> tuple[int, float | None]:
    """(label in {1,0,-1}, confidence) from the JSON answer; tolerant of stray text."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    ans = str(d.get("answer", "")).strip().lower()
    conf = d.get("confidence")
    conf = float(conf) if isinstance(conf, (int, float)) else None
    if ans == "yes":
        return 1, conf
    if ans == "no":
        return 0, conf
    if not d:  # fall back on a bare yes/no
        w = re.match(r"\W*(yes|no)\b", text.strip(), re.IGNORECASE)
        if w:
            return (1 if w.group(1).lower() == "yes" else 0), None
    return -1, conf


def p_yes_from_logprobs(text: str, token_ids_text: list[str], top_logprobs: list[dict[str, float]]) -> float | None:
    """P(yes) among {yes, no} at the token where the answer value starts.

    `token_ids_text[i]` is the decoded text of generated token i; `top_logprobs[i]` maps candidate token
    text -> logprob at that position. Returns None if the answer position is not found.
    """
    m = re.search(r'"answer"\s*:\s*"', text)
    if not m:
        return None
    start, pos = m.end(), 0
    for i, t in enumerate(token_ids_text):
        if pos + len(t) > start - 1 and pos >= start - len(t):  # token covering/after the opening quote
            cand = top_logprobs[i]
            py = sum(math.exp(lp) for tok, lp in cand.items() if tok.strip().lower().startswith("y"))
            pn = sum(math.exp(lp) for tok, lp in cand.items() if tok.strip().lower().startswith("n"))
            if py + pn > 0:
                return py / (py + pn)
        pos += len(t)
        if pos > start + 8:
            break
    return None


def done_ids(path: str | Path) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text().splitlines():
        try:
            out.add(json.loads(line)["loc_id"])
        except (json.JSONDecodeError, KeyError):
            continue  # truncated last line from a killed job
    return out


def make_record(loc_id: str, cfg: dict, text: str, p_yes: float | None, latency: float) -> dict:
    label, conf = parse_answer(text)
    return {"loc_id": loc_id, "labeler": cfg["name"], "model": cfg["model"], "revision": cfg.get("revision"),
            "prompt_id": cfg["prompt_id"], "image_mode": cfg["image_mode"], "raw": text, "label": label,
            "p_yes": p_yes, "confidence": conf, "abstain": label == -1, "latency_s": latency,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}


def _engine(cfg: dict):
    from vllm import LLM, SamplingParams

    try:  # vLLM >= 0.11 renamed guided decoding to structured outputs
        from vllm.sampling_params import StructuredOutputsParams

        constraint = {"structured_outputs": StructuredOutputsParams(json=ANSWER_SCHEMA)}
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams

        constraint = {"guided_decoding": GuidedDecodingParams(json=ANSWER_SCHEMA)}
    llm = LLM(model=cfg["model"], revision=cfg.get("revision"), tensor_parallel_size=cfg.get("tp", 1),
              max_model_len=cfg.get("max_model_len", 8192), dtype=cfg.get("dtype", "bfloat16"),
              limit_mm_per_prompt={"image": 8}, gpu_memory_utilization=cfg.get("gpu_mem_util", 0.9),
              trust_remote_code=cfg.get("trust_remote_code", False), seed=cfg.get("seed", 0),
              **cfg.get("llm_kwargs", {}))
    return llm, SamplingParams(temperature=0.0, max_tokens=48, logprobs=20, **constraint)


def run_many(cfg: dict, prompt: dict, tasks: list[tuple[list[str], str | Path]], load_arrays,
             chunk: int = 64) -> None:
    """Load the model ONCE, then label every task = (loc_ids, out_path). Each output is resumable."""
    todo = []
    for ids, out_path in tasks:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        done = done_ids(out_path)
        todo.append(([i for i in ids if i not in done], Path(out_path)))
    if not any(t[0] for t in todo):
        return
    llm, sp = _engine(cfg)
    for ids_todo, out_path in todo:
        with open(out_path, "a") as f:
            for s in range(0, len(ids_todo), chunk):
                ids = ids_todo[s:s + chunk]
                msgs = [build_messages(prep_images(load_arrays(i), cfg["naip_size"], cfg["highres_size"]),
                                       prompt, cfg["image_mode"]) for i in ids]
                t0 = time.time()
                outs = llm.chat(msgs, sp)
                lat = (time.time() - t0) / len(ids)
                for i, o in zip(ids, outs, strict=True):
                    c = o.outputs[0]
                    toks = [next(iter(lp.values())).decoded_token or "" for lp in c.logprobs]
                    tops = [{v.decoded_token or "": v.logprob for v in lp.values()} for lp in c.logprobs]
                    f.write(json.dumps(make_record(i, cfg, c.text, p_yes_from_logprobs(c.text, toks, tops),
                                                   lat)) + "\n")
                f.flush()


def run(cfg: dict, prompt: dict, locations, load_arrays, out_path: str | Path, chunk: int = 64) -> None:
    run_many(cfg, prompt, [(list(locations), out_path)], load_arrays, chunk)
