"""Label through a running vLLM OpenAI-compatible server (the persistent Deployment) instead of loading a model.

Same prompts, image preparation, answer schema, P(yes) extraction and JSONL format as `vlm.py`; only the
transport differs. Requests run concurrently and results are appended as they finish (resumable by loc_id).
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gully.labelers.vlm import (
    ANSWER_SCHEMA,
    build_messages,
    done_ids,
    make_record,
    p_yes_from_logprobs,
    prep_images,
)


def _data_url(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def to_api_messages(messages: list[dict]) -> list[dict]:
    """Replace `image_pil` parts by OpenAI-style `image_url` data-URL parts."""
    out = []
    for m in messages:
        if isinstance(m["content"], str):
            out.append(m)
            continue
        parts = [{"type": "image_url", "image_url": {"url": _data_url(p["image_pil"])}}
                 if p["type"] == "image_pil" else p for p in m["content"]]
        out.append({"role": m["role"], "content": parts})
    return out


def logprobs_to_lists(choice) -> tuple[list[str], list[dict[str, float]]]:
    """Token texts and {token: logprob} candidate dicts from an OpenAI chat-completion choice."""
    items = choice.logprobs.content if choice.logprobs and choice.logprobs.content else []
    toks = [it.token for it in items]
    tops = [{t.token: t.logprob for t in it.top_logprobs} or {it.token: it.logprob} for it in items]
    return toks, tops


def _open_append(path: Path):
    return open(path, "a")


async def label_all(client, cfg: dict, prompt: dict, ids: list[str], load_arrays, out_path: str | Path,
                    concurrency: int = 32, loader_threads: int = 8) -> int:
    """Label `ids` (skipping done ones); returns the number written."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = done_ids(out_path)
    todo = [i for i in ids if i not in done]
    sem, loop = asyncio.Semaphore(concurrency), asyncio.get_running_loop()
    pool = ThreadPoolExecutor(loader_threads)

    def build(loc_id: str):
        imgs = prep_images(load_arrays(loc_id), cfg["naip_size"], cfg["highres_size"])
        return to_api_messages(build_messages(imgs, prompt, cfg["image_mode"]))

    async def one(loc_id: str) -> dict:
        async with sem:
            msgs = await loop.run_in_executor(pool, build, loc_id)
            t0 = time.time()
            r = await client.chat.completions.create(
                model=cfg["model"], messages=msgs, temperature=0, max_tokens=48, logprobs=True,
                top_logprobs=20, seed=cfg.get("seed", 0),
                response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": ANSWER_SCHEMA}})
            c = r.choices[0]
            toks, tops = logprobs_to_lists(c)
            text = c.message.content or ""
            return make_record(loc_id, cfg, text, p_yes_from_logprobs(text, toks, tops), time.time() - t0)

    n = 0
    f = _open_append(out_path)
    try:
        for s in range(0, len(todo), concurrency * 4):  # windows bound memory
            for coro in asyncio.as_completed([one(i) for i in todo[s:s + concurrency * 4]]):
                f.write(json.dumps(await coro) + "\n")
                n += 1
            f.flush()
    finally:
        f.close()
    return n


async def wait_for_server(client, model: str, timeout_s: int = 7200) -> None:
    """Block until the server lists `model`; fail loudly if it serves a different one."""
    t0 = time.time()
    while True:
        try:
            ids = [m.id for m in (await client.models.list()).data]
            if model in ids:
                return
            raise RuntimeError(f"server serves {ids}, config wants {model}")
        except RuntimeError:
            raise
        except Exception:
            if time.time() - t0 > timeout_s:
                raise
            await asyncio.sleep(30)
