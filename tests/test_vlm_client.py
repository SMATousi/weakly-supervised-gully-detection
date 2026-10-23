import asyncio
import json
import math
from types import SimpleNamespace as NS

import numpy as np

from gully.labelers.vlm import build_messages, prep_images
from gully.labelers.vlm_client import label_all, logprobs_to_lists, to_api_messages

PROMPT = {"system": "s", "user": "q"}
CFG = {"name": "n", "model": "m", "prompt_id": "p", "image_mode": "multi_image", "naip_size": 32,
       "highres_size": 64}


def _arrays(_loc):
    r = np.random.default_rng(0)
    return [r.integers(0, 255, (16, 16, 3), dtype=np.uint8) for _ in range(7)] + [
        r.integers(0, 255, (50, 50, 3), dtype=np.uint8)]


def test_to_api_messages_has_data_urls():
    msgs = to_api_messages(build_messages(prep_images(_arrays(0), 32, 64), PROMPT, "multi_image"))
    urls = [p["image_url"]["url"] for p in msgs[1]["content"] if p["type"] == "image_url"]
    assert len(urls) == 8 and all(u.startswith("data:image/png;base64,") for u in urls)
    assert msgs[0]["content"] == "s"


def _choice(answer="yes", conf=0.9):
    text = json.dumps({"answer": answer, "confidence": conf})
    toks = ['{"', 'answer', '":', ' "', answer, '",', ' "', 'confidence', '":', f" {conf}", "}"]
    items = []
    for i, t in enumerate(toks):
        top = [NS(token=t, logprob=0.0)]
        if i == 4:
            top = [NS(token="yes", logprob=math.log(0.6)), NS(token="no", logprob=math.log(0.2))]
        items.append(NS(token=t, logprob=0.0, top_logprobs=top))
    return NS(message=NS(content=text), logprobs=NS(content=items))


def test_logprobs_to_lists():
    toks, tops = logprobs_to_lists(_choice())
    assert toks[4] == "yes" and abs(tops[4]["yes"] - math.log(0.6)) < 1e-12


def test_label_all_writes_and_resumes(tmp_path):
    calls = []

    class FakeClient:
        class chat:
            class completions:
                @staticmethod
                async def create(**kw):
                    calls.append(kw["model"])
                    return NS(choices=[_choice()])

    out = tmp_path / "x.jsonl"
    n = asyncio.run(label_all(FakeClient, CFG, PROMPT, ["U-1", "U-2", "U-3"], _arrays, out, concurrency=2))
    assert n == 3 and len(calls) == 3
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert {r["loc_id"] for r in recs} == {"U-1", "U-2", "U-3"}
    assert all(r["label"] == 1 and abs(r["p_yes"] - 0.75) < 1e-9 for r in recs)
    assert asyncio.run(label_all(FakeClient, CFG, PROMPT, ["U-1", "U-4"], _arrays, out, concurrency=2)) == 1
