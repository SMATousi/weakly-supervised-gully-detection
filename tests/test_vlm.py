import json
import math

import numpy as np

from gully.labelers.vlm import (
    build_messages,
    done_ids,
    make_collage,
    make_record,
    p_yes_from_logprobs,
    parse_answer,
    prep_images,
)

PROMPT = {"system": "sys", "user": "q"}


def _arrays():
    return [np.random.default_rng(i).integers(0, 255, (128 if i < 7 else 700,) * 2 + (3,), dtype=np.uint8)
            for i in range(8)]


def test_parse_answer():
    assert parse_answer('{"answer": "yes", "confidence": 0.8}') == (1, 0.8)
    assert parse_answer('noise {"answer":"No","confidence":0.1} tail') == (0, 0.1)
    assert parse_answer('{"answer": "unsure", "confidence": 0.5}')[0] == -1
    assert parse_answer("Yes.")[0] == 1
    assert parse_answer("garbage")[0] == -1


def test_messages_modes_and_order():
    ims = prep_images(_arrays(), 224, 448)
    assert [i.size[0] for i in ims] == [224] * 7 + [448]
    multi = build_messages(ims, PROMPT, "multi_image")
    assert sum(p["type"] == "image_pil" for p in multi[1]["content"]) == 8
    col = build_messages(ims, PROMPT, "collage")
    assert sum(p["type"] == "image_pil" for p in col[1]["content"]) == 1
    assert make_collage(ims).size == (1200, 600)


def test_p_yes_from_logprobs():
    text = '{"answer": "yes", "confidence": 0.9}'
    toks = ['{"', 'answer', '":', ' "', 'yes', '",', ' "', 'confidence', '":', ' 0.9', '}']
    tops = [{} for _ in toks]
    tops[4] = {"yes": math.log(0.6), "no": math.log(0.2), "unsure": math.log(0.1)}
    p = p_yes_from_logprobs(text, toks, tops)
    assert abs(p - 0.75) < 1e-9


def test_resume(tmp_path):
    f = tmp_path / "x.jsonl"
    cfg = {"name": "n", "model": "m", "prompt_id": "p", "image_mode": "multi_image"}
    f.write_text(json.dumps(make_record("U-1", cfg, '{"answer":"no","confidence":0.2}', 0.3, 1.0)) + "\n"
                 + '{"loc_id": "U-2", "lab')  # truncated tail from a killed job
    assert done_ids(f) == {"U-1"}
