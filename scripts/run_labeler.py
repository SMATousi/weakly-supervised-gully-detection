"""Run one VLM labeler (configs/labelers/<name>.yaml) over one or more location sets; resumable JSONL output.

  --set dev|test|labeled|pool|all   one or more; the model is loaded ONCE and every set is labeled in order
  --server URL                      use a running vLLM server (OpenAI API) instead of loading the model here
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

from gully.labelers import vlm
from gully.student.features import read_rgb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="+", default=["dev"], choices=["dev", "test", "labeled", "pool", "all"])
    ap.add_argument("--server", help="e.g. http://gully-vlm:8000/v1")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--index", default="artifacts/index/locations.parquet")
    ap.add_argument("--pvc-root", action="append", required=True)
    ap.add_argument("--out", default="artifacts/labels")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th location starting at i")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    with open(cfg["prompt_file"]) as f:
        prompt = yaml.safe_load(f)[cfg["prompt_id"]]
    roots = dict(r.split("=", 1) for r in a.pvc_root)
    idx = pd.read_parquet(a.index).set_index("loc_id")
    i, n = map(int, a.shard.split("/"))
    suffix = "" if n == 1 else f".shard{i}of{n}"

    def ids_for(name: str) -> list[str]:
        sel = {"dev": idx.split == "dev", "test": idx.split == "test", "labeled": idx.index.str[0] == "L",
               "pool": idx.index.str[0] == "U", "all": idx.split.notna()}[name]
        return list(idx.index[sel])[i::n][: a.limit]

    def load_arrays(loc_id: str):
        r = idx.loc[loc_id]
        d = Path(roots[r["pvc"]]) / r["img_dir"]
        return [read_rgb(d / r[f"img_naip_{k}"]) for k in range(7)] + [read_rgb(d / r["img_highres"])]

    tasks = [(ids_for(name), Path(a.out) / f"{cfg['name']}.{name}{suffix}.jsonl") for name in a.set]
    if a.server:
        import asyncio

        from openai import AsyncOpenAI

        from gully.labelers.vlm_client import label_all, wait_for_server

        async def go() -> None:
            client = AsyncOpenAI(base_url=a.server, api_key="none", timeout=600)
            await wait_for_server(client, cfg["model"])
            for ids, out in tasks:
                print(f"{out.name}: wrote {await label_all(client, cfg, prompt, ids, load_arrays, out, a.concurrency)}",
                      flush=True)

        asyncio.run(go())
    else:
        vlm.run_many(cfg, prompt, tasks, load_arrays)


if __name__ == "__main__":
    main()
