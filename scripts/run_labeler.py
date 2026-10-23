"""Run one VLM labeler (configs/labelers/<name>.yaml) over one or more location sets; resumable JSONL output.

  --set dev|test|labeled|pool|all   one or more; the model is loaded ONCE and every set is labeled in order
  --server URL                      use a running vLLM server (OpenAI API) instead of loading the model here
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml

from gully.data.images import read_rgb
from gully.labelers import vlm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="+", default=["dev"], choices=["dev", "test", "labeled", "pool", "all"])
    ap.add_argument("--prompt-ids", nargs="+", help="prompt sweep: label with each of these prompts (one model load); outputs are named <labeler>__<prompt>")
    ap.add_argument("--server", help="e.g. http://gully-vlm:8000/v1")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--index", default="artifacts/index/locations.parquet")
    ap.add_argument("--pvc-root", action="append", required=True)
    ap.add_argument("--out", default="artifacts/labels")
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th location starting at i")
    ap.add_argument("--skip-first", type=int, default=0, help="after sharding, skip the first K locations (another job covers them)")
    ap.add_argument("--slice", default="0/1", help="k/m: the k-th of m contiguous slices of the remaining locations")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    with open(cfg["prompt_file"]) as f:
        prompts = yaml.safe_load(f)
    prompt = prompts[cfg["prompt_id"]]
    roots = dict(r.split("=", 1) for r in a.pvc_root)
    idx = pd.read_parquet(a.index).set_index("loc_id")
    i, n = map(int, a.shard.split("/"))
    suffix = ("" if n == 1 else f".shard{i}of{n}") + ("" if a.slice == "0/1" else f".slice{a.slice.replace('/', 'of')}")

    def ids_for(name: str) -> list[str]:
        sel = {"dev": idx.split == "dev", "test": idx.split == "test", "labeled": idx.index.str[0] == "L",
               "pool": idx.index.str[0] == "U", "all": idx.split.notna()}[name]
        ids = list(idx.index[sel])[i::n][a.skip_first:]
        k, m = map(int, a.slice.split("/"))
        size = -(-len(ids) // m)  # contiguous slice k of m
        return ids[k * size:(k + 1) * size][: a.limit]

    def load_arrays(loc_id: str):
        r = idx.loc[loc_id]
        d = Path(roots[r["pvc"]]) / r["img_dir"]
        return [read_rgb(d / r[f"img_naip_{k}"]) for k in range(7)] + [read_rgb(d / r["img_highres"])]

    if a.prompt_ids:  # one task per (prompt, set); labeler name carries the prompt id
        tasks = []
        for pid in a.prompt_ids:
            tcfg = {**cfg, "prompt_id": pid, "name": f"{cfg['name']}__{pid}"}
            tasks += [(ids_for(name), Path(a.out) / f"{tcfg['name']}.{name}{suffix}.jsonl", prompts[pid], tcfg)
                      for name in a.set]
    else:
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
