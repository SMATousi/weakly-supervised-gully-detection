"""Run one VLM labeler (configs/labelers/<name>.yaml) over a set of locations; resumable JSONL output.

  --set dev|test|labeled|pool|all   which locations (dev/test from the split; labeled = all 548 tiles)
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
    ap.add_argument("--set", default="dev", choices=["dev", "test", "labeled", "pool", "all"])
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
    sel = {"dev": idx.split == "dev", "test": idx.split == "test", "labeled": idx.index.str[0] == "L",
           "pool": idx.index.str[0] == "U", "all": idx.split.notna()}[a.set]
    ids = list(idx.index[sel])
    i, n = map(int, a.shard.split("/"))
    ids = ids[i::n][: a.limit]

    def load_arrays(loc_id: str):
        r = idx.loc[loc_id]
        d = Path(roots[r["pvc"]]) / r["img_dir"]
        return [read_rgb(d / r[f"img_naip_{k}"]) for k in range(7)] + [read_rgb(d / r["img_highres"])]

    suffix = "" if n == 1 else f".shard{i}of{n}"
    vlm.run(cfg, prompt, ids, load_arrays, Path(a.out) / f"{cfg['name']}.{a.set}{suffix}.jsonl")


if __name__ == "__main__":
    main()
