"""Assign dev/test/pool/pool_excluded in the location index and write the E1 leakage statistics."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gully.config import load_config
from gully.data.splits import assign_blocks, min_distance


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="artifacts/index/locations.parquet")
    ap.add_argument("--gt", default="artifacts/gt/gt.parquet")
    ap.add_argument("--out", default="results/E1")
    args = ap.parse_args()
    cfg = load_config("data")["split"]

    idx = pd.read_parquet(args.index).drop(columns=["split"], errors="ignore")
    gt = pd.read_parquet(args.gt)[["loc_id", "y_main"]]
    idx = idx.merge(gt, on="loc_id", how="left")
    lab = idx[idx.loc_id.str[0] == "L"]
    pool = idx[idx.loc_id.str[0] == "U"]
    has = lab[lab.y_main.notna()].rename(columns={"y_main": "label"})

    dev_target = round(len(has) * cfg["dev_fraction"])
    block_split, blk = assign_blocks(has, cfg["block_size_m"], dev_target, cfg["seed"])
    split = pd.Series("test", index=idx.index)
    split[idx.loc_id.str[0] == "U"] = "pool"
    split[lab.index] = "labeled_null"
    split[has.index] = blk.map(block_split)

    # E1: distances from every labeled location to the nearest pool location, and the exclusion
    d_lab = min_distance(lab[["x", "y"]].to_numpy(), pool[["x", "y"]].to_numpy())
    d_pool = min_distance(pool[["x", "y"]].to_numpy(), lab[["x", "y"]].to_numpy())
    excl = pool.index[d_pool < cfg["exclusion_buffer_m"]]
    split[excl] = "pool_excluded"
    idx["split"] = split

    stats = {
        "dev_target": dev_target,
        "counts": idx["split"].value_counts().to_dict(),
        "pos_rate": {s: float(idx[idx.split == s].y_main.mean()) for s in ("dev", "test")},
        "buffer_m": cfg["exclusion_buffer_m"],
        "labeled_to_nearest_pool_m": {q: float(np.quantile(d_lab, q)) for q in (0, 0.05, 0.25, 0.5, 0.75, 1)},
        "labeled_with_pool_within_buffer": int((d_lab < cfg["exclusion_buffer_m"]).sum()),
        "pool_within_buffer": len(excl),
        "pool_within_2x_buffer": int((d_pool < 2 * cfg["exclusion_buffer_m"]).sum()),
        "dev_to_nearest_test_m": {
            q: float(np.quantile(min_distance(idx[idx.split == "dev"][["x", "y"]].to_numpy(),
                                              idx[idx.split == "test"][["x", "y"]].to_numpy()), q))
            for q in (0, 0.05, 0.5)},
        "seed": cfg["seed"],
        "block_size_m": cfg["block_size_m"],
    }
    # blocks never straddle splits by construction; check it
    assert not set(blk[idx.loc[has.index, "split"] == "dev"]) & set(blk[idx.loc[has.index, "split"] == "test"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "split_stats.json").write_text(json.dumps(stats, indent=2))
    idx.drop(columns=["y_main"]).to_parquet(args.index)
    idx[idx.split.isin(["dev", "test"])][["loc_id", "split"]].to_csv(out / "dev_test_assignment.csv", index=False)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
