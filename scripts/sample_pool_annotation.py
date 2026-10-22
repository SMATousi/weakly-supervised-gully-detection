"""D-4: draw ~100 random pool locations for expert annotation (prevalence and realistic precision).

Uniform random over pool locations, seeded. Locations with a truncated high-res image are not eligible
(0.3% of the pool; documented). Blind ids (1000+) hide the pool id from the annotators.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="artifacts/index/locations.parquet")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results/D4/sample.csv")
    a = ap.parse_args()
    idx = pd.read_parquet(a.index)
    el = idx[(idx.split == "pool") & (idx.highres_px >= 600)]
    pick = el.sample(n=a.n, random_state=a.seed).reset_index(drop=True)
    blind = np.random.default_rng(a.seed).permutation(a.n) + 1000
    out = pd.DataFrame({"loc_id": pick.loc_id, "legacy_id": pick.legacy_id, "blind_id": blind})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"{len(out)} sampled from {len(el)} eligible pool locations; wrote {a.out}")


if __name__ == "__main__":
    main()
