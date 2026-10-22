"""Write artifacts/labels/legacy_*.jsonl (pool + expert-labeled) from the old result files."""
import argparse
import json
from pathlib import Path

import pandas as pd

from gully.labelers.legacy import LEGACY, iter_records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/root/Gully_Detection/weak-supervision/large_dataset/results")
    ap.add_argument("--out", default="artifacts/labels")
    ap.add_argument("--index", default="artifacts/index/locations.parquet")
    args = ap.parse_args()
    idx = pd.read_parquet(args.index)
    ids = {p: idx.loc[idx.loc_id.str[0] == p, "legacy_id"].tolist() for p in ("U", "L")}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in LEGACY:
        n = {"U": 0, "L": 0}
        with open(out / f"{name}.jsonl", "w") as f:
            for split_dir, prefix in (("", "U"), ("test_data", "L")):
                try:
                    recs = list(iter_records(args.results, name, split_dir, prefix, ids[prefix]))
                except FileNotFoundError:
                    continue
                f.writelines(json.dumps(r) + "\n" for r in recs)
                n[prefix] = len(recs)
        print(f"{name}: pool={n['U']} labeled={n['L']}")


if __name__ == "__main__":
    main()
