import json
import subprocess
import sys

import pandas as pd

from gully.config import REPO_ROOT


def test_analyze_d4_on_synthetic(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    # 10 tiles: first 4 clearly positive (a score 4, others >0), rest all 0
    for a in (1, 2, 3):
        d = {str(1000 + i): {"images": {str(r): (4 if i < 4 and r == 0 else (3 if i < 4 else 0)) for r in
                                        [0, 1, 2, 3, 4, 5, 6, 11]}} for i in range(10)}
        (raw / f"annotator_{a}.json").write_text(json.dumps(d))
    pd.DataFrame({"loc_id": [f"U-{i:06d}" for i in range(10)], "legacy_id": range(10),
                  "blind_id": [1000 + i for i in range(10)]}).to_csv(tmp_path / "s.csv", index=False)
    pd.DataFrame({"loc_id": [f"U-{i:06d}" for i in range(10)],
                  "p1": [0.9] * 4 + [0.1] * 6}).to_parquet(tmp_path / "pseudo.parquet")
    out = tmp_path / "out.json"
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts/analyze_d4.py"), "--raw", str(raw), "--sample",
                    str(tmp_path / "s.csv"), "--scores", str(tmp_path / "pseudo.parquet"), "--out", str(out)],
                   check=True, capture_output=True)
    r = json.loads(out.read_text())
    assert r["strict_neg"]["positives"] == 4 and abs(r["strict_neg"]["prevalence"] - 0.4) < 1e-9
    assert r["scores"]["pseudo"]["acc"] == 1.0
