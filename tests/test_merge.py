import json
import subprocess
import sys

from gully.config import REPO_ROOT


def _w(p, ids):
    p.write_text("".join(json.dumps({"loc_id": i, "label": 1}) + "\n" for i in ids) + '{"loc_id": "U-9')


def test_merge_dedups_and_checks_counts(tmp_path):
    _w(tmp_path / "x__p.labeled.jsonl", ["L-000001", "L-000002"])
    _w(tmp_path / "x__p.pool.shard0of2.jsonl", ["U-000001", "U-000002"])
    _w(tmp_path / "x__p.pool.shard1of2.jsonl", ["U-000003", "U-000002"])  # U-000002 duplicated
    cmd = [sys.executable, str(REPO_ROOT / "scripts/merge_labels.py"), "--name", "x__p", "--src", str(tmp_path)]
    ok = subprocess.run([*cmd, "--expect-pool", "3", "--expect-labeled", "2"], check=False, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    assert len((tmp_path / "x__p.jsonl").read_text().splitlines()) == 5
    bad = subprocess.run([*cmd, "--expect-pool", "4"], check=False, capture_output=True, text=True)
    assert bad.returncode != 0
