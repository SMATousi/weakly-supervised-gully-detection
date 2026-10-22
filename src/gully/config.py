"""Config loading. All paths and seeds come from configs/*.yaml, never from code."""
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(name: str) -> dict:
    """Load configs/<name>.yaml (name may include subdirs, without the extension)."""
    with open(REPO_ROOT / "configs" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def git_sha() -> str:
    """Short commit hash, or 'unknown' when git is unavailable (e.g. inside the cluster image)."""
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], check=False, capture_output=True,
                              text=True, cwd=REPO_ROOT).stdout.strip() or "unknown"
    except OSError:
        return "unknown"
