"""Config loading. All paths and seeds come from configs/*.yaml, never from code."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(name: str) -> dict:
    """Load configs/<name>.yaml (name may include subdirs, without the extension)."""
    with open(REPO_ROOT / "configs" / f"{name}.yaml") as f:
        return yaml.safe_load(f)
