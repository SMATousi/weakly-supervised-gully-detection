# weakly-supervised-gully-detection

Weakly supervised ephemeral-gully detection: open-weight VLM labelers -> label model -> student on frozen
backbone features, evaluated on 484 expert-labeled locations.

Plan, audit, experiment matrix and paper revisions: [`spec/`](spec/README.md). Project context: [`CLAUDE.md`](CLAUDE.md).

## Setup
```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"          # base; add extras as needed: vlm, student, ws, wandb
ruff check . && pytest
```
Data locations and split settings: `configs/data.yaml`. Cluster manifests: `cluster/`.
