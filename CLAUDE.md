# Context for Claude: read this first

## What this repo is
A clean reimplementation of the weakly supervised ephemeral-gully (EG) detection pipeline, built for a **CVPR 2027 submission** on a ~3-week timeline. The plan, the audit of the old code, the experiment matrix and the paper revisions all live in `spec/` (start at `spec/README.md`). **`spec/` is the source of truth. Update it in the same commit as code changes when reality differs.**

Pipeline: open-weight VLMs label ~18k unlabeled locations (8 images each: 7 NAIP 1 m 2010–2023 + 1 MSDIS 15 cm 2023) → a weak-supervision label model (Snorkel and others) → pseudo-labels → a student on frozen backbone features → evaluation on 484 expert-labeled locations.

## Hard rules (agreed with the user)
- **Open-weight models only.** No paid APIs.
- **Test split is used only by `scripts/evaluate.py`**, once per final config. All selection (labelers, thresholds, class prior, epochs, hyperparameters) uses `dev`.
- Every paper number comes from a script + config in this repo, never from a notebook.
- Keep raw labeler outputs (JSONL); they get released.
- Commit/push only when the user asks. Commit attribution per the session's instructions.

## Where things were (previous machine; paths may differ here, so find and record them in `configs/data.yaml`)
- Old code: `/root/Gully_Detection` (GitHub repo of the same user; HEAD `511740b`). Key files are listed in `spec/01-code-audit.md` and `spec/02-architecture.md` (porting map).
- Old data paths hard-coded in the old code: pool images `/root/home/pvc/large_unlabeled_dataset/rgb_images/`, test images `/root/home/pvc/final_pos_neg_test_data_merging_25_2/all_tiles/`. File naming: `rgb_{0..6}_tile_{id}.tif`, `rgb_highres_tile_{id}.tif`.
- Ground truth: `Gully_Detection/labeling_tool/v2_results/final/agg/neg_strict_labels_majority.json` (main GT; 484 non-null: 257 pos / 227 neg). Per-annotator raw scores are in the same folder (3 annotators; refer to them by role, not name, in paper/spec).
- Legacy labeler outputs: `Gully_Detection/weak-supervision/large_dataset/results/` (pool) and `.../results/test_data/` (test).
- Paper (Overleaf-synced git repo, WACV 2026 kit): `/root/papers/weak-supervision-gully/68654fca3ff58f80fea5c916/wacv-2026-author-kit-template/`. Present on this machine; push access is configured via the remote URL (token stored in that repo's `.git/config`, never copy it elsewhere).

## Status as of 2026-10-22
- Done: paper review, cost/feasibility discussion, `spec/` written. **No code yet.**
- Decisions made (see `Decision:` in `spec/00-roadmap.md`): D-1 fixed dev/test; D-2 buffer = 2 tile widths (128 px at 1 m, so 256 m); D-3 resolved: 80 GB A100s, original E3 list stands; D-4 ~100-location random-pool annotation approved (must start by D5; prepare the annotation set early); D-5 decide later. Official CVPR 2027 dates still pending.
- Data located and recorded in `configs/data.yaml`; B1 confirmed from filenames. Cluster access works (`export PATH="$HOME/.krew/bin:$PATH"`; manifests in `cluster/`).
- Standing instruction: after each step, update the specs, then commit and push (user-approved).
- GPUs: A100 80 GB on Nautilus (D-3 resolved; see `spec/00-roadmap.md` Compute plan).
- D1 skeleton done: `pyproject.toml` (extras: vlm, student, ws, wandb, dev), `src/gully/` packages, `configs/` dirs, ruff+pytest CI, `uv venv .venv`. Heavy deps are extras, not installed yet.
- Next: D1-D2 `scripts/build_index.py`, `build_gt.py`, `legacy.py` (needs data access from a cluster pod), then B3 check and `make_splits.py`.

## First things to do on the data machine
1. Locate the data (pool tiles, test tiles, DEM tiles from `dem2so`) and the old repo; check GPU types/memory (`nvidia-smi`). Record them in `configs/data.yaml` and in this file.
2. **Verify audit item B1:** in the old student, `sorted(tile_files)[1]` was treated as the high-res image. Check the actual filenames in both pool and test dirs (the annotation tool keys test images `0..6, 11`).
3. **Verify B3:** how the paper's 0.70 student accuracy was chosen (test metrics were logged every epoch; check the wandb project).
4. Then follow `spec/00-roadmap.md` from D1.

## User preferences seen so far
- Wants candid, reviewer-style assessments with clear priorities.
- Tight deadline: favour finishing P0/P1 items over polish; follow the cut list in the roadmap.
