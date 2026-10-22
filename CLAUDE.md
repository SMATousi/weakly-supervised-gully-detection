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
- `scripts/build_gt.py` + `gully/data/gt.py` done: all 4 GT schemes reproduce the legacy majority files exactly (257/227 main); rules in `spec/01-code-audit.md` §5. Old repo is at `/root/Gully_Detection` on this machine; raw scores copied to `data/gt_raw/` (annotators by role).
- `scripts/build_index.py` done (run in a cluster pod; slow on CephFS, needs a process pool). `artifacts/index/locations.parquet` (18,727 rows) is local only (git-ignored); data facts in `spec/01-code-audit.md` §6 (alpha band, 59 truncated high-res tiles, variable high-res sizes).
- `legacy.py` + `scripts/ingest_legacy.py` done: 6 legacy labelers as JSONL in `artifacts/labels/`; audit accuracies (0.653 / 0.607 / 0.568 at cov 0.876) reproduce exactly (`tests/test_legacy.py`).
- `scripts/make_splits.py` done: dev 150 / test 334 / labeled_null 64; **E1: pool is >=1.8 km from every labeled location, so 0 pool locations excluded** (`results/E1/`). Closest dev-test pair 27 m (see `spec/03-experiments.md` results log).
- `ws/label_matrix.py`, `ws/label_models.py` (MV, Snorkel), `eval/metrics.py`, `scripts/run_e0.py` done. E0 label step: old pseudo-labels reproduced 100%; the B4 leak barely changes anything (class balance 0.47 vs dev 0.47); Snorkel dev acc 0.673 = best single LF (see `spec/03-experiments.md` results log).
- B3 NOT verifiable locally (only a 2024-07 wandb run exists); needs the user's wandb project.
- Next: `student/` (features, heads, train) for the E0 student comparison and E7/E11; copy the index to a writable PVC dir for cluster jobs; D-4 annotation sample list.

## First things to do on the data machine
1. Locate the data (pool tiles, test tiles, DEM tiles from `dem2so`) and the old repo; check GPU types/memory (`nvidia-smi`). Record them in `configs/data.yaml` and in this file.
2. **Verify audit item B1:** in the old student, `sorted(tile_files)[1]` was treated as the high-res image. Check the actual filenames in both pool and test dirs (the annotation tool keys test images `0..6, 11`).
3. **Verify B3:** how the paper's 0.70 student accuracy was chosen (test metrics were logged every epoch; check the wandb project).
4. Then follow `spec/00-roadmap.md` from D1.

## User preferences seen so far
- Wants candid, reviewer-style assessments with clear priorities.
- Tight deadline: favour finishing P0/P1 items over polish; follow the cut list in the roadmap.
