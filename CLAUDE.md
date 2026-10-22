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
- Paper (Overleaf-synced git repo, WACV 2026 kit): `/root/papers/WS+Gully/68654fca3ff58f80fea5c916/wacv-2026-author-kit-template/`. It may not exist on this machine; ask the user if it's needed.

## Status as of 2026-10-22
- Done: paper review, cost/feasibility discussion, `spec/` written. **No code yet.**
- Not started: roadmap D1. The user will say when to start.
- Waiting on the user: the open decisions D-1 to D-5 in `spec/00-roadmap.md` (split design, buffer distance, labeler list given GPU memory, the optional 100-location prevalence annotation, fallback venue) and the official CVPR 2027 dates.

## First things to do on the data machine
1. Locate the data (pool tiles, test tiles, DEM tiles from `dem2so`) and the old repo; check GPU types/memory (`nvidia-smi`). Record them in `configs/data.yaml` and in this file.
2. **Verify audit item B1:** in the old student, `sorted(tile_files)[1]` was treated as the high-res image. Check the actual filenames in both pool and test dirs (the annotation tool keys test images `0..6, 11`).
3. **Verify B3:** how the paper's 0.70 student accuracy was chosen (test metrics were logged every epoch; check the wandb project).
4. Then follow `spec/00-roadmap.md` from D1.

## User preferences seen so far
- Wants candid, reviewer-style assessments with clear priorities.
- Tight deadline: favour finishing P0/P1 items over polish; follow the cut list in the roadmap.
