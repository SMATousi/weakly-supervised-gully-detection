# Spec — CVPR 2027 revision of "Weakly Supervised Ephemeral Gully Detection"

This folder is the single source of truth for the 3-week revision. Read in order:

| File | What it is |
|---|---|
| [00-roadmap.md](00-roadmap.md) | Day-by-day plan, gates, cut list, open decisions |
| [01-code-audit.md](01-code-audit.md) | What the old code (`/root/Gully_Detection`) actually does, and the bugs that affect reported numbers |
| [02-architecture.md](02-architecture.md) | Target layout of this repo, data contracts, porting map from old code |
| [03-experiments.md](03-experiments.md) | Experiment matrix (IDs E0–E12), protocol, which paper table each feeds |
| [04-paper-revisions.md](04-paper-revisions.md) | Paper changes, section by section, including fixes the audit forces |

## Ground rules

1. **Open-weight models only.** Every labeler must be runnable by anyone with GPUs (no paid APIs).
2. **The test split is touched only by `scripts/evaluate.py`.** No LF selection, threshold, epoch, class prior or hyperparameter may come from it.
3. **Every number in the paper is produced by a script in this repo** from a config in `configs/`, with seed(s) recorded. No numbers from notebooks.
4. **Raw labeler outputs are kept** (JSONL with prompt, model revision, raw text, parsed label, confidence) and get released with the dataset.
5. When something in this spec turns out to be wrong, fix the spec in the same commit as the code.

## Status

- [x] Kickoff approved
- [ ] Gate G1: Baseline reproduced in the new repo, bugs fixed, leakage checked (end of D3)
- [ ] Gate G2: New labelers add value on dev (end of D7)
- [ ] Gate G3: Experiment freeze (end of D15)
- [ ] Submission
