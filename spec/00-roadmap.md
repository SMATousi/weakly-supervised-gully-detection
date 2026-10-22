# 00 — Roadmap (3 weeks)

**Day 1 = the kickoff day.** The dates below assume kickoff Fri 2026-10-23, so D21 = Thu 2026-11-12. **Replace them with the official CVPR 2027 dates** (abstract registration is usually ~1 week before the paper deadline; supplementary is usually ~1 week after).

Owner key: **C** = Claude (code, analysis, drafting), **H** = human only (data host, GPUs, annotators, author decisions), **GPU** = long unattended job.

## Compute plan
- 4×A100: VLM labeling (vLLM) on D3–D9; free afterwards for feature extraction and E14.
- 8×A10: feature extraction (D4–D5), then every student run.
- **Critical path: the full-pool VLM runs.** Everything else is minutes-scale once features are cached.

## Week 1: foundation and long jobs (D1–D7)

| Day | Task | Owner | Done when |
|---|---|---|---|
| D1 | Repo skeleton, env, configs (02-architecture) | C | `pip install -e .` works; CI lint |
| D1 | Give access to the data host; confirm GPU memory (A100 40 vs 80 GB); confirm DEM coverage of the 3 pool HUC12s | H | Paths in `configs/data.yaml` |
| D1–D2 | `build_index.py` (roles by filename, prefixed ids), `build_gt.py` (4 schemes, majority, κ), `legacy.py` ingestion | C | Index has 18,179 pool + 484 labeled rows; GT counts match 01-audit §3 |
| D2 | **Verify B1** (which image is high-res at index 1) and **B3** (how 0.70 was picked) | C+H | Written answer in 01-audit |
| D2 | `make_splits.py`: spatially blocked dev/test, 256 m pool exclusion (E1) | C | Split stats committed |
| D2–D3 | `labelers/vlm.py` on vLLM: multi-image + collage modes, JSON output, P(yes), resumable JSONL | C | Throughput measured on 50 locations |
| D3 | **E0** reproduce the old pipeline from legacy labels; then with B2/B4/B5 fixed | C | Old vs fixed numbers in `results/E0` |
| D3 | **Gate G1** | H+C | Pipeline reproduces; leakage quantified; go/no-go on split design |
| D3–D4 | E2 (collage vs multi-image) + **E3 dev screening** of 5–7 labelers | GPU | Dev metrics + pairwise error correlation |
| D4 | Select the labelers for the full pool; launch the pool + test runs back-to-back | C+H, GPU | Queue running |
| D4–D5 | `features.py`: extract frozen features (FlexiViT-S fixed, ViT-S, one RS foundation model) for pool+dev+test | GPU (A10) | Feature files + checksums |
| D4–D6 | E4 CLIP labelers; E5 terrain labeler **only if** DEM is available | C | Dev metrics |
| D5–D7 | `student/train.py` + `heads.py` (linear, mean-pool MLP, temporal attention); E7 and E11 on **legacy** labels for early signal | C | Seeds × heads run end-to-end |
| D7 | **Gate G2**: do the new labelers beat or diversify the old set on dev? Is the temporal head ≥ the linear head on dev? | H+C | Decide the final LF set and head |

## Week 2: the experiment matrix (D8–D15)

| Day | Task | Owner |
|---|---|---|
| D8–D9 | Pool runs finish; build L; **E6** label models (binary vs confidence LFs) | GPU, C |
| D9–D10 | **E7** WS vs distillation; **E11** heads, final LF set; 5 seeds each | C |
| D10–D11 | **E10** ablations (backbone, resolution, dates); **E12** Reece-supervised + supervised CV upper bound | C |
| D11–D12 | **E8** #LF curve; **E9** data scaling | C |
| D12–D13 | E14 efficiency; E15 qualitative figures + attention maps; GT-robustness table | C, GPU |
| D13–D14 | P2 items only if everything above is green: E13 self-training, Qwen3-VL-235B | C |
| D14 | Final test evaluation: one `evaluate.py` run per final config; bootstrap CIs, McNemar | C |
| D15 | **Gate G3, experiment freeze.** Tables regenerate from `results/` with one command | H+C |

## Week 3: writing (D16–D21)

| Day | Task | Owner |
|---|---|---|
| D16 | CVPR template; restructure per 04-paper-revisions R0–R1; decide the title | C draft, H decide |
| D16–D17 | Method + dataset sections (R3, R4); new pipeline figure | C draft, H review |
| D17–D18 | Experiments + discussion with the new tables (R5, R6) | C draft, H review |
| D18 | Related work expansion (R2) | C draft, H check citations |
| D19 | Supplement: prompts, labeler configs, GT schemes, extra ablations; release README | C |
| D19 | Register the abstract (if not already done) | H |
| D20 | Full read-through by all authors; anonymity check (no links, no names in PDFs/metadata) | H |
| D21 | Buffer, then submit | H |

## Cut list (in order, when behind)
1. E13 self-training, Qwen3-VL-235B
2. E5 terrain labeler
3. Extra WS frameworks beyond MV / DS / Snorkel / FlyingSquid
4. A second RS foundation-model backbone
5. E2 (fix multi-image, no comparison)
6. E9 down to 3 points (2k / 6k / all)

**Never cut:** E0, E1, E7, the dev/test discipline, CIs + McNemar, E3 with at least 2 new labelers.

## Risks
| Risk | Mitigation |
|---|---|
| Fixing B1–B5 lowers the old headline numbers | Expected and fine: the new comparison is internally consistent; E0 documents it. Don't chase the old 0.70. |
| New labelers aren't better than Qwen2.5-VL-72B | WS still benefits from *different* errors; CLIP/terrain LFs are cheap and orthogonal; the paper can report the negative result honestly. |
| GPUs shared or 40 GB | Use the ≤38B models; screen on dev first so we never run a useless model on the full pool. |
| Dev split too small (~150) for stable selection | Keep the selection space small (few hyperparameters); use bootstrap on dev for the gates. |
| Temporal head doesn't beat the simpler heads | Report it; fall back to the "dataset + benchmark + WS study" framing; the method claim shrinks to confidence-valued LFs. |

## Open decisions (need an answer from H)
- **D-1** Split: fixed dev/test (recommended, simpler) vs nested CV over the 484 labeled locations (more stable, ~2× compute and complexity).
- **D-2** Exclusion buffer distance (256 m proposed ≈ 2 tile widths).
- **D-3** Exact candidate list for E3, given the actual GPU memory.
- **D-4** Optional: the annotators label ~100 *random* pool locations (≈2–3 h each with the existing tool) to estimate real prevalence and realistic precision. High value for reviewers, costs annotator time. Must start by D5 to be usable.
- **D-5** Venue fallback if G2 fails (e.g., CVPR EarthVision workshop).
