# 04 — Paper revisions

Paper source: `/root/papers/weak-supervision-gully/68654fca3ff58f80fea5c916` (Overleaf-synced; currently the WACV 2026 kit, Paper ID 2172).

**Framing:** a dataset + benchmark + weak-supervision study for multi-temporal, multi-resolution remote sensing, with a small method contribution (temporal-attention student + confidence-valued VLM labelers). All models are open-weight and reproducible.

## R0 — Format (D16)
- [ ] Move to the CVPR 2027 author kit; remove the WACV paper ID. Keep third-person self-citation (do **not** apply the "replace with we" comment in `2-Related-Works.tex:109` for the review version).
- [ ] 8-page budget; move implementation detail, prompts and GT-scheme detail to the supplement.

## R1 — Title / Abstract / Intro
- [ ] Title: "Detection" → either "Classification"/"Recognition", or justify (location-level detection). Decide at D16 based on whether E15 attention maps localize.
- [ ] Abstract: fix "18,000 high-resolution images" → 18k locations / ~145k images; drop the "teacher" terminology for the label model; state numbers with CIs.
- [ ] Soften "most prominent cause of soil erosion" (`1-Introduction.tex:79`) and the "first" claims (permanent-gully DL mapping exists).
- [ ] Contributions: (1) dataset & benchmark; (2) open-weight VLM-labeler + WS pipeline with confidence-valued LFs; (3) temporal-attention student; (4) extensive analysis (WS vs distillation, scaling, labeler count). Drop "we use FlexiViT" as a contribution.

## R2 — Related work
- [ ] Add DL gully / erosion mapping (UAV, DEM, Loess Plateau, etc.).
- [ ] Add VLMs as annotators / pseudo-labeling and distillation from foundation models.
- [ ] Add remote sensing foundation models (SatMAE, Scale-MAE, Prithvi, DOFA, SkySense, RemoteCLIP, GeoChat/EarthDial).
- [ ] Add multi-temporal RS classification / change detection.
- [ ] Remove "dense, pixel-level annotation" (`2-Related-Works.tex:117`); the task is at location level.

## R3 — Method
- [ ] Describe VLM input exactly (multi-image vs collage, resolution), and the structured output + P(yes).
- [ ] Notation: K = locations, C = classes; noise-aware loss sums over k (locations), not i=1..m; fix the Snorkel factor indexing; number all equations.
- [ ] The student as actually built: frozen backbone, temporal-attention head (new); the patch sizes per resolution and the real image sizes (B8).
- [ ] Remove "MLP" wording if the linear head remains in any baseline (B12).

## R4 — Dataset
- [ ] Image roles and years (7 NAIP bins 2010–2023 + MSDIS 2023).
- [ ] Spatial split, exclusion buffer, # excluded (E1).
- [ ] GT: 3 annotators, 5-level scores, the per-annotator rule **then majority** (B12); Fleiss' κ; fix level "0" → "4" (`5-Discussion.tex:453`).
- [ ] Prevalence: test is balanced by design; discuss the shift vs the pool (B11).
- [ ] Release: images, GT, **all raw labeler outputs**, code, configs.

## R5 — Experiments (tables/figures to produce)
| Item | Source |
|---|---|
| Table: labelers on test (acc/P/R/F1/NPV/AUC ± CI, coverage) | E3, E4, (E5) |
| Table: WS frameworks with the new LF set, binary vs confidence LFs | E6 |
| **Main table**: best labeler, LM, student on {single-best, MV, Snorkel hard, Snorkel soft}, Reece-supervised, ours (temporal head), supervised CV upper bound; ± CI, McNemar | E7, E11, E12 |
| Ablation table: backbone, inputs, dates, head | E10, E11 |
| Figures: #LFs curve, data-scaling curve, LF correlation heatmap | E8, E9, E3 |
| Efficiency table | E14 |
| Qualitative figure(s) | E15 |
| GT-robustness table (4 schemes) | protocol |

## R6 — Discussion / errors to fix regardless
- [ ] "Qwen2.7-VL" → Qwen2.5-VL (`4-2-Experiments.tex:403,414`); cite the Qwen2.5-VL report, not `bai2023qwen` (Qwen-VL).
- [ ] `tab:gt_metrics`: F1 inconsistent with P/R. Recompute with the new `build_gt.py`, and say whether it's macro-averaged.
- [ ] `tab:ws_ablation_gully`: sentence outside `\caption{}` (`4-2-Experiments.tex:365`).
- [ ] Remove the wrong claim that the label model "reduces to voting" (`5-Discussion.tex:498`).
- [ ] Typos: "an student", "my have outpeformed", "detection EGs", "dataset represent".
- [ ] All numbers regenerated from `results/` (old numbers are invalid until E0 shows the bug fixes don't move them).
