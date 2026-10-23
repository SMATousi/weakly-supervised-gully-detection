# 00 — Roadmap (3 weeks)

**Day 1 = the kickoff day.** The dates below assume kickoff Fri 2026-10-23, so D21 = Thu 2026-11-12. **Replace them with the official CVPR 2027 dates** (abstract registration is usually ~1 week before the paper deadline; supplementary is usually ~1 week after).

Owner key: **C** = Claude (code, analysis, drafting), **H** = human only (data host, GPUs, annotators, author decisions), **GPU** = long unattended job.

## Compute plan
- **Nautilus, any GPU that fits** (user decision, updated 2026-10-22; originally A100 only). Big VLMs still need 80 GB A100s with tensor parallelism; small jobs (features, students) take whatever is free. A100 notes: Select with `nodeAffinity` on `nvidia.com/gpu.product`: `NVIDIA-A100-SXM4-80GB` (nodes with 4-8 GPUs: SDSC, Mizzou fiona, OSU, KSU, USD, UNL, ...) or `NVIDIA-A100-80GB-PCIe` (SDSU, Fullerton). About 140 full 80 GB A100s exist cluster-wide. Avoid `A100-PCIE-40GB` (mostly 1 GPU per node, so no tensor parallel) and the `MIG-1g.10gb` slices (SDSU).
- **Request A100s as `nvidia.com/a100`** (A100 nodes expose that resource, not `nvidia.com/gpu`). Asking for `nvidia.com/gpu` plus a big-memory selector only matches a few reserved L40 nodes, so those jobs never schedule.
- Namespace `gpn-mizzou-vigir` cannot use H100/H200/GH200 (quota 0); max 200 pods.
- VLM labeling (vLLM): one pod per model with TP=2 or 4 on one 80 GB node, or several 1-GPU pods for models <=30B; shard the pool by index as an Indexed Job. Student runs on cached features fit one A100 (or any smaller GPU).
- Free-GPU counts are not visible to us; expect queueing. Submit the long VLM jobs early and make them resumable.
- **Critical path: the full-pool VLM runs.** Everything else is minutes-scale once features are cached.

## Progress (updated 2026-10-22; `[x]` done, `[ ]` open, "(running)" = in progress)

**Waiting on H (blocks something):**
- [ ] Give the annotators the D-4 bundle and tool (`ali-vol-500g:gully/annotation_d4.tar`, `results/D4/sample.csv`); all three should label the same 100 IDs. Must start by D5.
- [ ] Legacy single-question prompt (not in the old repo), or confirm we use the new `sq_v1`.
- [ ] wandb project / per-epoch test metrics export, to settle B3.
- [ ] Official CVPR 2027 dates (replace the placeholder dates above).
- [ ] Dev/test border: keep as is (closest pair 27 m) or add a buffer (see `03-experiments.md` results log).

## Week 1: foundation and long jobs (D1-D7)

Owner: **C** Claude, **H** human, **GPU** unattended job.

**D1: skeleton and data**
- [x] C: repo skeleton, `pyproject.toml`, configs, ruff + pytest CI (`pip install -e .` works)
- [x] H: data host access; GPU memory confirmed (80 GB A100s on Nautilus); DEM covers all 18,179 pool locations; paths in `configs/data.yaml`
- [x] C: `build_index.py` (roles by filename, prefixed ids): 18,179 pool + 548 labeled rows
- [x] C: `build_gt.py` (4 schemes, majority, Fleiss kappa): matches the legacy files, 257 / 227
- [x] C: `legacy.py` ingestion of 6 legacy labelers; reproduces the audit accuracies

**D2: verification and splits**
- [x] C: verify **B1** (high-res image never reached the high-res branch, in pool or test)
- [ ] C+H: verify **B3** (how 0.70 was picked); blocked on the wandb export
- [x] C: `make_splits.py`: dev 150 / test 334, 1 km blocks; **E1** done (pool is >= 1.8 km from every labeled location; 0 excluded)

**D2-D3: labeler runner and E0**
- [x] C: `labelers/vlm.py` (multi-image + collage, JSON output, P(yes), resumable JSONL); helpers unit-tested
- [x] C: run `vlm.py` on vLLM for real: smoke test passed (Qwen2.5-VL-7B, 16 dev locations, JSON + P(yes) work; 0.49 s/location on an A10). Model loading from the PVC is slow (~35 MB/s)
- [ ] C: pin Hugging Face revisions in `configs/labelers/*.yaml`
- [x] C: **E0, label-model step**: old pseudo-labels reproduced 100%; B4 leak barely matters; Snorkel = best single LF on dev
- [ ] C: **E0, student step**: legacy-equivalent student vs fixed (B1/B2/B5/B7) on cached features
- [ ] H+C: **Gate G1**: pipeline reproduces, leakage quantified, go/no-go on split design

**D3-D4: labeler screening and launch**
- [x] GPU: prompt sweep on dev (`sq_v1` degenerate, always "yes"; Qwen3-VL-32B-FP8 with `sq_v4` chosen on dev: ROC-AUC 0.75, PR-AUC 0.80)
- [ ] GPU: **E2** collage vs multi-image (Qwen2.5-VL-72B on dev)
- [ ] GPU: **E3 dev screening** of 5-7 labelers on the 548 labeled locations, scored on dev only (running: Qwen3-VL-32B, Mistral-Small-3.2-24B, Qwen2.5-VL-72B; `scripts/screen_labelers.py` written; InternVL3-78B, GLM-4.5V still to submit)
- [ ] C+H: select the labelers for the full pool
- [ ] GPU: launch the pool + labeled-set runs back-to-back (queue running)

**D4-D5: features**
- [x] C: `student/features.py` + `scripts/extract_features.py` (tested; reads 8 files concurrently)
- [x] GPU: FlexiViT-S features for pool + labeled (18,727 locations, 7.5 GB, `ali-vol-500g:gully/features/flexivit_s`)
- [ ] GPU: ViT-S features (E10)
- [ ] GPU: one RS foundation-model backbone (E10; cut-list item 4)

**D4-D6: cheap labelers**
- [ ] C: **E4** CLIP / RemoteCLIP / GeoRSCLIP labelers
- [ ] C: **E5** terrain labeler from the DEM tiles (P2). Blocked: DEM exists for the 18,179 pool locations only, none for the 548 labeled ones, so it cannot be validated on dev. Cut unless DEM for the labeled tiles can be obtained

**D5-D7: student**
- [x] C: `student/heads.py` (linear, mean-pool MLP, temporal attention) and `train.py` (dev-based selection), toy-tested
- [x] C: `evaluate.py` (only reader of test labels; bootstrap CIs, McNemar), toy-tested
- [x] C: `fit_label_model.py` (pseudo-labels with a dev-only class prior) and `run_grid.py` (many students from one GPU-resident copy of the features)
- [ ] C: `predict_student.py` (scores for dev / test locations from a saved model)
- [x] C: **E7** and **E11** on legacy labels, first pass (4 pseudo-label sets x 3 heads x 1 seed): done, see `03-experiments.md` results log
- [ ] C: repeat with 5 seeds + bootstrap CIs and a dev-chosen threshold
- [ ] H+C: **Gate G2**: new labelers beat or diversify the old set on dev? temporal head >= linear head on dev?

**D-4 prevalence annotation**
- [x] C: sample 100 random pool locations (`results/D4/sample.csv`) and build the tool-ready bundle
- [ ] H: annotators label the 100 locations
- [x] C: `scripts/analyze_d4.py` ready (prevalence with Wilson CI under all 4 GT schemes; metrics of any pseudo-label / score file against the new labels); tested on synthetic annotations
- [ ] C: run `analyze_d4.py` on the real annotations once they exist

## Week 2: the experiment matrix (D8-D15)

- [x] C: E6 code ready and tested on synthetic data: `ws/label_models.py` (MV, Snorkel, Dawid-Skene, our triplet-method reimplementation), `ws/confidence_lfs.py` (dev-tuned abstain bands, soft logit aggregator; fixes B10). FlyingSquid itself is not used (needs an outdated pgmpy API); EBCC/MeTaL/NPLM/LELA not done (cut list 3)
- [x] C: `scripts/run_e6.py` (binary vs confidence-valued, dev-only scoring, dev class prior) and `scripts/merge_labels.py` ready; smoke-tested on the legacy labelers
- [x] GPU: Qwen3-VL-32B-FP8 `sq_v4` pool run (18,179 pool + 548 labeled, merged)
- [ ] GPU: more strong, diverse labelers (Mistral-Small-3.2-24B, Qwen2.5-VL-72B-AWQ, Qwen3-VL-30B-A3B-FP8) with the `sq_v4` prompt; dev screening first
- [x] C: merged shards; **E6** on {legacy 3 + Qwen3-VL-32B}: PR-AUC 0.69-0.71 -> 0.75-0.79 on dev (`03-experiments.md`)
- [x] C: **E7/E11** first full grid on the new pseudo-labels (7 label sets x 2 heads x 3 seeds): single-labeler distillation (32B) beats every WS aggregate on dev; temporal head best on PR-AUC. Does not yet support the WS claim
- [ ] D9-D10 C: **E7** WS vs distillation; **E11** heads with the final LF set; 5 seeds each
- [ ] D10-D11 C: **E10** ablations (backbone, resolution, dates); **E12** Reece-supervised + supervised CV upper bound
- [ ] D11-D12 C: **E8** #LF curve; **E9** data scaling
- [ ] D12-D13 C/GPU: **E14** efficiency; **E15** qualitative figures + attention maps; GT-robustness table
- [ ] D13-D14 C: P2 only if all green: E13 self-training, Qwen3-VL-235B
- [ ] D14 C: final test evaluation: one `evaluate.py` run per final config; bootstrap CIs, McNemar
- [ ] D15 H+C: **Gate G3, experiment freeze**; tables regenerate from `results/` with one command (`scripts/make_tables.py`, `eval/report.py` written; main test table plugs in once `results/final/*.json` exist)

## Week 3: writing (D16-D21)

- [ ] D16 C draft, H decide: CVPR template; restructure per 04-paper-revisions R0-R1; decide the title
- [ ] D16-D17 C draft, H review: method + dataset sections (R3, R4); new pipeline figure
- [ ] D17-D18 C draft, H review: experiments + discussion with the new tables (R5, R6)
- [ ] D18 C draft, H check citations: related work expansion (R2)
- [ ] D19 C: supplement (prompts, labeler configs, GT schemes, extra ablations); release README
- [ ] D19 H: register the abstract (if not already done)
- [ ] D20 H: full read-through by all authors; anonymity check (no links, no names in PDFs/metadata)
- [ ] D21 H: buffer, then submit

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
- **D-1** Split: fixed dev/test (recommended, simpler) vs nested CV over the 484 labeled locations (more stable, ~2× compute and complexity) (Decision: fixed dev/test).
- **D-2** Exclusion buffer distance (256 m proposed ≈ 2 tile widths)(Decision: 2 tile widths).
- **D-3** Exact candidate list for E3, given the actual GPU memory(Decision: use the cluster; resolved 2026-10-22: 80 GB A100s, see Compute plan and `03-experiments.md`).
- **D-4** Optional: the annotators label ~100 *random* pool locations (≈2–3 h each with the existing tool) to estimate real prevalence and realistic precision. High value for reviewers, costs annotator time. Must start by D5 to be usable (Decision: approved).
- **D-5** Venue fallback if G2 fails (e.g., CVPR EarthVision workshop) (Decision: We'll decide if we get there).
