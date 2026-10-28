# 03 — Experiments

## Protocol (applies to every experiment)

- **Splits.** The 484 labeled locations are split **spatially blocked and stratified** into `dev` (~150) and `test` (~334): grid blocks of 1 km, assigned whole to one split, seed fixed in `configs/data.yaml`. Pool locations within **256 m** of any dev/test centroid become `pool_excluded` and are never used for training (B11/leakage). Report how many were excluded.
- **dev is used for:** labeler selection, prompt/image-mode choice, label-model class prior and hyperparameters, CLIP thresholds, student epoch/threshold/hyperparameters.
- **test is used once per final configuration**, by `scripts/evaluate.py`.
- **Metrics:** Accuracy, Precision, Recall, F1, NPV, ROC-AUC, PR-AUC. 95% CIs by 2,000× bootstrap over test locations. Paired McNemar test between the final student and (a) best single labeler, (b) the label model.
- **Seeds:** students 5 seeds; report mean ± std, and use a 5-seed majority vote for McNemar.
- **GT schemes:** the main GT is `y_strict_neg`; Table "GT robustness" re-evaluates the final rows under all 4 schemes.

Each experiment writes `results/<E-id>/<run>.json` (config hash, git sha, seed, metrics). `make_tables.py` builds the LaTeX.

## Matrix

Priority **P0** = without it the paper is not defensible; **P1** = expected by CVPR reviewers; **P2** = cut first.

| ID | P | Question | Runs | Feeds |
|---|---|---|---|---|
| E0 | P0 | Can the new repo reproduce the old pipeline, before and after the bug fixes? | legacy LFs → Snorkel (with Y_dev=test vs dev prior) → student old-code-equivalent vs fixed | Sanity only; a footnote if numbers move |
| E1 | P0 | Leakage audit | distances pool↔test, HUC overlap, # excluded | Sec. 4 (dataset) text |
| E2 | P1 | Does image presentation matter for VLM labelers? | Qwen2.5-VL-72B on dev: collage-300 vs multi-image (native high-res + 7 NAIP) | Discussion; picks `image_mode` for E3 |
| E3 | P0 | New open labelers (dev screening, then full pool) | 5–7 candidates on dev (see below); keep the ones that are accurate or add *different errors*; run the kept ones on pool + test | Table 1 (labelers), Fig. LF-correlation |
| E4 | P1 | Cheap non-chat labelers | RemoteCLIP / GeoRSCLIP zero-shot, prompt ensemble, max over dates, abstain band tuned on dev | Table 1 |
| E5 | P2 | Terrain labeler | DEM flow accumulation / TWI within field → rule with abstain | Table 1 |
| E6 | P0 | Label models with the new LF set | MV, Dawid–Skene, Snorkel, triplet method (own reimplementation of the FlyingSquid estimator; the package does not install), (EBCC, MeTaL, NPLM, LELA only if time allows); binary vs confidence-valued LFs | Table WS-frameworks |
| E7 | P0 | **Is weak supervision better than plain distillation?** | Student trained on: best single labeler (hard), MV (hard), Snorkel (hard), Snorkel (soft) | Main table |
| E8 | P1 | # labelers curve | Snorkel + student with LF subsets of size 2…k (random orders, mean ± std) | Figure |
| E9 | P1 | Unlabeled-data scaling | student on 1k / 2.5k / 5k / 10k / all pool | Figure |
| E10 | P1 | Backbone & input ablations | FlexiViT-S (fixed, B1) vs ViT-S + resize vs **one** RS foundation model (DOFA or Prithvi); high-res only / NAIP only / both; 1 / 4 / 8 dates | Ablation table |
| E11 | P0 | **Method: temporal-attention head** | Linear-on-flattened (legacy) vs mean-pool MLP vs temporal attention over per-date tokens (+ date & resolution embeddings) | Main table + ablation |
| E12 | P1 | Supervised references | student trained on Reece et al. labels; 5-fold CV supervised on the expert labels (upper bound, reported separately) | Main table |
| E13 | P2 | Self-training | one round: student P(yes) as an additional LF, or relabel the pool | Ablation |
| E14 | P1 | Efficiency | params, GPU-ms/location: student vs 72B VLM (vLLM, same GPU) | Efficiency table |
| E15 | P1 | Qualitative | TP/FP/FN/TN examples with all 8 dates; attention-over-dates maps from E11 | Figures |

## E3 candidate labelers (open weights; pick ~4 to run on the full pool)

Pin exact HF revisions in `configs/labelers/`. Substitute a newer open release only if it beats the listed one on dev.

| Labeler | Size / fits 4×A100-80G | Expected cost on the pool* |
|---|---|---|
| Qwen2.5-VL-72B-Instruct (re-run via vLLM with P(yes)) | bf16, TP=4 | 6–12 h |
| Qwen3-VL-32B-Instruct (or 30B-A3B) | bf16, TP=2 | 2–4 h |
| InternVL3-78B or InternVL3.5-38B | bf16 | 3–12 h |
| GLM-4.5V (MoE, ~12B active) | bf16, TP=4 | 3–6 h |
| Gemma-3-27B-it | bf16 | 2–3 h |
| Llama-3.2-90B multi-question (legacy outputs reused) | — | 0 |
| Qwen3-VL-235B-A22B (AWQ) | tight | 8–15 h, P2 |

\*18.2k locations, single question, multi-image, vLLM batching; measure on dev first and update this table. **GPU basis (D-3, resolved):** 80 GB A100s (SXM4 nodes with 4-8 GPUs, PCIe nodes with 4), so the 72-78B models run bf16 at TP=4 and the table above stands. Do not schedule on the 40 GB PCIe A100s. Qwen3-VL-235B stays P2 (needs 8 GPUs on one SXM node or AWQ at TP=4).

Labeler output must be JSON (`{"answer": "yes|no|unsure", "confidence": 0-1}`), via guided decoding; also record logprob P(yes) of the first answer token.

## Method contribution (E11 + confidence LFs in E6)

1. **Temporal-attention student.** Per date, take frozen backbone tokens → a light projection → add a learned date embedding and a resolution embedding (NAIP vs high-res) → a 2-layer transformer over [CLS_t, tokens_t for all t] → classify. Motivation: gullies appear and disappear, so evidence on one date should be weighted by context from the other dates. It replaces a 199,680-dim linear probe that overfits.
2. **Confidence-valued LFs.** VLM P(yes) is thresholded on dev into {1, 0, abstain}, and/or fed as soft votes, so the label model gets more than 8 distinct posteriors (fixes B10).

Both are small enough to build in 1–2 days each, and both can be tested (E11, E6).

## Results log

### E1 leakage audit (2026-10-22, `scripts/make_splits.py` -> `results/E1/split_stats.json`)
- 484 labeled locations with a defined label -> dev 150 (53.3% positive) / test 334 (53.0%), 1 km blocks, seed 0. 64 labeled locations have no defined label (`labeled_null`, includes tile 547 with no scores).
- **The pool and the labeled set are spatially disjoint:** the nearest pool location to any labeled location is 1.8 km away (median 8.3 km). The 256 m exclusion buffer therefore removes **0** pool locations; there is no proximity leakage between pool pseudo-labels and the evaluation set. Report this as a finding, and keep the buffer in the code.
- Dev/test are separated by whole blocks, but locations on either side of a block border can be close: the closest dev-test pair is 27 m, 5% of dev locations have a test location within 66 m (median 0.8 km). Open item: keep as is (and say so), or add a dev-test buffer (costs test size). Default: keep, report.
- Pool prevalence vs test prevalence (B11) is not addressed by this; see D-4.

### E0, label-model step (2026-10-22, `scripts/run_e0.py` -> `results/E0/e0_label_model.json`; scored on dev only, n=150)
- **The old pseudo-labels are reproduced exactly:** the legacy variant (Snorkel, 3 legacy LFs, seed 12, class balance from all 484 expert labels) agrees with `train_lable_model_results.json` on 100% of the 18,179 pool hard labels. Pool coverage of all 3 LFs is 1.0 (the 60 missing multi-question labels in B9 are test-side only).
- **B4 (test leakage through `Y_dev`) has almost no effect here.** Snorkel uses `Y_dev` only to set the class balance. The dev-only balance (0.47/0.53) gives identical hard labels and posteriors within 0.006; no prior at all (0.5/0.5) gives identical hard labels and posteriors within 0.075. Dev metrics are identical in all three: acc 0.673, F1 0.726, PR-AUC 0.710. The prior matters only when it is far off (0.9/0.1 flips every pool label to negative). So the leak is a validity and reviewer-trust issue, not a numeric one; the fixed pipeline uses the dev prior and says so.
- **The prior is the real problem (B11):** the balanced prior (~0.47 positive) is a property of the test set, not the pool. The label model labels 33.3% of the pool positive.
- Majority vote of the same 3 LFs: dev acc 0.667, F1 0.719, PR-AUC 0.690 (about the same as Snorkel). Single LFs on dev: Llama-90B multi-question 0.556 (cov 0.90), Llama-90B single 0.647, Qwen-72B single 0.673, so Snorkel is no better than the best single LF on dev (0.673 = 0.673).
- B10 confirmed: the label model yields 7-8 distinct posteriors.
- Still to do for E0: the student with the legacy code path vs the fixed one (needs `student/`).

### E7 / E11 early signal on legacy labels (2026-10-22; `scripts/run_grid.py`, `results/E_early_legacy/`)
Setup: FlexiViT-S frozen features (NAIP 128 px / patch 16; high-res resized to 512 px / patch 64; 75 truncated or unreadable high-res images masked), pool train n=18,179, scored on **dev only** (n=150, threshold 0.5, epoch selected on dev PR-AUC), **one seed**, 30 epochs, AdamW lr 1e-3. No test labels were read.

| Pseudo-labels | Head | Dev acc | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Snorkel soft | linear (legacy-style) | 0.653 | 0.694 | 0.668 | 0.689 |
| Snorkel soft | mean-pool MLP | 0.680 | 0.753 | 0.685 | 0.695 |
| Snorkel soft | temporal attention | 0.667 | 0.725 | 0.707 | **0.752** |
| Snorkel hard | linear / MLP / temporal | 0.600 / 0.600 / 0.620 | 0.605 / 0.583 / 0.601 | 0.655 / 0.695 / 0.695 | 0.696 / 0.708 / 0.731 |
| MV hard | linear / MLP / temporal | 0.600 / 0.687 / 0.620 | 0.600 / 0.740 / 0.632 | 0.661 / 0.695 / 0.710 | 0.695 / 0.706 / 0.745 |
| Qwen-72B only (hard) | linear / MLP / temporal | 0.653 / 0.687 / 0.593 | 0.720 / 0.768 / 0.690 | 0.690 / 0.654 / 0.613 | 0.673 / 0.620 / 0.656 |

Reference on dev: Snorkel label model acc 0.673 / F1 0.726 / PR-AUC 0.710; Qwen-72B alone acc 0.673; MV 0.667.

What this does and does not show:
- **No student clearly beats its teachers on dev** (best acc 0.687 vs 0.673). With n=150 one standard error on accuracy is ~0.04 and there is one seed, so none of the accuracy differences above are significant. Weak teachers (dev acc ~0.67) cap what distillation can reach.
- **Suggestive, to be tested with 5 seeds + bootstrap:** (a) Snorkel *soft* labels beat *hard* labels on accuracy (0.667 avg vs 0.607) and F1 for all three heads; (b) the temporal head has the best PR-AUC in all three soft/MV groups (0.752 / 0.745 / 0.731, vs 0.69-0.71) but not better accuracy at threshold 0.5, so the gain is in ranking and may need a dev-chosen threshold; (c) the linear head peaks within the first 5 epochs (overfits its ~200k-dim input).
- B10 matters here: Snorkel on 3 binary LFs gives only 7 distinct soft labels, yet soft still helped. More informative (confidence-valued) LFs should widen this gap.
- Next: 5 seeds, a dev-chosen threshold, and the same grid on the new labelers once they exist.

### E3/E2-prompt: the first new prompt is degenerate (2026-10-23)
- `sq_v1` (long system prompt describing gullies, JSON answer) made Qwen2.5-VL-7B (16 dev smoke locations) and Qwen3-VL-8B (137 labeled locations) answer **"yes" for every location**, with P(yes) >= 0.999. Accuracy equals the base rate (54% positive), so the labeler is useless and P(yes) cannot be thresholded. Mistral-Small-3.2-24B confirmed it (64/64 "yes", P(yes) >= 0.986), so the cause is the prompt, not one model; its job and the queued Qwen2.5-VL-72B job were cancelled.
- Partial outputs are kept in `ali-vol-500g:gully/labels/degenerate_sq_v1/`. The 8B pool runs were cancelled to save GPU time.
- Next: a prompt sweep (`--prompt-ids sq_v2 sq_v3 sq_v4`, one model load) on the 548 labeled locations, scored on dev only. Pick the prompt on dev, then run the pool. Prompt is a hyperparameter selected on dev, like the labeler list.

### 8B prompt sweep on the labeled set, scored on dev (n=150) (2026-10-23, `scripts/screen_labelers.py`, `results/E3/screening_8b_prompts.json`)
| Qwen3-VL-8B prompt | Predicts positive | Acc (own answer) | ROC-AUC of P(yes) | Acc at dev-best threshold |
|---|---|---|---|---|
| sq_v2 | 96% | 0.520 | 0.641 | 0.640 |
| sq_v3 | 100% | 0.533 | 0.621 | 0.633 |
| sq_v4 | 88% | 0.627 | 0.653 | 0.647 |
- Still strongly biased to "yes" under all prompts; P(yes) is saturated (34-61 distinct values over 548 locations). An 8B model is a weak labeler here (ROC-AUC 0.62-0.65), well below Qwen2.5-VL-72B with the legacy collage prompt (acc 0.673).
- Error correlation between the three prompts is high (0.75-0.92): prompt changes add little diversity. Legacy Llama-3.2-90B errors are uncorrelated with the 8B (-0.1 to -0.3), which is what weak supervision can use.
- Next: bigger models. Qwen3-VL-32B-FP8 (35.5 GB) on 2 x 24 GB cards (`nvidia.com/gpu: 2`, easier to get than A100s); the A100 server has been pending for 6h43m. Also test `image_mode: collage` (E2).

### First useful new labeler: Qwen3-VL-32B-FP8, prompt sq_v2 (2026-10-23, dev n=150, `results/E3/screening_32b_v2.json`)
Own answer: acc 0.660, F1 0.617, precision 0.774, recall 0.513 (predicts 35% positive vs 53% in dev GT). P(yes): ROC-AUC 0.739, PR-AUC 0.798, acc 0.680 at the dev-best threshold 0.269 (optimistic: tuned on dev). Ran on 2 x 24 GB cards (`nvidia.com/gpu: 2`, FP8 weights 35.5 GB, eager mode, 0.95 GPU memory; a harmless allocator OOM warning appears on the first batch). Error correlation with legacy Qwen2.5-VL-72B 0.01, legacy Llama mq 0.03, legacy Llama sq 0.38, Qwen3-VL-8B sq_v4 -0.15: a genuinely different error profile. sq_v4 on the same model is still running; pick the prompt on dev before any pool run.

### Prompt choice for Qwen3-VL-32B-FP8 (dev n=150, `results/E3/screening_32b_prompts.json`)
| Prompt | Predicts positive | Acc | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| sq_v2 | 35% | 0.660 | 0.617 | 0.739 | 0.798 |
| sq_v4 | 51% | 0.653 | 0.667 | 0.750 | 0.801 |
Chosen on dev (PR-AUC, then F1): **sq_v4** (positive rate 51% vs 53% in dev GT; v2 under-predicts). Differences are within noise; error correlation v2/v4 is 0.66, so one prompt is enough. Pool run launched as 4 shards x 2 GPUs (`--prompt-ids sq_v4 --shard i/4`, ~1 location/s per job). Labeled-set outputs are named `<labeler>__sq_v4.labeled.jsonl`; pool shards use the same labeler name. Merge before building the label matrix.

### E6 on the combined labelers (2026-10-23, `scripts/run_e6.py`, `results/E6/e6_legacy3_q32.json`; dev n=150, dev class prior 0.47/0.53)
LFs: legacy Llama-90B multi-question, legacy Llama-90B single, legacy Qwen2.5-VL-72B single, new Qwen3-VL-32B-FP8 `sq_v4` (pool 18,179 + labeled 548, merged and count-checked).

| Label model | Acc | F1 | PR-AUC | Pool pos. rate | Distinct pool posteriors |
|---|---|---|---|---|---|
| MV (binary LFs) | 0.680 | 0.742 | 0.748 | 0.429 | 6 |
| Dawid-Skene (binary) | 0.687 | 0.764 | 0.765 | 0.546 | 17 |
| Triplet (binary) | 0.680 | 0.733 | 0.746 | 0.428 | 13 |
| Snorkel (binary) | 0.680 | 0.747 | 0.765 | 0.429 | 17 |
| MV, 32B as abstain band | 0.673 | 0.732 | 0.730 | 0.365 | 7 |
| Dawid-Skene, band | 0.687 | 0.759 | 0.789 | 0.533 | 23 |
| Triplet, band | 0.693 | 0.755 | 0.768 | 0.422 | 20 |
| Snorkel, band | 0.660 | 0.730 | **0.791** | 0.471 | 21 |
| Soft-logit aggregator (weights fit on dev, 5-fold CV on dev) | 0.640 | 0.667 | 0.749 | 0.334 | 101 |

(Old 3 labelers only: MV 0.667/0.719/0.690, Snorkel 0.673/0.726/0.710.) Band for the 32B labeler tuned on dev: abstain for P(yes) in (0.008, 0.995), covered-accuracy 0.82 at >= 50% coverage.
- Adding the 32B labeler raises PR-AUC from 0.69-0.71 to 0.75-0.79; the confidence-valued (abstain-band) variants are best on PR-AUC. Accuracy changes are within noise (one SE ~ 0.04 at n=150).
- Posteriors are far richer (17-23 distinct vs 7; B10), the soft-logit aggregator has 101, but it uses dev labels as supervision, so treat it as a supervised stacker, not weak supervision.
- **Pool prevalence is model-dependent (33-55% predicted positive)**, and the new labeler alone calls 31.5% of the pool positive vs 48% on the labeled set: prior shift (B11). The D-4 annotation is the only way to know.

### E7 / E11 on the new pseudo-labels (2026-10-23, `scripts/run_grid.py`, `results/E7_new_labelers/`; dev n=150, mean +- std over 3 seeds, threshold 0.5, epoch chosen on dev PR-AUC)
Same FlexiViT-S frozen features and training setup as the early grid. LFs: 3 legacy labelers + Qwen3-VL-32B-FP8 `sq_v4`.

| Training labels | Head | Acc | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| 32B alone, hard | temporal | 0.669 +- 0.017 | 0.631 +- 0.024 | 0.766 | **0.787 +- 0.005** |
| 32B alone, soft P(yes) | temporal | 0.662 +- 0.014 | 0.641 +- 0.041 | 0.752 | 0.782 +- 0.021 |
| Snorkel + 32B band (WS) | temporal | 0.638 +- 0.039 | 0.711 +- 0.049 | 0.717 | 0.754 +- 0.039 |
| Dawid-Skene + 32B band (WS) | temporal | 0.627 +- 0.027 | 0.701 +- 0.033 | 0.671 | 0.715 +- 0.043 |
| Majority vote, hard | temporal | 0.631 +- 0.027 | 0.679 +- 0.035 | 0.683 | 0.716 +- 0.026 |
| Snorkel (WS) | temporal | 0.629 +- 0.034 | 0.656 +- 0.097 | 0.702 | 0.715 +- 0.018 |
| Dawid-Skene (WS) | temporal | 0.593 +- 0.041 | 0.686 +- 0.016 | 0.686 | 0.705 +- 0.008 |
| 32B alone, hard | MLP | 0.671 +- 0.023 | 0.706 +- 0.036 | 0.737 | 0.719 +- 0.040 |
| Dawid-Skene + band | MLP | 0.664 +- 0.027 | 0.738 +- 0.023 | 0.660 | 0.659 +- 0.023 |
| Snorkel + band | MLP | 0.644 +- 0.010 | 0.704 +- 0.013 | 0.692 | 0.700 +- 0.028 |
(Mean over all label sets by head: MLP acc 0.649 / PR-AUC 0.690; temporal acc 0.636 / PR-AUC 0.739.)

Reading (dev only; not for the paper until the final test evaluation):
- **Plain distillation of the single best labeler (32B) gives the best students (PR-AUC ~0.78-0.79), better than every label-model aggregate (0.70-0.75), even though the label models score PR-AUC 0.75-0.79 themselves on dev.** This does NOT support the paper's claim that weak supervision beats distillation. Gaps are partly within noise (n=150, 3 seeds), but the direction is the same in every comparison.
- The temporal head wins on ranking (PR-AUC) over the MLP in 6/7 label sets, not on accuracy at 0.5.
- Hypotheses to test (E6/E7 follow-ups): (a) the legacy LFs are weak and distribution-shifted between pool and labeled set (legacy Llama-90B single: 2% positive on pool, 37% on labeled), so they inject noise into the label model; (b) the dev class prior (0.47) does not match the pool; (c) only one strong labeler exists, WS needs several good, diverse ones; (d) a dev-chosen threshold may matter more than the epoch. Next labelers: Mistral-Small-3.2-24B / Qwen2.5-VL-72B-AWQ with the `sq_v4` prompt, Qwen3-VL-30B-A3B-FP8.
- Fallback framing from the roadmap applies if WS keeps failing to beat distillation: dataset + benchmark + WS study with honest negative results, temporal head as the method contribution.

### E6 ablation: do the weak legacy labelers hurt the label model? (2026-10-23, `results/E6_ablation/`, dev n=150)
All 3-labeler subsets that contain the new 32B labeler (Q) vs all 4 (Q + L1 legacy Llama mq + L2 legacy Llama sq + L3 legacy Qwen-72B sq). Best PR-AUC per subset (Snorkel/DS with the 32B P(yes) band):
| Subset | Binary DS/Snorkel PR-AUC | Band DS / Snorkel PR-AUC |
|---|---|---|
| Q + L3 + L2 | 0.745 / 0.746 | 0.781 / 0.787 |
| Q + L3 + L1 | 0.692 / 0.692 | 0.788 / 0.787 |
| Q + L2 + L1 | 0.731 / 0.735 | 0.742 / 0.755 |
| all 4 | 0.765 / 0.765 | 0.789 / 0.791 |
Removing legacy labelers does not help; all 4 is as good or better. So the legacy labelers are not what limits WS. The 32B's own P(yes) already scores PR-AUC 0.801 / ROC-AUC 0.750 on dev, slightly above every aggregate (0.791 max): with these labelers the label model adds nothing over the best single labeler. WS needs more strong, different labelers: screening Qwen3-VL-30B-A3B-FP8 (`sq_v2`, `sq_v4`) next.

### Qwen3-VL-30B-A3B-FP8 screening (2026-10-23, `results/E3/screening_a3b.json`; dev n=150): not useful
| Prompt | Predicts positive | Acc | ROC-AUC | PR-AUC | Acc at dev-best threshold |
|---|---|---|---|---|---|
| sq_v2 | 93% | 0.527 | 0.656 | 0.711 | 0.667 |
| sq_v4 | 100% | 0.533 | 0.661 | 0.694 | 0.633 |
Yes-biased under both prompts and poor at ranking (ROC-AUC ~0.66, like the 8B), far below the dense Qwen3-VL-32B-FP8 `sq_v4` (ROC-AUC 0.750, PR-AUC 0.801). Error correlation between its two prompts is 0.85. **Not run on the pool.** Its errors are only weakly correlated with the 32B (0.12 / -0.01), but a weak labeler adds noise, not signal. Observed so far: among the open models tried (Qwen2.5-VL-7B, Qwen3-VL-8B, Qwen3-VL-30B-A3B, Qwen3-VL-32B), only the dense 32B is a good labeler here; small or low-active-parameter models say "yes" to almost everything. Next candidates must be dense and large: Qwen2.5-VL-72B (AWQ, 4 GPUs), InternVL3-78B, Mistral-Small-3.2-24B (1 x 80 GB or FP8); plus a second variant of the 32B (collage image mode, E2; a different prompt) for some diversity.

### E2: image presentation for Qwen3-VL-32B-FP8 (2026-10-23, `results/E3/screening_32b_collage.json`; dev n=150)
| Presentation, prompt | Predicts pos. | Acc | F1 | ROC-AUC | PR-AUC | Acc at dev-best thr. |
|---|---|---|---|---|---|---|
| **Collage (4x2 grid with year labels, 1200x600), sq_v4** | 33% | **0.713** | 0.667 | 0.786 | **0.815** | **0.747** |
| Collage, sq_v2 | 29% | 0.707 | 0.645 | 0.795 | 0.813 | 0.747 |
| Multi-image (8 images, NAIP 224 px, high-res 672 px), sq_v4 | 51% | 0.653 | 0.667 | 0.750 | 0.801 | 0.707 |
| Multi-image, sq_v2 | 35% | 0.660 | 0.617 | 0.739 | 0.798 | 0.680 |
- **Collage beats multi-image by ~6 points of accuracy and ~0.04 ROC-AUC on both prompts.** The legacy collage presentation was not a flaw for this model: the one-image grid with year labels works better than 8 separate images (precision 0.88-0.91, recall ~0.5: conservative). This is a finding for the paper (E2) and corrects my prior assumption that multi-image would win.
- Error correlation collage vs multi-image (sq_v4): 0.47; collage vs legacy Qwen-72B: -0.06. Two views of the same model are moderately different voters.
- Chosen on dev: **collage + sq_v4**. Pool run launched (4 shards x 2 GPUs). Both views of the 32B can then be labeling functions.

### E13: noise-robust and student-in-the-loop training, wave 1 (2026-10-25; `scripts/run_robust.py`, `scripts/analyze_e13.py`, `results/E13_robust_sil/`; dev n=150, mean +- std over 3 seeds)
Setup: the same frozen FlexiViT-S features and training recipe as E7; pseudo-labels = the 32B labeler's `P(yes)` (multi-image; 143 distinct values); confidence of a pseudo-label = |p - 0.5|; the **confident set is the 50% most confident of each pseudo-class** (class-balanced, ties kept: 9,090 of 18,179 locations); robust loss = generalized cross-entropy, q = 0.7; the "student" that labels uncertain samples is an exponential moving average (decay 0.995) of the head, sharpened with temperature 0.5, with its weight ramped in over 5 epochs. Code: `src/gully/student/robust.py` (47 tests pass). Jobs: 4 x (1 GPU, 4 CPU, 12 Gi) with the scheduling shape of the `smatousi-yieldsat-protocol-own3` jobs (A10 / RTX 3090 / RTX 4090 only, a 19-host exclusion list).

| Training | Head | Acc | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| All samples, KL (reference; reproduces the E7 student: 0.662 / 0.782) | temporal | 0.671 +- 0.017 | 0.626 | 0.732 | **0.772 +- 0.001** |
| V1: confident only + robust loss | temporal | 0.589 +- 0.010 | 0.635 | 0.638 | 0.658 +- 0.021 |
| V2: student labels the uncertain, after a 10-epoch warm-up | temporal | 0.596 +- 0.019 | 0.629 | 0.662 | 0.676 +- 0.015 |
| V3: student labels the uncertain, from epoch 0 | temporal | 0.591 +- 0.008 | 0.569 | 0.666 | 0.671 +- 0.027 |
| All samples, KL (reference) | MLP | 0.640 +- 0.027 | 0.633 | 0.729 | 0.726 +- 0.038 |
| V1 | MLP | 0.620 +- 0.035 | 0.673 | 0.652 | 0.632 +- 0.018 |
| V2 | MLP | 0.596 +- 0.034 | 0.584 | 0.663 | 0.627 +- 0.017 |
| V3 | MLP | 0.613 +- 0.024 | 0.677 | 0.634 | 0.612 +- 0.024 |

- **Negative result: none of V1, V2, V3 improves on plain distillation; all lose 0.05-0.11 PR-AUC** (several standard deviations), for both heads. The recipe is sound: the reference row reproduces the earlier E7 student.
- **V2 phase analysis:** the second phase (the student labels the uncertain samples) beat the best warm-up epoch in only 1 of 6 runs; the selected epoch was usually before the student targets began (mean best epoch ~8 against warm-up 10), so V2's score is mostly the V1-like warm-up model.
- **Hypotheses, untested:** (a) the VLM's confidence is not aligned with its correctness: P(yes) is near 0 or 1 for many locations, so the confident half is where its yes-bias is strongest, and the less-confident half carries the ranking information that dropping removes (and halves the training set); (b) the student's own targets on the uncertain half reinforce its errors; (c) GCE on near-saturated labels adds little.
- **Not yet run (wave 2, open):** the controls `all_gce` (robust loss only) and `drop_kl` (dropping only), which attribute the loss to dropping or to the loss; the same grid on the label-model pseudo-labels (`band_snorkel_legacy3_q32`); a sweep of the kept fraction (0.3 / 0.7 / 0.9). Also not tested: confidence defined by agreement between labelers instead of by one labeler's probability.
- **Cluster note:** all new pods were refused for ~30 minutes by the NRP utilization-flag admission webhook (`pod.nrp-nautilus.io` / `job.nrp-nautilus.io`) at the start; after it cleared, the first job was placed within a minute. Keep helper pods short-lived and right-sized (the flag is account-wide).

## E5: DEM terrain labelers (2026-10-25; `results/E5/dem_labelers.json`, `results/E6_dem/`)
- **Setup:** 35 terrain attributes per 1 m DEM tile (`data/dem.py`, 18,727/18,727 tiles ok; no absolute elevation); 21 labelers (`labelers/dem.py`), each one attribute with direction (sign of dev AUC), abstention band (coverage >= 50%) and score map (dev ECDF) fitted on dev only. Dev-split votes of these labelers are **5-fold cross-fitted** (`crossfit_dev`), because tuning a band on dev labels and scoring on the same labels is optimistic (unit test on pure noise: in-sample covered accuracy > 0.54, cross-fitted ~0.50).
- **Individually weak.** Out-of-fold dev AUC: acc_max 0.672, twi_max 0.629, aspect_circ_var 0.624, sink_depth_max 0.576, frac_concave 0.570, prof_curv_min 0.564; the other 15 are 0.41-0.56, i.e. 9 of 21 are *below* 0.5 out of fold (their in-sample direction was noise). With n=150 (AUC SE ~0.045) only the first 2-3 are distinguishable from chance, before correcting for picking among 21. The pool and labeled positive-vote rates differ a lot for several (e.g. tpi15_frac_hollow, slope_p90, twi_p90): labeled locations were chosen near mapped gullies, so terrain distributions shift between pool and labeled sets.
- **Label models (dev acc / PR-AUC), 4 VLM labelers + 21 DEM vs. reference 4 VLM (acc 0.68-0.69):** mv 0.580 / 0.803, DS 0.647 / 0.528, triplet 0.600 / 0.779, Snorkel 0.513 / 0.518; banded: 0.573, 0.647, 0.593, 0.540. DEM only: mv 0.573, DS 0.653, triplet 0.587, Snorkel 0.560; banded Snorkel 0.653; soft-logit stacker (CV on dev) 0.693 / 0.729.
- **Conclusion (negative):** adding the DEM labelers does not help and hurts every label model's accuracy; the 21 labelers are correlated and mostly noise, so the generative models (DS, Snorkel) overweight them. The high PR-AUC of mv/triplet is an artifact of near-all-positive posteriors (pool_pos 0.84-0.96) and must not be cited as improvement. The DEM-only stacker at 0.693 is the only hint of signal, but its inputs' directions/bands were fitted on the same 150 dev locations, so it is not a clean estimate; do not report it as a result without a separate check.
- **Not done:** an a-priori (hydrology-motivated, not dev-fitted) direction for a handful of attributes; a student trained on DEM-augmented pseudo-labels; DEM as an input channel to the student (a different experiment, not a labeler). Both E5 variants stay out of the paper's main tables; the paper may cite this as a negative result in the Discussion if space allows.
- **E5 follow-up, top-3 subset (2026-10-25; `results/E6_dem/e6_legacy3_q32_dem3.json`, `e6_dem3_only.json`):** the 3 DEM labelers with out-of-fold dev AUC > 0.6 (acc_max, twi_max, aspect_circ_var). **The subset was picked by looking at dev AUC, so every dev number below is optimistic** (selection among 21 on the same 150 locations; the cross-fitting does not cover this selection step).
  - 4 VLM + 3 DEM: binary mv/DS/triplet/Snorkel acc 0.680 / 0.667 / 0.660 / 0.673, PR-AUC 0.75-0.79; banded acc 0.680-0.687, PR-AUC 0.786-0.796. Reference (4 VLM only) 0.68-0.69: **no gain**, the DS/Snorkel collapse of the 21-LF set disappears.
  - 3 DEM only: binary acc 0.633-0.680, banded 0.693-0.727 (band_ds 0.727), PR-AUC 0.64-0.68 with pool_pos 0.79-0.91 (near-all-positive posteriors, so PR-AUC is not informative). Single-attribute terrain rules reach roughly the accuracy of the VLM labelers on dev, but this is the most selection-inflated number here.
  - Conclusion: still no evidence that DEM improves the combined label model. A fair test needs hydrology-chosen attributes fixed before looking at dev, or the 90-location tail that is not in dev (test must stay untouched).

## E3b: collage screening of more open VLMs (launched 2026-10-28; results in `results/E3/screening_collage_models.json` when done)
- **Protocol:** each model labels the 548 labeled locations (collage 4x2, 1200x600, prompt `sq_v4`, JSON output, P(yes) from logprobs), scored on dev (n=150) with `scripts/screen_labelers.py`; only models that are accurate or add different errors go to the 18k pool. Small GPUs only (24 GB cards, tensor parallel), per user decision; model weights cached on a new 1 Ti volume `gully-models` (`cluster/gully_models_pvc.yaml`; `HF_HOME` in the job template).
- **Submitted (10):** Qwen2.5-VL-72B-AWQ (TP4), InternVL3-78B-AWQ (TP4; lmdeploy-format AWQ, may not load in vLLM), InternVL3.5-38B (fp8 on load, TP4), Mistral-Small-3.2-24B (fp8 on load, TP2), Gemma-3-27B (FP8-dynamic, TP2), Qwen2.5-VL-32B-AWQ (TP2), GLM-4.5V-FP8 (TP8), MiniCPM-V-4.5 (TP1), Pixtral-12B (TP2), Kimi-VL-A3B (TP2).
- **Not run, with reasons:** Llama 4 (excluded by the user); Qwen3-VL-235B, Pixtral-Large, Llama-3.2-90B (do not fit small GPUs: 120-250 GB); NVLM-D-72B, Molmo-72B, LLaVA-OneVision-72B (no 4-bit checkpoint, 150-290 GB downloads, older/weaker); Qwen3-VL-32B-Thinking (long reasoning conflicts with the constrained JSON answer); Ovis2-34B (repo id not found); Phi-4-multimodal (vision LoRA path); remote-sensing VLMs (GeoChat, TEOChat, ...: not served by vLLM).
- **2026-10-28 status:** after ~2 h, 9 of the 10 collage jobs were still pending (24 GB cards in 2-, 4- and 8-GPU groups are scarce). MiniCPM-V-4.5 loaded, then ran out of memory while vLLM profiled 8 video items on a 24 GB card; fixed by limiting multimodal inputs to what the collage needs (`limit_mm_per_prompt` = 1 image for collage mode; `mm_limit: {image: 1, video: 0}` in its config). Retries pick the fix up from the PVC code copy.
- **2026-10-28, 4 h in:** still 0 of 10 scheduled (MiniCPM-V failed 3 times, the last two before the memory fix reached the PVC). Single-A100 variants (`*-1gpu_sq_collage.yaml`, for Mistral, Gemma, Qwen2.5-VL-32B, Pixtral, Kimi) were all still pending after 9 min and were cancelled, per the rule "A100 only if one card suffices and it is granted fast". The configs stay for later.
