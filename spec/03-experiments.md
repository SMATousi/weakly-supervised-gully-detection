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
| E6 | P0 | Label models with the new LF set | MV, Dawid–Skene, Snorkel, FlyingSquid, EBCC, MeTaL, (NPLM, LELA if they port quickly); binary vs confidence-valued LFs | Table WS-frameworks |
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
