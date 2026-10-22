# 01 — Audit of the existing code

Source: `/root/Gully_Detection` (HEAD `511740b`). Audited 2026-10-22, without access to the image data (`/root/home/pvc/...` isn't on this machine), so items marked **VERIFY** need a check on the data host.

## 1. What the current pipeline actually does

| Stage | Code | Behaviour |
|---|---|---|
| Tiles | `Data_Downloading/`, `Data_Preparation/` | Per location: `rgb_{0..6}_tile_{id}.tif` (NAIP, 2-year bins 2010–2024, 1 m, 128×128 px) + `rgb_highres_tile_{id}.tif` (MSDIS 2023, ~15 cm, ~853 px). |
| VLM labeling | `weak-supervision/large_dataset/{ollama,pytorch}_query*.py` | Ollama, one request at a time. **The 8 images are pasted into a 4×2 collage at 300×300 px each** (`collage_images`), so the 15 cm image is downsampled to ~43 cm/px and the 1 m images are upsampled. Free-text answer, mapped to a label via string/CLIP-embedding matching. |
| Multi-question | `pytorch_query_multi_question.py` | 15 yes/no questions to Llama-3.2-Vision-90B; an LLM (Llama 3.2 or Llama 4 Scout) aggregates them into a label. |
| Label matrix | `large_dataset/results/WS_on_Gully.ipynb` | 3 LFs: Llama3.2-90B multi-question, Llama3.2-90B single, Qwen2.5-VL-72B single. Qwen2.5-VL-7B and Llama 4 Scout were tried and dropped. |
| Label model | same notebook | `snorkel.LabelModel.fit(L_train, Y_dev, n_epochs=100, seed=12)`, where `Y_dev` = **test-set GT**. |
| Pseudo-labels | `train_lable_model_results.json` | 18,179 locations, `proba` + `label`. |
| Student | `gully_detection/train_rgb_8_images_WS_accel.py`, `timm_flexiViT.py`, `dataset.py:EightImageDataset_WS*` | Two **frozen** `flexivit_small.1200ep_in1k` copies (128 px/patch 16 and 764 px/patch 95, each giving an 8×8 grid). All tokens are flattened: 8 × 65 × 384 = 199,680 dims → LayerNorm → BatchNorm → Dropout → **one Linear layer** (not an MLP). Loss: KL to the LM probabilities. |
| GT | `labeling_tool/v2_results/final/aggregate_labels.py`, `agg/*_majority.json` | 3 annotators, 5-level per-image scores; per-annotator rule (strict/lenient pos/neg), then **majority across annotators**. `neg_strict_labels_majority.json` has 511 keys, 484 non-null (257 pos / 227 neg). |

## 2. Bugs and issues that affect reported numbers

Severity: **S1** = may change headline results, **S2** = affects validity or reviewer trust, **S3** = hygiene.

| ID | Sev | Where | Issue | Fix |
|---|---|---|---|---|
| B1 | S1 | `dataset.py:EightImageDataset_WS*.__getitem__` + `timm_flexiViT.py:forward` | The high-res branch takes **index 1** of `sorted(tile_files)`. Sorted order is `rgb_0, rgb_1, …, rgb_6, rgb_highres`, so index 1 is a 1 m NAIP image upsampled to 764 px, and the real 15 cm image (index 7) is shrunk to 128 px. **VERIFIED 2026-10-22 (filenames):** pool has `rgb_0..6` + `rgb_highres`, so sorted index 1 = `rgb_1` (NAIP). Test has `rgb_0..6` + `rgb_11` (high-res); string-sorted order is `0,1,11,2,…`, so index 1 is again `rgb_1` and the real high-res is index 2. The high-res branch never saw the 15 cm image in either split. | Select images by explicit role (`highres`, `naip_2010`, …) from the location index, never by sort position. |
| B2 | S1 | `train_rgb_8_images_WS_accel.py` training loop | `backward → step → zero_grad → step`. The second `step()` with zero grads still moves Adam's parameters (momentum), so each iteration makes two updates. | One `zero_grad/backward/step`. |
| B3 | S1 | same | Test metrics are computed **every epoch**, with no rule for which epoch is reported. If 0.70 was the best epoch on test, that's selection on the test set. **VERIFY** in the wandb project how the paper number was picked. | Select on dev; evaluate test once. |
| B4 | S1 | `WS_on_Gully.ipynb` | `LabelModel.fit(L_train, Y_dev=test GT)`. Snorkel uses `Y_dev` to set the class balance, so test labels leak into the pseudo-labels and into every student trained on them. | Fit with `class_balance` estimated from **dev**, or learned without labels; report both. |
| B5 | S2 | test `DataLoader(..., drop_last=True)` + `accelerator.gather` | Test samples are dropped (484 mod batch). Under multi-GPU, the distributed sampler pads with duplicates, so metrics are computed on the wrong n. | Single-process eval on the exact set, with an asserted n. |
| B6 | S2 | `train_rgb_8_images_WS_accel.py` | The validation split is 10% of the *pseudo-labeled* pool, scored against the LM argmax. It measures agreement with the teacher, not accuracy. | Keep it for monitoring only. Model selection uses the expert-labeled dev split. |
| B7 | S2 | transforms | `ToTensor()` only. No model-specific normalization (timm `data_config`). | Use `timm.data.resolve_data_config`. |
| B8 | S2 | dataset | The 15 cm image is resized to 764 px. The paper says FlexiViT avoids resizing. | Either use the native size with a matching patch size, or describe the resize honestly. |
| B9 | S2 | LF data | The multi-question LF is missing for 60/484 test locations (coverage 0.876); train coverage is unknown. | Recompute; treat missing as abstain explicitly and report coverage. |
| B10 | S2 | LM output | With 3 binary LFs, the label model produces only **8 distinct soft labels**, so the "noise-aware" loss is close to hard labels. | Confidence-valued LFs (VLM P(yes)) plus more LFs. |
| B11 | S2 | data | **Large prior shift** between test and unlabeled pool. Llama3.2-90B single predicts positive on 30% of test but **2%** of the unlabeled pool; Qwen2.5-VL-72B 70% vs 52%. The test set is balanced by construction (sampled around Reece et al. polygons); the pool isn't. | Report prevalence; class prior from dev, not test; discuss in paper; optional small random-pool annotation (see roadmap decision D-4). |
| B12 | S2 | paper vs code | The paper says GT voting requires "all the labelers"; the code uses a **majority of 3**. The paper says "MLP"; the code uses one linear layer. The paper never mentions the collage. | Paper text follows code (04-paper-revisions). |
| B13 | S2 | ids | Train and test locations are both keyed by bare integers from different directories (`'516'` exists in both namespaces). | Prefixed ids: `U-000123` / `L-000516`. |
| B14 | S3 | everywhere | Hard-coded absolute paths, mandatory wandb, partial seeding (`random.seed` commented out), notebooks as the source of truth. | Configs + scripts + full seeding. |
| B15 | S3 | labeling | Ollama is sequential, and its Llama 3.2 Vision effectively takes one image (the collage hid this). | vLLM with batching; multi-image or collage chosen explicitly (E2). |

## 3. Facts recomputed from the files (test GT = `neg_strict_labels_majority`, n=484)

| Labeler | Test acc | Test pos-rate | Pool pos-rate | Keep? |
|---|---|---|---|---|
| Qwen2.5-VL-72B single | 0.653 | 0.70 | 0.52 | yes |
| Llama3.2-V-90B single | 0.607 | 0.30 | 0.02 | yes |
| Llama3.2-V-90B multi-question | 0.568 (cov 0.876) | — | — | yes |
| Qwen2.5-VL-7B single | 0.490 | 0.03 | — | no (degenerate) |
| Llama 4 Scout single | 0.531 | 1.00 | — | no (degenerate) |
| Snorkel LM (3 LFs) | 0.643, F1 0.693, PR-AUC 0.762 | — | LM train labels: 6,047 pos / 12,132 neg | — |

## 4. Reusable assets

- `dem2so/`, `Data_Preparation/rgb_dem_*`: **a DEM tiling pipeline already exists** (DEM and stream-order tiles per HUC). This makes the terrain labeler (E5) much cheaper than assumed. **VERIFY** whether the DEM covers the 3 unlabeled HUC12s.
- `labeling_tool/`: annotation GUI and aggregation. Reuse it for any extra annotation (decision D-4).
- `weak-supervision/large_dataset/results/*.json`: existing LF outputs for 18,179 + 548 locations. Ingest them as-is, so the old Table 2 can be reproduced (E0).
- The 15 multi-question prompts (inside the multi-question JSON keys) and the single-question prompt (`ollama_query.py --prompt`): copy them verbatim into `configs/prompts/`.

## 5. Ground-truth schemes (recovered 2026-10-22; implemented in `src/gully/data/gt.py`)

The old repo has no script that produces `agg/*_labels*.json`; the per-annotator rules were recovered by matching the saved per-annotator labels (100% agreement, tested in `tests/test_gt.py`). Scores per image are 0-4, where 2 is unsure; a location whose 8 scores are all 2 is null for that annotator. Raw scores are in `data/gt_raw/annotator_{1,2,3}.json` (547 locations), legacy outputs in `data/gt_legacy/`.

| Scheme (new name) | Legacy file | Per-annotator positive iff | Fleiss kappa | Pos / Neg (majority) |
|---|---|---|---|---|
| `strict_neg` (**main**) | `neg_strict` | any non-2 score > 0 (negative only if all non-2 scores are 0) | 0.582 | 257 / 227 |
| `lenient_neg` | `neg_len_1or0` (identical to `pos_len_4or3`) | any score >= 3 | 0.646 | 231 / 256 |
| `lenient_pos` | `pos_len_4-1` | some score 4 and >= 2 scores >= 1 | 0.552 | 107 / 381 |
| `strict_pos` | `pos_strict` | some score 4 and >= 2 scores >= 3 | 0.557 | 103 / 385 |

Majority: over annotators with a non-null vote; null if fewer than 2 votes or a tie. Of the 547 scored locations, 484 are non-null under the main scheme. Note `strict_neg` is the most permissive for "positive" and `strict_pos` the most conservative: the scheme names are legacy, so state the rule, not the name, in the paper.

## 6. Data facts from the index (`scripts/build_index.py`, 2026-10-22)

- Pool: 18,179 locations x 8 images, all with a DEM tile (so the terrain labeler E5 is feasible). Test dir: 548 locations x 8 (`rgb_0..6`, `rgb_11`); location `547` has no annotator scores, so 547 are scored, 484 non-null under the main GT.
- All tiles are EPSG:6344 (UTM 15N, metres), so split distances can be computed directly.
- NAIP: 128 x 128 px at 0.91-0.96 m/px, not exactly 1 m. Tile width is 108-122 m (pool 115-117 m), so the 256 m exclusion buffer is about 2.1-2.4 tile widths. Keep 256 m.
- High-res (15 cm) native size varies: pool 764 px (9,063) or 780 px (9,025); test 720 / 782 / 817 px. The audit's "~853 px" was wrong. A fixed-size student input therefore needs a resize (B8 stays open), done explicitly and recorded.
- High-res band count: pool always 4 (RGBA), test 3 (325) or 4 (223). **Drop the alpha band** when loading.
- **59 pool locations have a truncated high-res image** (<500 px wide, as small as 15 px; tile at the edge of MSDIS coverage). Flag them (`highres_px`) and decide at feature extraction: mask the image or exclude those locations from the high-res experiments. The expert-labeled set has none.
