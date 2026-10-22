# 02 — Repository architecture and data contracts

## Layout

```
weakly-supervised-gully-detection/
├── spec/                      # this folder
├── configs/
│   ├── data.yaml              # data roots, HUC list, split seed, buffer distance
│   ├── labelers/*.yaml        # one file per labeler (model id + revision, prompt, image mode, sampling)
│   ├── prompts/               # single_question.yaml, multi_question.yaml (17 Qs + aggregator)
│   ├── label_models.yaml
│   └── student/*.yaml         # backbone, head, loss, optimizer, seeds
├── src/gully/
│   ├── data/
│   │   ├── index.py           # build/read the location index (below)
│   │   ├── splits.py          # spatially blocked dev/test split, pool exclusion buffer
│   │   └── images.py          # load a location's images by role, native res, normalization
│   ├── labelers/
│   │   ├── vlm.py             # vLLM runner: batching, multi-image or collage, JSON-schema output, logprob P(yes)
│   │   ├── multi_question.py  # 15-question + aggregator pipeline (ported)
│   │   ├── clip_rs.py         # RemoteCLIP / GeoRSCLIP zero-shot scores
│   │   ├── terrain.py         # DEM-derived labeler (stretch)
│   │   └── legacy.py          # ingest old JSON outputs as LFs
│   ├── ws/
│   │   ├── label_matrix.py    # assemble L (and confidence matrix) from labeler outputs
│   │   └── label_models.py    # Snorkel, MV, Dawid–Skene, FlyingSquid, EBCC, MeTaL, … behind one interface
│   ├── student/
│   │   ├── features.py        # precompute frozen-backbone features once per (backbone, image role)
│   │   ├── heads.py           # linear (legacy), temporal-attention head
│   │   └── train.py           # noise-aware training, dev-based model selection
│   └── eval/
│       ├── metrics.py         # acc, P, R, F1, NPV, ROC-AUC, PR-AUC; bootstrap CI; McNemar
│       └── report.py          # results/*.json → LaTeX tables
├── scripts/                   # thin CLIs: build_index, make_splits, run_labeler, build_L, fit_label_model,
│                              # extract_features, train_student, evaluate, make_tables
├── results/                   # small, committed: metrics JSON, tables, figures
└── artifacts/                 # large, git-ignored: features, checkpoints, raw labeler outputs
```

Tooling: Python ≥3.10, `uv` or a conda env pinned in `pyproject.toml`; `vllm`, `timm`, `torch`, `snorkel`, `scikit-learn`, `rasterio`, `pandas/pyarrow`. wandb is optional (`--wandb`).

## Data contracts

### Location index — `artifacts/index/locations.parquet`
| column | type | note |
|---|---|---|
| `loc_id` | str | `U-xxxxxx` (unlabeled pool) or `L-xxxxxx` (expert-labeled) |
| `huc12` | str | |
| `x`, `y`, `crs` | float, float, str | tile centroid |
| `img_naip_0` … `img_naip_6` | str | paths; year bin in `configs/data.yaml` |
| `img_highres` | str | MSDIS 15 cm |
| `split` | str | `pool`, `pool_excluded`, `dev`, `test` |
| `legacy_id` | str | old integer key, for ingesting old outputs |
| `tile_width_m`, `naip_px`, `naip_res_m`, `highres_px`, `highres_res_m`, `highres_bands` | num | header facts used for QC (see 01-code-audit §6) |
| `dem` | str | DEM tile filename (pool only) |
| `pvc`, `img_dir`, `dem_dir` | str | where the files live; `img_*` columns are filenames relative to `img_dir` |

### Ground truth — `artifacts/gt/gt.parquet`
`loc_id`, `annotator`, `image_role`, `score (0–4)`, plus derived `y_strict_neg`, `y_lenient_neg`, `y_lenient_pos`, `y_strict_pos` (majority over annotators, null when undefined), `y_main` = `y_strict_neg`.

### Labeler output — `artifacts/labels/<labeler>.jsonl` (one line per location)
```json
{"loc_id": "U-000123", "labeler": "qwen2.5-vl-72b_sq_multiimg", "model": "Qwen/Qwen2.5-VL-72B-Instruct",
 "revision": "<hf sha>", "prompt_id": "sq_v1", "image_mode": "multi_image|collage",
 "raw": "...", "label": 1, "p_yes": 0.83, "abstain": false, "latency_s": 1.9, "ts": "..."}
```
`label ∈ {0,1,-1}` (`-1` = abstain/unparseable). `p_yes` comes from token logprobs when available, otherwise null.

### Label matrix — `artifacts/ws/L_{pool,dev,test}.parquet`
Rows = `loc_id`, columns = labelers; values in {-1,0,1}. Plus `P_*.parquet` holding `p_yes`.

### Pseudo-labels — `artifacts/ws/pseudo_<label_model>_<lfset>.parquet`
`loc_id`, `p1`, `y_hat`.

## Porting map (old → new)

| Old | New | Notes |
|---|---|---|
| `Data_Preparation/*`, `Data_Downloading/*` | **not ported** | Tiles already exist; only `build_index.py` reads them. |
| `large_dataset/ollama_query.py`, `pytorch_query*.py` | `labelers/vlm.py` | Rewrite on vLLM; keep the prompts verbatim; keep the collage function as an `image_mode`. |
| `pytorch_query_multi_question.py` | `labelers/multi_question.py` | Port the logic; aggregator through the same vLLM server. |
| `large_dataset/results/*.json` | `labelers/legacy.py` | Ingest as LFs `legacy_*`, so E0 reproduces the paper. |
| `WS_on_Gully.ipynb` | `ws/*` + `scripts/fit_label_model.py` | Fix B4. |
| `labeling_tool/v2_results/final/aggregate_labels.py` | `scripts/build_gt.py` | Port all 4 schemes + majority; add Fleiss' κ. |
| `gully_detection/timm_flexiViT.py` | `student/features.py` + `heads.py:LinearHead` | Fix B1/B7/B8; features precomputed. |
| `gully_detection/train_rgb_8_images_WS_accel.py` | `student/train.py` | Fix B2/B3/B5/B6. |
| `gully_detection/train_eight_rgb_accel_kfold_GTs.py` | `student/train.py --supervised` | The "SM + Reece" and supervised upper-bound baselines. |
| `dem2so/` | `labelers/terrain.py` (stretch) | Reuse the DEM tiles only. |

## Key design choice: frozen features, precomputed

The old student already freezes FlexiViT, so we extract features **once** per (backbone, image role, location) into `artifacts/features/<backbone>/<role>.npy` (fp16). For example, 18.7k locations × 8 images × 65 tokens × 384 dims × 2 bytes ≈ 7.5 GB per backbone; keep the CLS + mean-pooled tokens too (≈ 115 MB). Then every student run (seeds, label sets, heads, ablations) takes minutes on one A10. **That is what makes the experiment matrix fit in 3 weeks.** Fine-tuning the backbone is out of scope unless the gates leave time.
