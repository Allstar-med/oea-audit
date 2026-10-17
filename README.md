# Operation-aware estimand audit for frame-level AUROC in surgical video

Code and selected derived results for the paper *Operation-aware estimand audit for frame-level AUROC in surgical video: comparison populations, mixed-label eligibility and clustered uncertainty* (submitted to Medical Image Analysis). This public repository excludes Endoscapes frame-level files and surgical images; see Data access below.

In a surgical-video test set made of many recordings, a pooled frame-level AUROC mostly compares frames from different operations. The audit in `src/oea_audit.py` reports, for any frame-level classifier, the pooled AUROC (E1), the cross-unit AUROC (X), the pair-weighted within-unit AUROC (W), the equal-weight mean over mixed-label units (E2), the within-unit pair fraction (alpha), the number of mixed-label units, the Kish effective number of units behind W, and cluster-bootstrap intervals with matched resamples for paired model comparisons. `src/reference_anchored.py` adds the difference in W between a model and a simple within-unit reference score (here the time of a frame within its recording), with matched unit-level intervals and a t interval for W that uses n_eff - 1 degrees of freedom.

## Quick start: audit your own predictions

```bash
pip install -r requirements.txt
mkdir -p results
python src/oea_audit.py --csv predictions.csv --unit video_id --label y --score score --model model_id \
       --n-boot 10000 --seed 20261010 --out results/
```

The input file needs one row per frame and model, with a unit identifier (operation, video or clip), a binary label and a score. The script writes `estimands_with_ci.csv` and `paired_model_comparisons.csv` (Benjamini–Hochberg adjusted within each estimand).

## Reproducing the paper

```bash
pip install -r requirements.txt
bash reproduce.sh
```

`reproduce.sh` is the full-study workflow. It requires the Endoscapes files listed below, which are **not included** in this public repository. Do not run it expecting the public checkout alone to regenerate every table and figure. With authorized local copies placed in the expected paths, it recomputes the estimands, intervals, sensitivity analyses, simulation study and figures. Bootstrap intervals and p-values carry Monte Carlo error; fixed seeds make runs repeatable on the same library versions.

| Directory | Contents |
|---|---|
| `src/` | audit library (`oea_audit.py`), analysis, simulation and figure scripts |
| `data/endoscapes/` | aggregate 10,000-replicate interval tables only; row-level Endoscapes data are withheld |
| `data/sages/` | SAGES 2024 test labels (three rating columns per frame), scores of the three retrained checkpoints, video metadata, model manifest (training settings and the SHA-256 hashes recorded before inference) and archived results |
| `data/frozen/` | frozen aggregate tables; per-video provenance and disagreement tables are withheld |
| `data/figure_frames/sages/` | SAGES frames used in the figures; Endoscapes frames are withheld |

`data/sages/provenance/` holds the dataset audit, inference log and automated validation report of the SAGES inference run. Local file paths were removed from `data/sages/model_manifest.json` before release; the SHA-256 recorded for the manifest in the inference log refers to the original file. Model weights are not distributed. The audit needs row-level labels, scores and unit identifiers; only the SAGES rows are public here.

## Data access

For full reproduction, authorized users must supply the three `data/endoscapes/locked_test_predictions_seed*.csv` files, `endoscapes_test_linkage.csv`, `endoscapes_test_annotator_ratings.csv`, `excluded_keyframes.csv`, the Endoscapes figure frames, and the withheld per-video figure/provenance tables. These are not available from this repository. The aggregate Endoscapes results are included as reference outputs, but cannot be independently regenerated from this checkout alone.

The Endoscapes authors' [project page](https://github.com/CAMMA-public/Endoscapes) describes CC BY-NC-SA terms, while the [PhysioNet distribution](https://physionet.org/content/endoscapes-2023/1.0.0/) identifies the dataset as restricted-access under its own data-use agreement. This release does not resolve that difference or grant redistribution rights. In the absence of confirmed permission for these specific row-level files and frames, they are omitted.

## Data sources and licences

* Endoscapes (Mascagni et al., 2025, *Scientific Data* 12:331): cite the dataset; access and use the underlying files under the terms of the source from which they were obtained. No row-level data or frames are redistributed here.
* SAGES Critical View of Safety Challenge 2024 (Alapatt et al., 2025, arXiv:2509.17100): [Hugging Face dataset](https://huggingface.co/datasets/CAMMA-public/SAGES_CVS_Challenge_2024), revision `deef093e7fc6522a6c5e6bd3d3bab3408e7588d4`, [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The included SAGES labels and figure frames retain that licence and attribution.
* Model scores and original aggregate analysis outputs in `data/`: CC BY-NC-SA 4.0; this does not change the source dataset terms.
* Code in `src/`: MIT licence (see `LICENSE`).

Non-commercial use only applies to the data. Please cite the dataset papers when using them.

## Citation

If you use this code, please cite the paper (citation to be added on publication).
