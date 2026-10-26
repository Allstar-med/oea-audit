# Operation-aware estimand audit for frame-level AUROC in surgical video

Code and selected reference data for the paper *Frame-level AUROC in surgical video mostly compares frames from different operations*.

In a surgical-video test set made of many recordings, a pooled frame-level AUROC mostly compares frames from different operations. The audit in `src/oea_audit.py` reports, for any frame-level classifier, the pooled AUROC (E1), the cross-unit AUROC (X), the pair-weighted within-unit AUROC (W), the equal-weight mean over mixed-label units (E2), the within-unit pair fraction (alpha), the number of mixed-label units, the Kish effective number of units behind W, and cluster-bootstrap intervals with matched resamples for paired model comparisons. `src/baseline_contrast.py` adds the difference in W between a model and a simple within-unit baseline score (here the time of a frame within its recording), with matched unit-level intervals and a t interval that uses n_eff - 1 degrees of freedom.

## Quick start: audit your own predictions

```bash
pip install -r requirements.txt
python src/oea_audit.py --csv predictions.csv --unit video_id --label y --score score --model model_id \
       --n-boot 10000 --seed 20261010 --out results/
```

The input file needs one row per frame and model, with a unit identifier (operation, video or clip), a binary label and a score. The script writes `estimands_with_ci.csv` and `paired_model_comparisons.csv` (Benjamini–Hochberg adjusted within each estimand).

## Reproducing the paper

The full workflow requires Endoscapes inputs supplied locally under their applicable terms. This public checkout omits Endoscapes frame-level labels and predictions, frame/video linkage, individual annotator ratings, excluded-keyframe records, operation-level disagreement/provenance tables, and images. A fresh public clone cannot run the full workflow without those inputs.

The required local paths are `data/endoscapes/locked_test_predictions_seed{17,29,43}.csv`, `data/endoscapes/endoscapes_test_linkage.csv`, `data/endoscapes/endoscapes_test_annotator_ratings.csv`, `data/frozen/Tables/Pair_disagreement.csv`, and the figure images in `data/figure_frames/endoscapes/`.

After supplying these files, run:

```bash
pip install -r requirements.txt
bash reproduce.sh
```

`reproduce.sh` runs the analysis from `data/`: estimands, bootstrap intervals and model comparisons, sensitivity analyses, time-baseline contrasts, simulations, tables and figures. Runtime depends on the machine. Random seeds are fixed; use the same package versions when comparing repeated runs.

The estimates and intervals of the primary analysis, and the tables and figures that show them (`tables.py`, `fig1_overview.py`, `fig5_fig8.py`, `fig6_fig7_tables.py`, `graphical_abstract.py`), are read from the archived results of the original analysis in `data/frozen/`, `data/sages/results/` and `data/endoscapes/`. `src/estimands_bootstrap.py` recomputes them with a separately written implementation and different random streams and writes them to `results/reanalysis/`. In the local verification, all 18 primary point estimates agreed to within 7e-16, and the largest absolute difference between the checked interval endpoints was 0.00433. Different bootstrap streams give different interval endpoints. Everything else, including Table 7 and the simulations, is computed directly by the scripts here.

Figures are drawn with matplotlib and request the Liberation Sans font. If it is unavailable, matplotlib uses a fallback font, which can change the layout.

| Directory | Contents |
|---|---|
| `src/` | audit library (`oea_audit.py`), analysis, simulation and figure scripts |
| `data/endoscapes/` | Two aggregate 10,000-draw bootstrap interval tables |
| `data/sages/` | SAGES 2024 test labels (three rating columns per frame), scores of the three retrained checkpoints, video metadata, model manifest (training settings and the SHA-256 hashes recorded before inference) |
| `data/sages/results/` | archived results of the original SAGES analysis |
| `data/sages/provenance/` | dataset audit, inference log and validation report of the SAGES inference run (records only; not read by the scripts) |
| `data/frozen/` | Archived aggregate tables of the original analysis |
| `data/figure_frames/sages/` | SAGES frames used in the figures |

The audit command above can be used with your own predictions. The SAGES country-bootstrap analysis can also run from the public files alone:

```bash
mkdir -p results/reanalysis
python src/country_bootstrap.py data results/reanalysis/country_bootstrap_W_minus_E1.csv
```

Model weights are not distributed. Local file paths and internal notes were removed from `data/sages/model_manifest.json` and `data/sages/provenance/dataset_audit.json`, so the SHA-256 values recorded for these two files in `validation.json` and in the inference log refer to the original versions. The recorded hashes of `sages_labels.csv`, `sages_predictions.csv` and `inference_log.json` match the files here. `execution.log` and `video_manifest.csv`, also listed in `validation.json`, are not included, and the `requirements.txt` listed there is that of the inference environment. The checks in `validation.json` refer to the original inference bundle, which contained no images; the SAGES frames in `data/figure_frames/sages/` were added for the figures.

Some archived files are kept for reference but are not read by `reproduce.sh`: `data/sages/results/{aggregate_results,operation_support_primary,ranking_agreement,oea_py_crosscheck}.csv` and `data/frozen/Tables/27_model_comparisons.csv`.

## Data sources and licences

* Endoscapes (Mascagni et al., 2025, *Scientific Data* 12:331): the authors' [CAMMA project page](https://github.com/CAMMA-public/Endoscapes) lists CC BY-NC-SA 4.0 for its public release. The [PhysioNet distribution](https://physionet.org/content/endoscapes-2023/1.0.0/) has separate restricted-access terms. This repository contains aggregate analysis outputs only; it does not grant permission to redistribute the omitted source data.
* SAGES Critical View of Safety Challenge 2024 (Alapatt et al., 2025, arXiv:2509.17100): [Hugging Face dataset](https://huggingface.co/datasets/CAMMA-public/SAGES_CVS_Challenge_2024), revision `deef093e7fc6522a6c5e6bd3d3bab3408e7588d4`, [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). The included SAGES labels and figure frames retain that licence and attribution.
* Model scores and original aggregate analysis outputs in `data/`: CC BY-NC-SA 4.0; this does not change the source dataset terms.
* Code in `src/`: MIT licence (see `LICENSE`).

Use of the data is limited to non-commercial purposes. Please cite the dataset papers when using them.

## Citation

If you use this code, please cite the paper (citation to be added on publication).
