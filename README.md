# Operation-aware estimand audit for frame-level AUROC in surgical video

Code and selected reference data for *Operation-aware estimand audit for frame-level AUROC in surgical video: comparison populations, mixed-label eligibility and clustered uncertainty* (submitted to Medical Image Analysis).

The audit in `src/oea_audit.py` separates pooled frame AUROC (E1), cross-operation AUROC (X), pair-weighted within-operation AUROC (W), and the equal-weight mean over mixed-label operations (E2). It also reports the within-operation pair fraction, mixed-label support, effective number of operations, and paired cluster-bootstrap intervals. `src/baseline_contrast.py` compares W with a within-operation time baseline.

## Audit your own predictions

Use Python 3.11 or later. The input CSV needs one row per frame and model, a unit identifier, a binary label, a score, and a model identifier.

```bash
python -m pip install -r requirements.txt
mkdir -p results
python src/oea_audit.py --csv predictions.csv --unit video_id --label y --score score --model model_id \
       --n-boot 10000 --seed 20261010 --out results/
```

The command writes `estimands_with_ci.csv` and `paired_model_comparisons.csv`.

## What is public here

| Directory | Contents |
|---|---|
| `src/` | Audit, bootstrap, sensitivity, simulation, table and figure scripts |
| `data/sages/` | SAGES 2024 test labels, checkpoint scores, metadata, archived results and provenance records |
| `data/figure_frames/sages/` | SAGES frames used in figures |
| `data/frozen/Tables/` | Aggregate reference tables |
| `data/endoscapes/` | Two aggregate 10,000-draw interval tables only |

Model weights are not distributed. The SAGES provenance files document the original inference run; their recorded hashes for redacted manifests refer to the original versions, not to the copies here.

This public checkout **does not** contain Endoscapes row-level labels or predictions, frame/video linkage, annotator ratings, excluded-keyframe records, per-video provenance or disagreement tables, or Endoscapes images. These files were present in the supplied archive but are withheld from this public repository. The original scripts remain so authorized users can run them with local input files under applicable terms.

`reproduce.sh` is the full-study workflow. It requires the withheld Endoscapes files in their documented paths; running it from a fresh public clone will stop when they are absent. The included aggregate tables are reference results, not a substitute for those row-level inputs. SAGES and simulation analyses can be examined separately using the corresponding scripts in `src/`.

## Sources and licences

* The Endoscapes authors' [project page](https://github.com/CAMMA-public/Endoscapes) describes CC BY-NC-SA 4.0, while the [PhysioNet distribution](https://physionet.org/content/endoscapes-2023/1.0.0/) has separate restricted-access terms. This repository does not resolve that difference or grant permission to redistribute the withheld files. Cite Mascagni et al., *Scientific Data* 12:331 (2025) when using the dataset.
* The included SAGES labels and figure frames come from the [SAGES Critical View of Safety Challenge 2024 dataset](https://huggingface.co/datasets/CAMMA-public/SAGES_CVS_Challenge_2024), revision `deef093e7fc6522a6c5e6bd3d3bab3408e7588d4`, under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). Cite Alapatt et al., arXiv:2509.17100 (2025).
* Model scores and original aggregate analysis outputs in `data/` are provided under CC BY-NC-SA 4.0, without changing source dataset terms.
* Original code in `src/` is under the MIT licence in `LICENSE`.

Please cite the paper when it is published.
