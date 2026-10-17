#!/usr/bin/env bash
# Reproduces every number, table and figure of the paper from the files in data/.
# Run from the repository root:  bash reproduce.sh
# Runtime on a 2-core CPU: about 15–20 minutes (the bootstrap and simulation steps dominate).
set -euo pipefail
mkdir -p results/reanalysis results/simulation results/tables figures

echo "[1/6] Estimands, 10,000-replicate cluster bootstrap and paired model comparisons (Endoscapes, SAGES)"
python src/independent_v2.py data results/reanalysis

echo "[2/6] SAGES sensitivity: countries as the resampling unit"
python src/country_bootstrap.py data results/reanalysis/country_bootstrap_W_minus_E1.csv

echo "[3/6] Monte Carlo variability of bootstrap p-values (SAGES C2)"
python src/mc_pvalue_check.py data results/reanalysis/mc_pvalue_check_C2.csv

echo "[4/6] Simulation study"
python src/simulation.py results/simulation
python src/simulation_design.py data results/simulation 2000 1000

echo "[4b] Endoscapes sensitivity analyses (annotators, leave-one-operation-out, disagreement reference)"
python src/endoscapes_sensitivity.py data results/reanalysis
python src/time_baseline.py data results/reanalysis/time_baseline.csv
python src/reference_anchored.py data results/reanalysis 10000

echo "[5/6] Figures"
python src/make_image_figs.py data/figure_frames/endoscapes data/figure_frames/sages data/endoscapes/endoscapes_test_linkage.csv \
       data/sages/sages_labels.csv data/frozen/Tables/Pair_disagreement.csv figures
python src/make_feature_figs.py data/frozen figures
python src/make_fig1.py data results/reanalysis/reference_anchored.csv figures
python src/make_figures.py data/frozen figures results/reanalysis/endoscapes_annotator_sensitivity.csv
python src/make_sim_figure.py results/simulation figures results/tables
python src/make_graphical_abstract.py data/frozen data/sages/results figures data/endoscapes/Endoscapes_10000_paired_contrast_CI.csv

echo "[6/6] Results tables and Figs. 6-7"
python src/make_tables.py data/frozen results/tables
python src/make_t7.py results/reanalysis/reference_anchored.csv results/tables
python src/make_v3_results.py data results/reanalysis figures results/tables
echo "Done. Outputs are in results/ and figures/."
