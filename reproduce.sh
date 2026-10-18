#!/usr/bin/env bash
# Recompute all results, tables and figures from data/. Run from the repository root.
# Takes about 15 minutes on a laptop CPU.
set -euo pipefail
mkdir -p results/reanalysis results/simulation results/tables figures

echo "estimands and bootstrap"
python src/estimands_bootstrap.py data results/reanalysis
python src/country_bootstrap.py data results/reanalysis/country_bootstrap_W_minus_E1.csv
python src/pvalue_streams.py data results/reanalysis/mc_pvalue_check_C2.csv

echo "Endoscapes sensitivity analyses and time baseline"
python src/endoscapes_sensitivity.py data results/reanalysis
python src/baseline_estimands.py data results/reanalysis/time_baseline.csv
python src/baseline_contrast.py data results/reanalysis 10000

echo "simulations"
python src/simulation.py results/simulation
python src/simulation_designs.py data results/simulation 2000 1000

echo "tables"
python src/tables.py data/frozen results/tables
python src/table7.py results/reanalysis/baseline_contrast.csv results/tables

echo "figures"
python src/fig3_fig9.py data/figure_frames/endoscapes data/figure_frames/sages data/endoscapes/endoscapes_test_linkage.csv \
    data/sages/sages_labels.csv data/frozen/Tables/Pair_disagreement.csv figures
python src/fig1_overview.py data results/reanalysis/baseline_contrast.csv figures
python src/fig2_framework.py figures
python src/fig4_simulation.py results/simulation figures results/tables
python src/fig5_fig8.py data/frozen figures results/reanalysis/endoscapes_annotator_sensitivity.csv
python src/fig6_fig7_tables.py data results/reanalysis figures results/tables
python src/graphical_abstract.py data/frozen data/sages/results figures data/endoscapes/Endoscapes_10000_paired_contrast_CI.csv
echo "done"
