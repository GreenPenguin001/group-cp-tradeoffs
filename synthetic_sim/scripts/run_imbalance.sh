#!/usr/bin/env bash
set -euo pipefail

python -m synthcp.minority_imbalance_diagnosis_with_freq_larger_fonts \
  --outdir outputs_mc_imbalance \
  --alpha 0.1 \
  --imbalance-seeds 40 \
  --imbalance-n-cal-total 400 \
  --imbalance-n-test 4000 \
  --imbalance-weights 0.60,0.25,0.10,0.05 \
  --ratio-seeds 40 \
  --ratio-n-test 800 \
  --ratio-n-cals 12,25,50,100,200,400 \
  --num-table-points 5
