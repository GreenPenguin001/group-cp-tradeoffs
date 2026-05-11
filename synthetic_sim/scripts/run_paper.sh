#!/usr/bin/env bash
set -euo pipefail

python -m synthcp.cli --experiment paper --outdir outputs_mc_paper --alpha 0.1 --seeds 40 --n_cal 50 --n_test 500
