# Fundamental/Structural Trade-offs Between Equalized Coverage and Equalized Set Size in Group-Conditional Conformal Prediction

This repository keeps only the code and the minimum instructions needed to rerun the experiments. Generated outputs, caches, downloaded datasets, and raw FACET files are not included.

## Setup

Tested with Python `3.10.15`.

```bash
conda env create -f environment.yml -n group_cp_tradeoffs
conda activate group_cp_tradeoffs
pip install -e ./synthetic_sim
```

Unless a step starts with `cd`, run the commands below from the repository root.

The real-data experiments are notebook-based. After installing dependencies, launch them with Jupyter Notebook from the experiment folder so their local relative paths resolve correctly.

When you run a notebook from its own experiment folder, it will create local `outputs/` and `cache/` subfolders automatically. Those generated folders are intentionally not tracked in this clean package.

The real-data notebooks are already set to the main experiment path: `RUN_MODE="train_and_analyze"` and `EXPERIMENT_MODE="full"` unless a notebook explicitly says otherwise.

The real-data notebooks use CUDA automatically when it is available and otherwise fall back to CPU. The same code path still runs on CPU, but paper-scale runs will be substantially slower there.

## Data Placement

- `synthetic_sim`: no external data required.
- `multiNLI`: the notebooks download `nyu-mll/multi_nli` from Hugging Face into the local `cache/` directory they create at runtime.
- `bio_bias`: the notebooks download `LabHC/bias_in_bios` from Hugging Face into the local `cache/` directory they create at runtime.
- `FACET`: place `annotations.zip`, `imgs_1/`, `imgs_2/`, and `imgs_3/` inside `FACET/`. The notebook also accepts the same files under `FACET/data/facet/`.

## Reproducing Main Paper Outputs

### `synthetic_sim`

Run the main synthetic paper bundle:

```bash
python -m synthcp.cli --experiment paper --outdir synthetic_sim/outputs_mc_paper --alpha 0.1 --seeds 40 --n_cal 50 --n_test 500
```

Main paper figures are written under `synthetic_sim/outputs_mc_paper/paper/`.

Optional imbalance diagnostics:

```bash
python -m synthcp.minority_imbalance_diagnosis_with_freq_larger_fonts --outdir synthetic_sim/outputs_mc_imbalance --alpha 0.1 --imbalance-seeds 40 --imbalance-n-cal-total 400 --imbalance-n-test 4000 --imbalance-weights 0.60,0.25,0.10,0.05 --ratio-seeds 40 --ratio-n-test 800 --ratio-n-cals 12,25,50,100,200,400 --num-table-points 5
```

### `multiNLI`

Run one score notebook from its own folder, for example:

```bash
cd multiNLI/simple
jupyter notebook multi_nli.ipynb
```

Open the notebook and run all cells. It prints a run folder as `RUN_ROOT`, usually `multiNLI/<score>/outputs/<RUN_NAME>/`.

Then make the main figure/table files from that run folder:

```bash
cd multiNLI/simple
python outputs/data_plot.py outputs/<RUN_NAME>
```

The main files are:

- `multiNLI/<score>/outputs/<RUN_NAME>/paper_ready/main_text/figures/figure_primary_overview.{png,pdf}`
- `multiNLI/<score>/outputs/<RUN_NAME>/paper_ready/main_text/tables/table_primary_mechanism_summary.{csv,tex}`
- `multiNLI/<score>/outputs/<RUN_NAME>/paper_ready/main_text/tables/table_primary_section3_support.{csv,tex}`

Appendix files and a full file list are recorded in `paper_ready/manifest.json`.

Repeat the same workflow for:

- `multiNLI/saps/multi_nli_saps.ipynb`
- `multiNLI/raps/multi_nli_raps.ipynb`

### `bio_bias`

Run one score notebook from its own folder, for example:

```bash
cd bio_bias/simple
jupyter notebook simple.ipynb
```

Open the notebook and run all cells. It prints a run folder as `RUN_ROOT`, usually `bio_bias/<score>/outputs/<RUN_NAME>/`.

Then redraw the main paper figure set from that run folder:

```bash
cd bio_bias/simple
python outputs/data_plot.py --result-root outputs/<RUN_NAME> --outdir outputs/<RUN_NAME>/paper_redraw
```

The main redraw output is:

- `bio_bias/<score>/outputs/<RUN_NAME>/paper_redraw/biobias_primary_alpha_4panel_redraw.pdf`

The same command also writes the per-panel PDFs and a small manifest CSV in the same directory.

Repeat the same workflow for:

- `bio_bias/saps/saps.ipynb`
- `bio_bias/raps/raps.ipynb`

### `FACET`

Launch the notebook from the `FACET` folder:

```bash
cd FACET
jupyter notebook facet_raps.ipynb
```

Run all cells. The notebook writes its run folder under `FACET/outputs/<RUN_NAME>/` and already exports the main fixed-alpha plots plus appendix files.

To also build the compact export from that run folder, run:

```bash
cd FACET
python outputs/data_plot.py outputs/<RUN_NAME>
```

The main files are:

- `FACET/outputs/<RUN_NAME>/paper_ready/main_text/figures/figure_primary_overview.{png,pdf}`
- `FACET/outputs/<RUN_NAME>/paper_ready/main_text/figures/figure_primary_floor_check.{png,pdf}`
- `FACET/outputs/<RUN_NAME>/paper_ready/main_text/tables/table_primary_summary.{csv,tex}`

Appendix files and a full file list are recorded in `paper_ready/manifest.json`.

## Notes

- `multiNLI/*/outputs/data_plot.py`, `bio_bias/*/outputs/data_plot.py`, and `FACET/outputs/data_plot.py` are the normal second step after the notebook run finishes.
- `multiNLI/emp_cal_size/data_plot.py`, `bio_bias/emp_cal_size/data_plot.py`, and `FACET/emp_cal_size/data_plot.py` are separate calibration-size analyses. They are not part of the main fresh-run path above and expect `result*.zip` bundles next to the script when used.
