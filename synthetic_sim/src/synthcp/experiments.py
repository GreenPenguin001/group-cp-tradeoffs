from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .distributions import (
    make_multigroup_coverage_to_size_hard,
    make_multigroup_gaussian,
    make_multigroup_gamma,
    make_multigroup_size_to_coverage_hard,
    make_multigroup_t_heavy,
    make_multigroup_wide_gaussian,
    make_two_group_gaussian,
)
from .metrics import mc_run_coverage_to_size, mc_run_pooled_metrics, mc_run_size_to_coverage
#from .plots import plot_multigroup_main, plot_seed_stability, plot_coverage_to_size_main, plot_two_group_main, plot_size_to_coverage_main
from .plots import (
    plot_multigroup_main,
    plot_seed_stability,
    plot_coverage_to_size_main,
    plot_two_group_main,
    plot_size_to_coverage_main,
    plot_bidirectional_policy_main,
)

def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)



def run_seeded_sweep(
    experiment: str,
    sweep_values: np.ndarray,
    make_dists: Callable[[float], list],
    runner: Callable,
    alpha: float,
    seeds: int,
    outdir: str,
    n_cal: int,
    n_test: int,
):
    out = Path(outdir)
    summary_rows = []
    group_rows = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        for sweep in sweep_values:
            dists = make_dists(float(sweep))
            weights = np.full(len(dists), 1.0 / len(dists))
            gdf, summary = runner(dists, weights, alpha, rng, n_cal, n_test)
            gdf.insert(0, "seed", seed)
            gdf.insert(1, "sweep", sweep)
            gdf.insert(2, "experiment", experiment)
            group_rows.append(gdf)
            summary_rows.append({"seed": seed, "sweep": sweep, "experiment": experiment, "n_cal": n_cal, "n_test": n_test, **summary})
    summary_df = pd.DataFrame(summary_rows)
    group_df = pd.concat(group_rows, ignore_index=True)
    save_csv(summary_df, out / "summary_metrics.csv")
    save_csv(group_df, out / "group_metrics.csv")
    return summary_df, group_df



def experiment_two_group(outdir: str, alpha: float = 0.1, seeds: int = 30, n_cal: int = 400, n_test: int = 4000):
    sweep_values = np.linspace(0.0, 2.0, 20)
    return run_seeded_sweep("two_group", sweep_values, make_two_group_gaussian, mc_run_pooled_metrics, alpha, seeds, outdir, n_cal, n_test)



def experiment_multigroup(outdir: str, alpha: float = 0.1, seeds: int = 30, family: str = "gaussian", n_cal: int = 400, n_test: int = 4000):
    sweep_values = np.linspace(0.0, 1.4, 20)
    family_map = {
        "gaussian": make_multigroup_gaussian,
        "wide": make_multigroup_wide_gaussian,
        "gamma": make_multigroup_gamma,
        "t": make_multigroup_t_heavy,
    }
    make_dists = family_map[family]
    return run_seeded_sweep(f"multigroup_{family}", sweep_values, make_dists, mc_run_pooled_metrics, alpha, seeds, outdir, n_cal, n_test)



def experiment_coverage_to_size(outdir: str, alpha: float = 0.1, seeds: int = 30, n_cal: int = 400, n_test: int = 4000):
    sweep_values = np.linspace(0.0, 1.4, 20)
    return run_seeded_sweep("coverage_to_size", sweep_values, make_multigroup_gaussian, mc_run_coverage_to_size, alpha, seeds, outdir, n_cal, n_test)



def experiment_coverage_to_size_hard(outdir: str, alpha: float = 0.1, seeds: int = 30, n_cal: int = 400, n_test: int = 4000):
    sweep_values = np.linspace(0.0, 1.8, 16)
    return run_seeded_sweep("coverage_to_size_hard", sweep_values, make_multigroup_coverage_to_size_hard, mc_run_coverage_to_size, alpha, seeds, outdir, n_cal, n_test)



def experiment_size_to_coverage(outdir: str, alpha: float = 0.1, seeds: int = 30, n_cal: int = 400, n_test: int = 4000):
    sweep_values = np.linspace(0.0, 1.4, 20)
    return run_seeded_sweep("size_to_coverage", sweep_values, make_multigroup_gaussian, mc_run_size_to_coverage, alpha, seeds, outdir, n_cal, n_test)



def experiment_size_to_coverage_hard(outdir: str, alpha: float = 0.1, seeds: int = 30, n_cal: int = 400, n_test: int = 4000):
    sweep_values = np.linspace(0.0, 2.0, 15)
    return run_seeded_sweep("size_to_coverage_hard", sweep_values, make_multigroup_size_to_coverage_hard, mc_run_size_to_coverage, alpha, seeds, outdir, n_cal, n_test)



def build_paper_outputs(base_outdir: str):
    base = Path(base_outdir)
    paper_dir = base / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)

    s_two = pd.read_csv(base / "two_group" / "summary_metrics.csv")
    g_two = pd.read_csv(base / "two_group" / "group_metrics.csv")
    s_mg = pd.read_csv(base / "multigroup_gaussian" / "summary_metrics.csv")
    s_wide = pd.read_csv(base / "multigroup_wide" / "summary_metrics.csv")
    s_gamma = pd.read_csv(base / "multigroup_gamma" / "summary_metrics.csv")
    s_t = pd.read_csv(base / "multigroup_t" / "summary_metrics.csv")
    s_c2s = pd.read_csv(base / "coverage_to_size_hard" / "summary_metrics.csv")
    g_c2s = pd.read_csv(base / "coverage_to_size_hard" / "group_metrics.csv")
    s_s2c = pd.read_csv(base / "size_to_coverage_hard" / "summary_metrics.csv")
    g_s2c = pd.read_csv(base / "size_to_coverage_hard" / "group_metrics.csv")

    sweep_two = float(s_two.loc[s_two["sweep"].sub(1.0).abs().idxmin(), "sweep"])
    example_two = g_two[(g_two["seed"] == 0) & (np.isclose(g_two["sweep"], sweep_two))].copy()
    plot_two_group_main(example_two, s_two, paper_dir / "figure1_two_group.pdf")
    plot_multigroup_main(s_mg, s_wide, s_gamma, s_t, paper_dir / "figure2_multigroup.pdf")

    sweep_c2s = float(s_c2s.loc[s_c2s["sweep"].sub(1.2).abs().idxmin(), "sweep"])
    ex_c2s = g_c2s[(g_c2s["seed"] == 0) & (np.isclose(g_c2s["sweep"], sweep_c2s))].copy()
    plot_coverage_to_size_main(s_c2s, ex_c2s, paper_dir / "figure3_coverage_to_size.pdf")

    sweep_s2c = float(s_s2c.loc[s_s2c["sweep"].sub(1.2).abs().idxmin(), "sweep"])
    ex_s2c = g_s2c[(g_s2c["seed"] == 0) & (np.isclose(g_s2c["sweep"], sweep_s2c))].copy()
    plot_size_to_coverage_main(s_s2c, ex_s2c, paper_dir / "figure4_size_to_coverage.pdf")

    plot_bidirectional_policy_main(
    cov2size_summary_df=s_c2s,
    cov2size_example_df=ex_c2s,
    size2cov_summary_df=s_s2c,
    size2cov_example_df=ex_s2c,
    outpath=paper_dir / "figure1x4_bidirectional.pdf",)

    combined = pd.concat([s_two, s_mg, s_wide, s_gamma, s_t, s_c2s, s_s2c], ignore_index=True, sort=False)
    plot_seed_stability(combined, paper_dir / "appendix_seed_stability.pdf")

    setup_table = pd.DataFrame(
        [
            ["two_group", "MC two-group Gaussian shift", 2, "delta", 0.1, "L2 distortion", "pooled-threshold finite-sample mechanism"],
            ["multigroup_gaussian", "MC 4-group Gaussian shift", 4, "scale", 0.1, "L2 distortion", "multi-group lower bound under sampling"],
            ["multigroup_wide", "MC 4-group wide Gaussian family", 4, "scale", 0.1, "L2 distortion", "variance heterogeneity under sampling"],
            ["multigroup_gamma", "MC 4-group Gamma family", 4, "scale", 0.1, "L2 distortion", "skewed-family heterogeneity under sampling"],
            ["multigroup_t", "MC 4-group heavy-tail t family", 4, "scale", 0.1, "L2 distortion", "heavy-tail heterogeneity under sampling"],
            ["coverage_to_size_hard", "MC 8-group asymmetric Gaussian-mixture family", 8, "scale", 0.1, "size disparity", "hard coverage -> size tradeoff with a visible empirical gap"],
            ["size_to_coverage_hard", "MC 8-group heavy-tail / Gaussian mixture family", 8, "scale", 0.1, "coverage distortion", "hard size -> coverage tradeoff with a larger visible empirical gap"],
        ],
        columns=["experiment", "family", "groups", "sweep", "alpha", "main_metric", "target"],
    )
    save_csv(setup_table, paper_dir / "setup_table.csv")
    save_csv(combined, paper_dir / "all_summary_metrics.csv")



def run_paper(base_outdir: str, alpha: float = 0.1, seeds: int = 30, n_cal: int = 400, n_test: int = 4000):
    base = Path(base_outdir)
    experiment_two_group(str(base / "two_group"), alpha, seeds, n_cal, n_test)
    experiment_multigroup(str(base / "multigroup_gaussian"), alpha, seeds, "gaussian", n_cal, n_test)
    experiment_multigroup(str(base / "multigroup_wide"), alpha, seeds, "wide", n_cal, n_test)
    experiment_multigroup(str(base / "multigroup_gamma"), alpha, seeds, "gamma", n_cal, n_test)
    experiment_multigroup(str(base / "multigroup_t"), alpha, seeds, "t", n_cal, n_test)
    experiment_coverage_to_size(str(base / "coverage_to_size"), alpha, seeds, n_cal, n_test)
    experiment_size_to_coverage(str(base / "size_to_coverage"), alpha, seeds, n_cal, n_test)
    experiment_coverage_to_size_hard(str(base / "coverage_to_size_hard"), alpha, seeds, int(n_cal/2), int(n_test/5))
    experiment_size_to_coverage_hard(str(base / "size_to_coverage_hard"), alpha, seeds, int(n_cal/2), int(n_test/5))
    build_paper_outputs(str(base))
