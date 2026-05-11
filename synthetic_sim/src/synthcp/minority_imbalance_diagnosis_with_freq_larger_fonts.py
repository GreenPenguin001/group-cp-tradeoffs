from __future__ import annotations

"""Diagnostics for imbalance-induced minority-group effects in the synthetic package.

This module adds two diagnostics that are not covered by the main paper bundle:

1. A multi-group imbalance experiment, where pooled calibration is performed with
   non-uniform group masses and a fixed total calibration budget. The main
   visualization now tracks the rarest (minority) group directly rather than a
   worst-group summary, so the group identity is preserved.
2. Ratio tables for the coverage->size and size->coverage tradeoff experiments,
   reporting empirical/oracle ratios at representative heterogeneity locations as
   n_cal increases.

Recommended usage (from the project root, after setting PYTHONPATH=src):

export PYTHONPATH=src
python -m synthcp.2minority_imbalance_diagnosis_with_freq_larger_fonts \
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

The script saves both figures and CSV tables under the requested output directory.
"""

import argparse
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update(
    {
        "figure.dpi": 220,
        "savefig.dpi": 400,
        "font.size": 15,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 16,
        "lines.linewidth": 2.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


if __package__ in {None, ""}:  # pragma: no cover - convenience for direct execution
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from synthcp.distributions import (
        make_multigroup_coverage_to_size_hard,
        make_multigroup_gaussian,
        make_multigroup_gamma,
        make_multigroup_size_to_coverage_hard,
        make_multigroup_t_heavy,
        make_multigroup_wide_gaussian,
    )
    from synthcp.metrics import (
        mc_run_coverage_to_size,
        mc_run_size_to_coverage,
        oracle_pooled_metrics,
        sample_quantile,
        sigma_weighted,
        weighted_l2,
    )
    from synthcp.plots import add_grid
else:
    from .distributions import (
        make_multigroup_coverage_to_size_hard,
        make_multigroup_gaussian,
        make_multigroup_gamma,
        make_multigroup_size_to_coverage_hard,
        make_multigroup_t_heavy,
        make_multigroup_wide_gaussian,
    )
    from .metrics import (
        mc_run_coverage_to_size,
        mc_run_size_to_coverage,
        oracle_pooled_metrics,
        sample_quantile,
        sigma_weighted,
        weighted_l2,
    )
    from .plots import add_grid


EMP_COLOR = "C0"
BND_COLOR = "C1"
BAL_COLOR = "0.45"
MINORITY_COLOR = "C3"

DEFAULT_IMBALANCE_WEIGHTS = np.array([0.60, 0.25, 0.10, 0.05], dtype=float)

MULTIGROUP_FAMILIES: dict[str, Callable[[float], list]] = {
    "gaussian": make_multigroup_gaussian,
    "wide": make_multigroup_wide_gaussian,
    "gamma": make_multigroup_gamma,
    "t": make_multigroup_t_heavy,
}

TRADEOFF_EXPERIMENTS = {
    "coverage_to_size_hard": {
        "make_dists": make_multigroup_coverage_to_size_hard,
        "runner": mc_run_coverage_to_size,
        "sweep_values": np.linspace(0.0, 1.8, 16),
        "ratio_metric": "ratio_to_bound",
        "sigma_col": "sigma_delta_oracle",
        "display_name": "coverage-to-size",
    },
    "size_to_coverage_hard": {
        "make_dists": make_multigroup_size_to_coverage_hard,
        "runner": mc_run_size_to_coverage,
        "sweep_values": np.linspace(0.0, 2.0, 15),
        "ratio_metric": "ratio_to_bound",
        "sigma_col": "sigma_delta_oracle",
        "display_name": "size-to-coverage",
    },
}


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)


def parse_int_list(text: str) -> list[int]:
    values = [int(tok.strip()) for tok in text.split(",") if tok.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def parse_float_weights(text: str) -> np.ndarray:
    weights = np.array([float(tok.strip()) for tok in text.split(",") if tok.strip()], dtype=float)
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError("Expected a comma-separated list of weights.")
    if np.any(weights <= 0):
        raise ValueError("All imbalance weights must be strictly positive.")
    return weights / weights.sum()


def counts_from_weights(total: int, weights: np.ndarray) -> np.ndarray:
    if total < len(weights):
        raise ValueError(f"total calibration budget {total} is smaller than the number of groups {len(weights)}")
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    counts = np.ones(len(weights), dtype=int)
    remaining = total - len(weights)
    expected = remaining * weights
    base = np.floor(expected).astype(int)
    counts += base
    remainder = remaining - int(base.sum())
    if remainder > 0:
        frac = expected - base
        order = np.argsort(-frac)
        counts[order[:remainder]] += 1
    return counts


def grouped_mean(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    g = (
        df.groupby("sweep")
        .agg(
            x_mean=(x, "mean"),
            y_mean=(y, "mean"),
            y_q10=(y, lambda s: float(np.quantile(s, 0.10))),
            y_q90=(y, lambda s: float(np.quantile(s, 0.90))),
        )
        .reset_index()
        .sort_values("sweep")
    )
    return g


def select_representative_sweeps(sweeps: np.ndarray, num_points: int) -> np.ndarray:
    sweeps = np.asarray(sorted(set(float(x) for x in sweeps)), dtype=float)
    positive = sweeps[sweeps > sweeps.min() + 1e-12]
    if len(positive) == 0:
        raise ValueError("No positive heterogeneity sweep values were found.")
    if num_points >= len(positive):
        return positive
    idx = np.linspace(0, len(positive) - 1, num_points)
    idx = np.unique(np.round(idx).astype(int))
    return positive[idx]


def mc_run_pooled_metrics_with_total_budget(
    dists: list,
    weights: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
    n_cal_total: int,
    n_test_per_group: int,
):
    """Finite-sample pooled-threshold experiment with imbalanced group masses.

    Unlike the main package, this diagnostic uses a *total* calibration budget and
    allocates calibration points proportionally to the requested group weights.
    This lets the pooled threshold inherit the intended imbalance, which is the
    relevant bottleneck mechanism here.
    """
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    cal_counts = counts_from_weights(n_cal_total, weights)
    cal = [d.rvs(int(n), rng) for d, n in zip(dists, cal_counts)]
    test = [d.rvs(n_test_per_group, rng) for d in dists]

    q_g_hat = np.array([sample_quantile(x, 1 - alpha) for x in cal], dtype=float)
    q_hat = sample_quantile(np.concatenate(cal), 1 - alpha)
    coverage_hat = np.array([np.mean(x <= q_hat) for x in test], dtype=float)
    eps_hat = coverage_hat - (1 - alpha)
    sigma_delta_hat = sigma_weighted(q_g_hat, weights)

    oracle = oracle_pooled_metrics(dists, weights, alpha)
    q_g_oracle = np.asarray(oracle["q_g_oracle"], dtype=float)
    minority_idx = int(np.argmin(weights))
    hardest_idx = int(np.argmax(q_g_oracle))

    group_df = pd.DataFrame(
        {
            "group": [d.name for d in dists],
            "weight": weights,
            "n_cal_group": cal_counts,
            "q_g_hat": q_g_hat,
            "q_pooled_hat": q_hat,
            "coverage_hat": coverage_hat,
            "eps_hat": eps_hat,
            "abs_eps_hat": np.abs(eps_hat),
            "q_g_oracle": q_g_oracle,
            "q_pooled_oracle": float(oracle["q_pooled_oracle"]),
            "is_minority": [i == minority_idx for i in range(len(dists))],
            "is_hardest": [i == hardest_idx for i in range(len(dists))],
        }
    )

    distortion_l2 = weighted_l2(eps_hat, weights)
    distortion_linf = float(np.max(np.abs(eps_hat)))
    minority_abs_gap = float(abs(eps_hat[minority_idx]))
    hardest_abs_gap = float(abs(eps_hat[hardest_idx]))

    summary = {
        "n_cal_total": n_cal_total,
        "n_test_per_group": n_test_per_group,
        "min_group_n_cal": int(cal_counts.min()),
        "max_group_n_cal": int(cal_counts.max()),
        "q_pooled_hat": float(q_hat),
        "sigma_delta_hat": float(sigma_delta_hat),
        "sigma_delta_oracle": float(oracle["sigma_delta_oracle"]),
        "distortion_l2": float(distortion_l2),
        "distortion_linf": distortion_linf,
        "minority_abs_gap": minority_abs_gap,
        "hardest_abs_gap": hardest_abs_gap,
        "bound_oracle": float(oracle["bound_oracle"]),
        "ratio_to_bound": float(distortion_l2 / max(oracle["bound_oracle"], 1e-12)),
        "minority_to_l2": float(minority_abs_gap / max(distortion_l2, 1e-12)),
        "hardest_to_l2": float(hardest_abs_gap / max(distortion_l2, 1e-12)),
        "minority_group": group_df.loc[minority_idx, "group"],
        "hardest_group": group_df.loc[hardest_idx, "group"],
    }
    return group_df, summary




def build_minority_is_largest_by_sweep(group_df: pd.DataFrame, summary_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize how often the minority group is the largest-gap group across seeds.

    The output is a long CSV indexed by (family, sweep), with one row per
    heterogeneity level for the imbalanced scenario.
    """
    imb_groups = group_df[group_df["scenario"] == "imbalanced"].copy()
    if imb_groups.empty:
        return pd.DataFrame(
            columns=[
                "family",
                "sweep",
                "sigma_delta_oracle",
                "minority_is_largest_freq",
                "minority_is_largest_count",
                "num_trials",
            ]
        )

    keys = ["family", "scenario", "seed", "sweep"]
    max_abs = imb_groups.groupby(keys)["abs_eps_hat"].transform("max")
    tol = 1e-12
    minority_rows = imb_groups[imb_groups["is_minority"]].copy()
    minority_rows["minority_is_largest"] = (
        minority_rows["abs_eps_hat"] >= (max_abs[minority_rows.index] - tol)
    ).astype(int)

    sigma_map = (
        summary_df[summary_df["scenario"] == "imbalanced"][["family", "sweep", "sigma_delta_oracle"]]
        .drop_duplicates()
    )

    freq_df = (
        minority_rows.groupby(["family", "sweep"], as_index=False)
        .agg(
            minority_is_largest_freq=("minority_is_largest", "mean"),
            minority_is_largest_count=("minority_is_largest", "sum"),
            num_trials=("minority_is_largest", "size"),
        )
        .sort_values(["family", "sweep"])
    )
    freq_df = freq_df.merge(sigma_map, on=["family", "sweep"], how="left")
    return freq_df[
        [
            "family",
            "sweep",
            "sigma_delta_oracle",
            "minority_is_largest_freq",
            "minority_is_largest_count",
            "num_trials",
        ]
    ]


def run_multigroup_imbalance(
    outdir: Path,
    alpha: float,
    seeds: int,
    n_cal_total: int,
    n_test_per_group: int,
    imbalance_weights: np.ndarray,
    num_table_points: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outdir.mkdir(parents=True, exist_ok=True)
    all_summary: list[dict] = []
    all_groups: list[pd.DataFrame] = []

    balanced_weights = np.full(len(imbalance_weights), 1.0 / len(imbalance_weights), dtype=float)
    #sweep_values = np.linspace(0.0, 1.4, 20)
    sweep_values = np.linspace(0.0, 2.0, 25)

    for family, make_dists in MULTIGROUP_FAMILIES.items():
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            for sweep in sweep_values:
                dists = make_dists(float(sweep))
                for scenario, weights in [("balanced", balanced_weights), ("imbalanced", imbalance_weights)]:
                    gdf, summary = mc_run_pooled_metrics_with_total_budget(
                        dists=dists,
                        weights=weights,
                        alpha=alpha,
                        rng=rng,
                        n_cal_total=n_cal_total,
                        n_test_per_group=n_test_per_group,
                    )
                    gdf.insert(0, "scenario", scenario)
                    gdf.insert(0, "family", family)
                    gdf.insert(0, "sweep", float(sweep))
                    gdf.insert(0, "seed", seed)
                    all_groups.append(gdf)

                    all_summary.append(
                        {
                            "seed": seed,
                            "sweep": float(sweep),
                            "family": family,
                            "scenario": scenario,
                            **summary,
                        }
                    )

            print(f"[multigroup_imbalance][{family}] seed {seed} finished", flush=True)

    summary_df = pd.DataFrame(all_summary)
    group_df = pd.concat(all_groups, ignore_index=True)
    save_dataframe(summary_df, outdir / "multigroup_imbalance_summary.csv")
    save_dataframe(group_df, outdir / "multigroup_imbalance_group_metrics.csv")
    minority_freq_df = build_minority_is_largest_by_sweep(group_df, summary_df)
    save_dataframe(minority_freq_df, outdir / "minority_is_largest_by_sweep.csv")

    # Representative table at a few heterogeneity locations.
    table_rows = []
    for family in MULTIGROUP_FAMILIES:
        fam_df = summary_df[summary_df["family"] == family]
        selected_sweeps = select_representative_sweeps(fam_df["sweep"].unique(), num_table_points)
        for sweep in selected_sweeps:
            bal = fam_df[(fam_df["scenario"] == "balanced") & np.isclose(fam_df["sweep"], sweep)]
            imb = fam_df[(fam_df["scenario"] == "imbalanced") & np.isclose(fam_df["sweep"], sweep)]
            table_rows.append(
                {
                    "family": family,
                    "sweep": float(sweep),
                    "sigma_delta_oracle": float(imb["sigma_delta_oracle"].mean()),
                    "balanced_l2_mean": float(bal["distortion_l2"].mean()),
                    "imbalanced_l2_mean": float(imb["distortion_l2"].mean()),
                    "minority_abs_gap_mean": float(imb["minority_abs_gap"].mean()),
                    "minority_to_l2_mean": float(imb["minority_to_l2"].mean()),
                    "ratio_to_bound_mean": float(imb["ratio_to_bound"].mean()),
                    "min_group_n_cal": int(round(imb["min_group_n_cal"].mean())),
                    "max_group_n_cal": int(round(imb["max_group_n_cal"].mean())),
                }
            )
    table_df = pd.DataFrame(table_rows).sort_values(["family", "sigma_delta_oracle"])
    save_dataframe(table_df, outdir / "multigroup_imbalance_table.csv")

    plot_multigroup_imbalance(summary_df, outdir / "multigroup_imbalance.pdf")
    plot_multigroup_imbalance(summary_df, outdir / "multigroup_imbalance.png")

    setup_df = pd.DataFrame(
        {
            "setting": ["balanced_weights", "imbalance_weights", "n_cal_total", "n_test_per_group", "alpha", "seeds"],
            "value": [
                ",".join(f"{x:.4f}" for x in balanced_weights),
                ",".join(f"{x:.4f}" for x in imbalance_weights),
                int(n_cal_total),
                int(n_test_per_group),
                float(alpha),
                int(seeds),
            ],
        }
    )
    save_dataframe(setup_df, outdir / "multigroup_imbalance_setup.csv")
    return summary_df, group_df, table_df


def plot_multigroup_imbalance(summary_df: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.8, 9.8), sharex=False)
    titles = {
        "gaussian": "Gaussian family",
        "wide": "Wide Gaussian family",
        "gamma": "Gamma family",
        "t": "Heavy-tail t family",
    }

    for ax, family in zip(axes.ravel(), MULTIGROUP_FAMILIES.keys()):
        fam = summary_df[summary_df["family"] == family]
        imb_l2 = grouped_mean(fam[fam["scenario"] == "imbalanced"], "sigma_delta_oracle", "distortion_l2")
        imb_minority = grouped_mean(fam[fam["scenario"] == "imbalanced"], "sigma_delta_oracle", "minority_abs_gap")
        imb_bound = grouped_mean(fam[fam["scenario"] == "imbalanced"], "sigma_delta_oracle", "bound_oracle")

        ax.plot(
            imb_bound["x_mean"],
            imb_bound["y_mean"],
            linestyle="--",
            linewidth=2.1,
            color=BND_COLOR,
            label="Oracle bound",
        )

        ax.plot(
            imb_l2["x_mean"],
            imb_l2["y_mean"],
            linewidth=2.3,
            color=EMP_COLOR,
            label="Imbalanced L2",
        )
        ax.fill_between(
            imb_l2["x_mean"],
            imb_l2["y_q10"],
            imb_l2["y_q90"],
            color=EMP_COLOR,
            alpha=0.12,
        )
        ax.plot(
            imb_minority["x_mean"],
            imb_minority["y_mean"],
            linewidth=2.3,
            color=MINORITY_COLOR,
            label="Minority abs. gap",
        )
        ax.fill_between(
            imb_minority["x_mean"],
            imb_minority["y_q10"],
            imb_minority["y_q90"],
            color=MINORITY_COLOR,
            alpha=0.10,
        )

        ax.set_title(titles[family], pad=8)
        ax.set_xlabel("Heterogeneity Scale")
        ax.set_ylabel("Distortion")
        ax.tick_params(axis="both", labelsize=11.5)
        add_grid(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        fontsize=16,
        handlelength=2.4,
        columnspacing=1.8,
    )
    fig.subplots_adjust(top=0.88, hspace=0.30, wspace=0.20)
    save_figure(fig, outpath)


def run_tradeoff_ratio_tables(
    outdir: Path,
    alpha: float,
    seeds: int,
    n_cal_values: list[int],
    n_test: int,
    num_sigma_points: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    uniform_weights = np.full(8, 1.0 / 8.0, dtype=float)

    for exp_name, cfg in TRADEOFF_EXPERIMENTS.items():
        make_dists = cfg["make_dists"]
        runner = cfg["runner"]
        sweep_values = cfg["sweep_values"]
        sigma_col = cfg["sigma_col"]
        ratio_metric = cfg["ratio_metric"]

        raw_rows: list[dict] = []
        for n_cal in n_cal_values:
            for seed in range(seeds):
                rng = np.random.default_rng(seed)
                for sweep in sweep_values:
                    dists = make_dists(float(sweep))
                    _, summary = runner(dists, uniform_weights, alpha, rng, int(n_cal), int(n_test))
                    pooled_oracle = oracle_pooled_metrics(dists, uniform_weights, alpha)
                    raw_rows.append(
                        {
                            "experiment": exp_name,
                            "display_name": cfg["display_name"],
                            "seed": seed,
                            "sweep": float(sweep),
                            "n_cal": int(n_cal),
                            "sigma_delta_oracle": float(pooled_oracle["sigma_delta_oracle"]),
                            **summary,
                        }
                    )

                print(f"[{exp_name}][n_cal={int(n_cal)}] seed {seed} finished", flush=True)

        raw_df = pd.DataFrame(raw_rows)
        save_dataframe(raw_df, outdir / f"{exp_name}_ratio_raw.csv")

        selected_sweeps = select_representative_sweeps(sweep_values, num_sigma_points)
        selected_df = raw_df[raw_df["sweep"].isin(selected_sweeps)].copy()

        long_df = (
            selected_df.groupby(["n_cal", "sweep", sigma_col])
            .agg(
                ratio_mean=(ratio_metric, "mean"),
                ratio_std=(ratio_metric, "std"),
                ratio_q10=(ratio_metric, lambda s: float(np.quantile(s, 0.10))),
                ratio_q90=(ratio_metric, lambda s: float(np.quantile(s, 0.90))),
            )
            .reset_index()
            .sort_values(["n_cal", sigma_col])
        )
        long_df["ratio_std"] = long_df["ratio_std"].fillna(0.0)
        save_dataframe(long_df, outdir / f"{exp_name}_ratio_table_long.csv")

        sigma_levels = sorted(long_df[sigma_col].unique())
        wide_mean = long_df.pivot(index="n_cal", columns=sigma_col, values="ratio_mean").reset_index()
        wide_mean.columns = ["n_cal"] + [f"sigma_delta_{x:.4f}" for x in sigma_levels]
        save_dataframe(wide_mean, outdir / f"{exp_name}_ratio_table_wide.csv")

        wide_std = long_df.pivot(index="n_cal", columns=sigma_col, values="ratio_std").reset_index()
        wide_std.columns = ["n_cal"] + [f"sigma_delta_{x:.4f}" for x in sigma_levels]
        save_dataframe(wide_std, outdir / f"{exp_name}_ratio_table_std.csv")

        setup_df = pd.DataFrame(
            {
                "setting": ["alpha", "seeds", "n_test", "n_cal_values", "selected_sweeps", "selected_sigma_delta"],
                "value": [
                    float(alpha),
                    int(seeds),
                    int(n_test),
                    ",".join(str(x) for x in n_cal_values),
                    ",".join(f"{x:.4f}" for x in selected_sweeps),
                    ",".join(f"{x:.4f}" for x in sigma_levels),
                ],
            }
        )
        save_dataframe(setup_df, outdir / f"{exp_name}_ratio_table_setup.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Imbalance diagnostics for the synthetic conformal package (minority-group version)")
    parser.add_argument("--outdir", type=str, default="outputs_mc_imbalance")
    parser.add_argument("--alpha", type=float, default=0.1)

    parser.add_argument("--imbalance-seeds", type=int, default=40)
    parser.add_argument("--imbalance-n-cal-total", type=int, default=400)
    parser.add_argument("--imbalance-n-test", type=int, default=4000)
    parser.add_argument("--imbalance-weights", type=str, default="0.60,0.25,0.10,0.05")

    parser.add_argument("--ratio-seeds", type=int, default=40)
    parser.add_argument("--ratio-n-test", type=int, default=800)
    parser.add_argument("--ratio-n-cals", type=str, default="12,25,50,100,200,400")
    parser.add_argument("--num-table-points", type=int, default=5)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    imbalance_weights = parse_float_weights(args.imbalance_weights)
    if len(imbalance_weights) != 4:
        raise ValueError("The multigroup imbalance diagnostic expects exactly four group weights.")
    ratio_n_cals = parse_int_list(args.ratio_n_cals)

    run_multigroup_imbalance(
        outdir=outdir / "multigroup_imbalance",
        alpha=float(args.alpha),
        seeds=int(args.imbalance_seeds),
        n_cal_total=int(args.imbalance_n_cal_total),
        n_test_per_group=int(args.imbalance_n_test),
        imbalance_weights=imbalance_weights,
        num_table_points=int(args.num_table_points),
    )

    run_tradeoff_ratio_tables(
        outdir=outdir / "tradeoff_ratio_tables",
        alpha=float(args.alpha),
        seeds=int(args.ratio_seeds),
        n_cal_values=ratio_n_cals,
        n_test=int(args.ratio_n_test),
        num_sigma_points=int(args.num_table_points),
    )


if __name__ == "__main__":
    main()
