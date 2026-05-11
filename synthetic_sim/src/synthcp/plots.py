
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors


EMP_COLOR = "C0"
BND_COLOR = "C1"
GRID_ALPHA = 0.30
SIZE_COLOR = "#E15759"
COVERAGE_COLOR = "#59A14F"


plt.rcParams.update(
    {
        "figure.dpi": 180,
        "savefig.dpi": 400,
        "font.size": 15.6,
        "axes.titlesize": 16.8,
        "axes.labelsize": 17.6,
        "xtick.labelsize": 15.0,
        "ytick.labelsize": 15.0,
        "legend.fontsize": 15.6,
        "lines.linewidth": 5.4,
        "axes.linewidth": 1.10,
        "xtick.major.width": 1.10,
        "ytick.major.width": 1.10,
        "xtick.major.size": 4.8,
        "ytick.major.size": 4.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "mathtext.fontset": "dejavusans",
    }
)


def savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=400, bbox_inches="tight")
    plt.close(fig)



def add_grid(ax):
    ax.set_axisbelow(True)
    ax.grid(
        True,
        which="major",
        linestyle="--",
        linewidth=0.95,
        alpha=0.36,
    )


def lighten_color(color: str, factor: float = 0.62) -> str:
    rgb = np.array(mcolors.to_rgb(color))
    mixed = rgb + (1.0 - rgb) * float(factor)
    return mcolors.to_hex(np.clip(mixed, 0.0, 1.0))



def _group_by_sweep(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    g = (
        df.groupby("sweep")
        .agg(
            x_mean=(x, "mean"),
            y_mean=(y, "mean"),
            y_std=(y, "std"),
            y_q10=(y, lambda s: float(np.quantile(s, 0.10))),
            y_q20=(y, lambda s: float(np.quantile(s, 0.20))),
            y_q80=(y, lambda s: float(np.quantile(s, 0.80))),
            y_q90=(y, lambda s: float(np.quantile(s, 0.90))),
        )
        .reset_index()
        .sort_values("sweep")
    )
    g["y_std"] = g["y_std"].fillna(0.0)
    return g



def _band_from_grouped(g: pd.DataFrame, band: str):
    mean = g["y_mean"].to_numpy(dtype=float)
    if band == "q10q90":
        lower = g["y_q10"].to_numpy(dtype=float)
        upper = g["y_q90"].to_numpy(dtype=float)
    elif band == "quantile":
        lower = g["y_q20"].to_numpy(dtype=float)
        upper = g["y_q80"].to_numpy(dtype=float)
    else:
        std = g["y_std"].to_numpy(dtype=float)
        lower, upper = mean - std, mean + std
    return lower, upper



def scatter_seeds(
    ax,
    df: pd.DataFrame,
    x: str,
    y: str,
    jitter_scale: float = 0.0,
    alpha: float = 0.18,
    size: float = 18,
    *,
    center_on_sweep_mean: bool = False,
):
    if center_on_sweep_mean:
        xmap = df.groupby("sweep")[x].mean()
        xs = df["sweep"].map(xmap).to_numpy(dtype=float)
    else:
        xs = df[x].to_numpy(dtype=float).copy()
    ys = df[y].to_numpy(dtype=float)
    if jitter_scale > 0:
        rng = np.random.default_rng(0)
        xs = xs + rng.normal(0.0, jitter_scale, size=len(xs))
    ax.scatter(xs, ys, s=size, alpha=alpha, color=EMP_COLOR, zorder=1)



def plot_empirical_mean_band(
    ax,
    df: pd.DataFrame,
    x: str,
    y: str,
    label: str = "MC empirical",
    *,
    show_scatter: bool = True,
    jitter_scale: float = 0.0,
    band: str = "std",
    line_width: float = 2.8,
    marker_size: float = 0.0,
    scatter_alpha: float = 0.09,
    scatter_size: float = 20.0,
    band_alpha: float = 0.16,
    zorder_line: int = 5,
    center_scatter_on_sweep_mean: bool = False,
):
    g = _group_by_sweep(df, x, y)
    xs = g["x_mean"].to_numpy(dtype=float)
    mean = g["y_mean"].to_numpy(dtype=float)
    lower, upper = _band_from_grouped(g, band)

    if show_scatter:
        scatter_seeds(
            ax,
            df,
            x,
            y,
            jitter_scale=jitter_scale,
            alpha=scatter_alpha,
            size=scatter_size,
            center_on_sweep_mean=center_scatter_on_sweep_mean,
        )

    ax.fill_between(
        xs,
        lower,
        upper,
        color=EMP_COLOR,
        alpha=band_alpha,
        zorder=zorder_line - 1,
    )
    marker = None if marker_size <= 0 else "o"
    ax.plot(
        xs,
        mean,
        color=EMP_COLOR,
        marker=marker,
        linewidth=line_width,
        markersize=max(marker_size, 0.0),
        label=label,
        zorder=zorder_line,
    )



def plot_oracle_curve(
    ax,
    df: pd.DataFrame,
    x: str,
    y: str,
    label: str = "Oracle bound",
    *,
    line_width: float = 2.8,
    marker_size: float = 0.0,
    alpha: float = 0.98,
    zorder: int = 3,
):
    g = _group_by_sweep(df, x, y)
    xs = g["x_mean"].to_numpy(dtype=float)
    mean = g["y_mean"].to_numpy(dtype=float)
    marker = None if marker_size <= 0 else "o"
    ax.plot(
        xs,
        mean,
        color=BND_COLOR,
        marker=marker,
        linestyle="--",
        linewidth=line_width,
        markersize=max(marker_size, 0.0),
        label=label,
        alpha=alpha,
        zorder=zorder,
    )



def _panel_letter(ax, label: str):
    ax.text(
        -0.10,
        1.10,
        label,
        transform=ax.transAxes,
        fontsize=16.4,
        fontweight="bold",
        va="bottom",
    )



def _compact_legend(ax):
    return ax.legend(
        frameon=False,
        loc="upper left",
        handlelength=2.4,
        borderaxespad=0.15,
        labelspacing=0.18,
    )



def draw_group_size_gaps(
    ax,
    example_df: pd.DataFrame,
    *,
    annotate: bool = True,
    title: str = "Group size gaps",
    ylabel: str = "Size gap",
):
    gap = example_df[["group", "size_gap"]].copy()
    values = gap["size_gap"].to_numpy(dtype=float)
    colors = [SIZE_COLOR if v >= 0 else lighten_color(SIZE_COLOR) for v in values]
    ax.axhline(0.0, linestyle="--", linewidth=1.45, color="gray")
    ax.bar(gap["group"], values, color=colors, edgecolor=SIZE_COLOR, linewidth=1.0, width=0.90, alpha=0.94)
    if annotate:
        span = max(np.max(np.abs(values)), 1e-8)
        off = 0.055 * span
        for i, v in enumerate(values):
            ax.text(
                i,
                v + (off if v >= 0 else -off),
                f"{v:+.2f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=12.8,
                fontweight="medium",
            )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.margins(x=0.02, y=0.18 if annotate else 0.08)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    add_grid(ax)



def draw_size_disparity(
    ax,
    summary_df: pd.DataFrame,
    *,
    title: str = "Size disparity vs heterogeneity",
    ylabel: str = "Size disparity",
):
    plot_oracle_curve(ax, summary_df, "sigma_delta_oracle", "bound_oracle", "Oracle bound", line_width=3.0)
    plot_empirical_mean_band(
        ax,
        summary_df,
        "sigma_delta_oracle",
        "size_disparity",
        "MC empirical",
        show_scatter=True,
        jitter_scale=0.008,
        band="q10q90",
        band_alpha=0.18,
        scatter_alpha=0.10,
        scatter_size=24,
        line_width=3.6,
        center_scatter_on_sweep_mean=True,
    )
    ax.set_title(title)
    ax.set_xlabel("Heterogeneity")
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    add_grid(ax)



def draw_group_coverage_shifts(
    ax,
    example_df: pd.DataFrame,
    *,
    annotate: bool = True,
    title: str = "Group coverage shifts",
    ylabel: str = "Coverage shift",
):
    gap = example_df[["group", "coverage_shift"]].copy()
    values = gap["coverage_shift"].to_numpy(dtype=float)
    colors = [COVERAGE_COLOR if v >= 0 else lighten_color(COVERAGE_COLOR) for v in values]
    ax.axhline(0.0, linestyle="--", linewidth=1.45, color="gray")
    ax.bar(gap["group"], values, color=colors, edgecolor=COVERAGE_COLOR, linewidth=1.0, width=0.90, alpha=0.94)
    if annotate:
        span = max(np.max(np.abs(values)), 1e-8)
        off = 0.055 * span
        for i, v in enumerate(values):
            ax.text(
                i,
                v + (off if v >= 0 else -off),
                f"{v:+.2f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=12.8,
                fontweight="medium",
            )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.margins(x=0.02, y=0.18 if annotate else 0.08)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    add_grid(ax)



def draw_coverage_distortion(
    ax,
    summary_df: pd.DataFrame,
    *,
    title: str = "Coverage distortion vs heterogeneity",
):
    plot_oracle_curve(ax, summary_df, "sigma_lambda_oracle", "bound_oracle", "Oracle bound", line_width=3.0)
    plot_empirical_mean_band(
        ax,
        summary_df,
        "sigma_lambda_oracle",
        "coverage_distortion",
        "MC empirical",
        show_scatter=True,
        jitter_scale=0.010,
        band="q10q90",
        band_alpha=0.18,
        scatter_alpha=0.10,
        scatter_size=24,
        line_width=3.6,
        center_scatter_on_sweep_mean=True,
    )
    ax.set_title(title)
    ax.set_xlabel("Heterogeneity")
    ax.set_ylabel("Coverage distortion")
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    add_grid(ax)



def plot_bidirectional_policy_main(
    cov2size_summary_df: pd.DataFrame,
    cov2size_example_df: pd.DataFrame,
    size2cov_summary_df: pd.DataFrame,
    size2cov_example_df: pd.DataFrame,
    outpath: Path,
):
    fig = plt.figure(figsize=(18.9, 5.55))
    # Give A/B their own small spacer while preserving the existing B/C and C/D spacing intent.
    gs = fig.add_gridspec(1, 7, width_ratios=[1.05, 0.07, 1.18, 0.12, 1.05, 0.09, 1.18], wspace=0.10)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 2])
    axC = fig.add_subplot(gs[0, 4])
    axD = fig.add_subplot(gs[0, 6])

    draw_group_size_gaps(
        axA,
        cov2size_example_df,
        annotate=True,
        title="Group size distortion",
        ylabel="Size distortion",
    )
    draw_size_disparity(
        axB,
        cov2size_summary_df,
        title="Size distortion vs heterogeneity",
        ylabel="Size distortion",
    )
    draw_group_coverage_shifts(
        axC,
        size2cov_example_df,
        annotate=True,
        title="Group coverage distortion",
        ylabel="Coverage distortion",
    )
    draw_coverage_distortion(axD, size2cov_summary_df, title="Coverage distortion vs heterogeneity")

    for ax, lab in zip([axA, axB, axC, axD], ["A", "B", "C", "D"]):
        _panel_letter(ax, lab)

    _compact_legend(axB)
    _compact_legend(axD)

    fig.subplots_adjust(left=0.045, right=0.962, bottom=0.17, top=0.90)
    savefig(fig, outpath)



def plot_two_group_main(example_df: pd.DataFrame, summary_df: pd.DataFrame, outpath: Path):
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0), gridspec_kw={"wspace": 0.22})

    q0 = float(example_df.loc[example_df["group"] == "g0", "q_g_oracle"].iloc[0])
    q1 = float(example_df.loc[example_df["group"] == "g1", "q_g_oracle"].iloc[0])
    q = float(example_df["q_pooled_oracle"].iloc[0])

    xs = np.linspace(min(q0, q, q1) - 3, max(q0, q, q1) + 3, 400)

    from .distributions import make_two_group_gaussian

    dists = make_two_group_gaussian(float(example_df["sweep"].iloc[0]))
    group_colors = ["C0", "C1"]
    for d, c in zip(dists, group_colors):
        axes[0].plot(xs, d.cdf(xs), label=d.name, color=c, linewidth=2.8)

    threshold_specs = [
        (q0, r"$q_0$", group_colors[0], (0, (6, 2))),
        (q, r"$q$", "black", (0, (2, 2))),
        (q1, r"$q_1$", group_colors[1], (0, (6, 2))),
    ]
    for xval, lab, color, linestyle in threshold_specs:
        axes[0].axvline(xval, linestyle=linestyle, linewidth=2.4, color=color, alpha=0.95)
        axes[0].text(
            xval + 0.05,
            0.06,
            lab,
            fontsize=13.2,
            color=color,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.88),
        )
    axes[0].set_xlabel("Score threshold")
    axes[0].set_ylabel("CDF")
    axes[0].legend(frameon=False, loc="upper left", handlelength=2.0)

    plot_oracle_curve(axes[1], summary_df, "sweep", "bound_oracle", "Oracle bound")
    plot_empirical_mean_band(
        axes[1],
        summary_df,
        "sweep",
        "distortion_l2",
        "MC empirical",
        show_scatter=True,
        jitter_scale=0.015,
        band="q10q90",
        scatter_alpha=0.09,
        band_alpha=0.16,
        line_width=2.9,
        scatter_size=18,
    )
    axes[1].set_xlabel("Heterogeneity")
    axes[1].set_ylabel("L2 distortion")
    axes[1].legend(frameon=False, loc="upper left", handlelength=2.0)

    for ax in np.ravel(axes):
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        add_grid(ax)

    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.18, top=0.93)
    savefig(fig, outpath)



def plot_multigroup_main(gauss_df: pd.DataFrame, wide_df: pd.DataFrame, gamma_df: pd.DataFrame, t_df: pd.DataFrame, outpath: Path):
    fig, axes = plt.subplots(1, 4, figsize=(18.2, 4.35), sharey=False, gridspec_kw={"wspace": 0.16})

    panels = [
        (gauss_df, "Gaussian family"),
        (wide_df, "Wide Gaussian family"),
        (gamma_df, "Gamma family"),
        (t_df, "Heavy-tail t family"),
    ]

    for i, (ax, (df, title)) in enumerate(zip(axes, panels)):
        plot_oracle_curve(ax, df, "sigma_delta_oracle", "bound_oracle", "Oracle bound")
        plot_empirical_mean_band(
            ax,
            df,
            "sigma_delta_oracle",
            "distortion_l2",
            "MC empirical",
            show_scatter=True,
            jitter_scale=0.004,
            band="q10q90",
            scatter_alpha=0.08,
            scatter_size=16,
            band_alpha=0.15,
            line_width=3.8,
            center_scatter_on_sweep_mean=True,
        )
        ax.set_title(title)
        ax.set_xlabel("Heterogeneity")
        if i == 0:
            ax.set_ylabel("L2 distortion")
        else:
            ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=True)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        if i == 0:
            _compact_legend(ax)
        else:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        add_grid(ax)

    fig.subplots_adjust(left=0.04, right=0.995, bottom=0.18, top=0.90)
    savefig(fig, outpath)



def plot_coverage_to_size_main(summary_df: pd.DataFrame, example_df: pd.DataFrame, outpath: Path):
    fig = plt.figure(figsize=(11.5, 3.95))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.18], wspace=0.18)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    draw_group_size_gaps(ax0, example_df)
    draw_size_disparity(ax1, summary_df)
    _compact_legend(ax1)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.18, top=0.90)
    savefig(fig, outpath)



def plot_size_to_coverage_main(summary_df: pd.DataFrame, example_df: pd.DataFrame, outpath: Path):
    fig = plt.figure(figsize=(11.5, 3.95))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.95, 1.18], wspace=0.18)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    draw_group_coverage_shifts(ax0, example_df)
    draw_coverage_distortion(ax1, summary_df)
    _compact_legend(ax1)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.18, top=0.90)
    savefig(fig, outpath)



def plot_seed_stability(paper_df: pd.DataFrame, outpath: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0), gridspec_kw={"wspace": 0.22})

    metrics = [
        ("multigroup_gaussian", "distortion_l2", "multi-group distortion", "sigma_delta_hat"),
        ("coverage_to_size_hard", "size_disparity", "size disparity", "sigma_delta_hat"),
        ("size_to_coverage_hard", "coverage_distortion", "coverage distortion", "sigma_lambda_hat"),
    ]

    xlabel_map = {
        "sigma_delta_hat": r"$\hat{\sigma}_\Delta$",
        "sigma_lambda_hat": r"$\hat{\sigma}_\Lambda$",
    }

    ylabel_map = {
        "distortion_l2": "L2 distortion",
        "size_disparity": "Size disparity",
        "coverage_distortion": "Coverage distortion",
    }

    for ax, (exp_name, metric, title, xcol) in zip(axes, metrics):
        df = paper_df[paper_df["experiment"] == exp_name].copy()
        plot_empirical_mean_band(
            ax,
            df,
            xcol,
            metric,
            "MC mean ± sd",
            show_scatter=True,
            jitter_scale=0.004,
            band="std",
            scatter_alpha=0.08,
            band_alpha=0.16,
            scatter_size=14,
            line_width=2.7,
        )
        ax.set_title(title)
        ax.set_xlabel(xlabel_map[xcol])
        ax.set_ylabel(ylabel_map[metric])
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        add_grid(ax)

    _compact_legend(axes[0])
    fig.subplots_adjust(left=0.06, right=0.995, bottom=0.18, top=0.90)
    savefig(fig, outpath)
