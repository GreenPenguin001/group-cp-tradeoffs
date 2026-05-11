#!/usr/bin/env python3
"""
Post-process an existing FACET conformal output directory into paper-ready
figures and tables.

Usage:
    python data_plot_process_figures.py /path/to/facet_output_root

Expected inputs (from the FACET notebooks, with flexible fallbacks):
    all_seeds_alpha_summary.csv
    all_seeds_primary_alpha_summary.csv
    all_seeds_primary_alpha_group_metrics.csv
    all_seeds_temperature_sweep_summary.csv                       (optional)
    aggregate_temperature_sweep_summary.csv                       (optional)
    appendix_reporting_exports/summary_temperature_sweep_group*_mean_over_seeds.csv (optional)
    appendix_reporting_exports/all_seeds_temperature_sweep_group*.csv                (optional)
    appendix_reporting_exports/temperature_plot_source_csv/*.csv                     (optional)
    seed_*/tables/temperature_sweep_summary.csv                 (optional)
    seed_*/split_meta.json                      (preferred metadata source)

Design choices:
- Main text emphasizes the primary-point trade-off picture and the empirical
  lower-bound check.
- Appendix summarizes support across alpha and across the controlled
  temperature perturbation.
- The exporter avoids introducing new mathematical notation in titles or table
  headers; existing paper notation is kept only where already standard.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.ticker import MaxNLocator, FixedLocator, FormatStrFormatter
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Global style
# -----------------------------------------------------------------------------
plt.rcParams.update(
    {
        "figure.dpi": 220,
        "savefig.dpi": 300,
        "font.size": 17,
        "axes.titlesize": 22,
        "axes.labelsize": 15,
        "xtick.labelsize": 14.5,
        "ytick.labelsize": 15.0,
        "legend.fontsize": 12.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

PANEL_COLORS = {
    "gap": "#4E79A7",
    "size": "#E15759",
    "cov": "#59A14F",
    "floor": "#4E79A7",
    "bound": "#B07AA1",
}

GRAY_STEM = "0.72"
GRAY_ZERO = "0.35"
GRAY_BAND = "0.975"

FACET_GROUP_ORDER = ["Younger", "Middle", "Older", "Unknown"]
SCORE_DISPLAY = {"simple": "Simple", "saps": "SAPS", "raps": "RAPS"}


def score_display_name(score_name: str) -> str:
    return SCORE_DISPLAY.get(str(score_name).lower(), str(score_name).title())


def facet_group_name_from_index(idx: int) -> str:
    if 0 <= int(idx) < len(FACET_GROUP_ORDER):
        return FACET_GROUP_ORDER[int(idx)]
    return str(idx)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clear_generated_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
    return path


def savefig(
    fig: plt.Figure,
    path_base: Path,
    rect: Optional[tuple[float, float, float, float]] = None,
) -> None:
    if rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=rect)
    fig.savefig(str(path_base.with_suffix(".png")), bbox_inches="tight")
    fig.savefig(str(path_base.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_num_plain(x: float) -> str:
    if pd.isna(x):
        return ""
    ax = abs(float(x))
    if ax != 0 and (ax < 1e-4 or ax >= 1e4):
        return f"{x:.2e}"
    return f"{x:.4f}"


def format_ratio(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.3f}"


def latex_table_from_df(
    df: pd.DataFrame,
    caption: str,
    label: str,
    column_format: Optional[str] = None,
) -> str:
    body = df.to_latex(index=False, escape=False, column_format=column_format)
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{body}"
        "\\end{table}\n"
    )


def candidate_csv(run_root: Path, basename: str, per_seed_relpath: str | None = None) -> Path:
    p = run_root / basename
    if p.exists():
        return p
    if per_seed_relpath is not None:
        candidates = sorted(run_root.glob(per_seed_relpath))
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"Could not find {basename} under {run_root}")


def score_name_from_text(name: str) -> str:
    name = str(name).lower()
    for score in ("simple", "saps", "raps"):
        if score in name:
            return score
    return "unknown"


def find_metadata(run_root: Path) -> dict:
    merged: dict = {}
    sources: list[Path] = []

    score_cfg = run_root / "appendix_reporting_exports" / "score_and_run_config" / "score_config.json"
    if score_cfg.exists():
        sources.append(score_cfg)

    split_meta = sorted(run_root.glob("seed_*/split_meta.json"))
    sources.extend(split_meta[:1])

    exp_report = sorted(run_root.glob("seed_*/experiment_report.json"))
    sources.extend(exp_report[:1])

    for src in sources:
        try:
            data = read_json(src)
            if src.name == "experiment_report.json":
                cfg = dict(data.get("config", {}))
                cfg["run_name"] = str(data.get("run_name", ""))
                data = cfg
            for k, v in data.items():
                if k not in merged or merged[k] in (None, "", [], {}):
                    merged[k] = v
        except Exception:
            pass
    return merged


def read_score_name(run_root: Path) -> str:
    meta = find_metadata(run_root)
    for key in ("conformal_score", "score", "run_name"):
        if key in meta:
            s = score_name_from_text(meta[key])
            if s != "unknown":
                return s
    return score_name_from_text(run_root.name)


def read_dataset_name(run_root: Path) -> str:
    meta = find_metadata(run_root)
    for key in ("experiment_dataset", "dataset_name"):
        if key in meta:
            name = str(meta[key])
            if name.lower() == "facet":
                return "FACET"
            return name
    if "facet" in run_root.name.lower():
        return "FACET"
    return "Dataset"


def read_primary_alpha(run_root: Path, df_primary_summary: pd.DataFrame) -> float:
    meta = find_metadata(run_root)
    if "primary_alpha" in meta and pd.notna(meta["primary_alpha"]):
        return float(meta["primary_alpha"])
    if "alpha" in df_primary_summary.columns and len(df_primary_summary) > 0:
        return float(df_primary_summary["alpha"].iloc[0])
    return 0.10


def read_score_config(run_root: Path, df_primary_summary: pd.DataFrame) -> dict:
    meta = find_metadata(run_root)
    out = {
        "score_temperature": np.nan,
        "raps_lambda": np.nan,
        "raps_k_reg": np.nan,
        "saps_lambda": np.nan,
        "temperature_sweep_group": np.nan,
        "temperature_sweep_group_name": "",
        "temperature_sweep_values": "",
        "seeds": "",
    }
    for key in out:
        if key in meta:
            out[key] = meta[key]
    if pd.isna(out["score_temperature"]) and "chosen_score_temperature" in df_primary_summary.columns:
        s = df_primary_summary["chosen_score_temperature"].dropna()
        if len(s) > 0:
            out["score_temperature"] = float(s.iloc[0])

    if (not out["temperature_sweep_group_name"]) and pd.notna(out["temperature_sweep_group"]):
        try:
            out["temperature_sweep_group_name"] = facet_group_name_from_index(int(out["temperature_sweep_group"]))
        except Exception:
            pass
    return out


def aggregate_numeric(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if c not in set(group_cols) and pd.api.types.is_numeric_dtype(df[c])]
    grouped = df.groupby(group_cols, as_index=False)[numeric_cols].agg(["mean", "std", "count"]).reset_index()
    grouped.columns = [
        col if isinstance(col, str) else (col[0] if col[1] == "" else f"{col[0]}_{col[1]}")
        for col in grouped.columns
    ]
    return grouped


def collapse_primary_group_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if "seed" in df.columns:
        out = aggregate_numeric(df, ["group", "group_name"])
    else:
        rename_map = {c: f"{c}_mean" for c in df.columns if c not in {"group", "group_name"}}
        out = df.rename(columns=rename_map)

    if "group" in out.columns:
        out = out.sort_values("group").reset_index(drop=True)
    if "group_name" in out.columns:
        out["group_name"] = out.apply(
            lambda r: facet_group_name_from_index(r["group"]) if pd.isna(r.get("group_name")) or str(r.get("group_name", "")).strip() == "" else r["group_name"],
            axis=1,
        )
    return out


def summarize_primary_seed_table(df: pd.DataFrame) -> pd.Series:
    if "seed" in df.columns and len(df) > 1:
        s = df.mean(numeric_only=True)
        if "alpha" in df.columns:
            s["alpha"] = df["alpha"].iloc[0]
        return s
    return df.iloc[0]


def add_row_bands(ax: plt.Axes, n_rows: int) -> None:
    for i in range(n_rows):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=GRAY_BAND, zorder=0)


def symmetric_limit(values: np.ndarray, errs: np.ndarray, pad: float = 0.12) -> float:
    lim = float(np.max(np.abs(values) + np.abs(errs)))
    if lim <= 0:
        lim = 1.0
    return lim * (1.0 + pad)


def lighten_color(color: str, factor: float = 0.55) -> str:
    rgb = np.array(mcolors.to_rgb(color))
    mixed = rgb + (1.0 - rgb) * float(factor)
    return mcolors.to_hex(np.clip(mixed, 0.0, 1.0))


def symmetric_three_ticks(lim: float) -> list[float]:
    return [-float(lim), 0.0, float(lim)]


def get_series(df: pd.DataFrame, mean_col: str, std_col: str) -> tuple[np.ndarray, np.ndarray]:
    vals = df[mean_col].to_numpy(dtype=float)
    if std_col in df.columns:
        errs = np.nan_to_num(df[std_col].to_numpy(dtype=float), nan=0.0)
    else:
        errs = np.zeros_like(vals)
    return vals, errs


def ratio_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    denom_mean = out.get("m_eff_segment_times_sigma_delta_mean", pd.Series(np.nan, index=out.index))
    numer_mean = out.get("rms_cov_pooled_mean", pd.Series(np.nan, index=out.index))
    out["ratio_floor_mean"] = numer_mean / denom_mean.replace(0.0, np.nan)
    if "count" not in out.columns and "rms_cov_pooled_count" in out.columns:
        out["count"] = out["rms_cov_pooled_count"]
    return out


def load_alpha_summary(run_root: Path) -> pd.DataFrame:
    p = candidate_csv(run_root, "all_seeds_alpha_summary.csv", "seed_*/tables/alpha_summary.csv")
    return pd.read_csv(p)


def load_primary_summary(run_root: Path) -> pd.DataFrame:
    p = candidate_csv(run_root, "all_seeds_primary_alpha_summary.csv", "seed_*/tables/primary_alpha_summary.csv")
    return pd.read_csv(p)


def load_primary_group_metrics(run_root: Path) -> pd.DataFrame:
    p = candidate_csv(run_root, "all_seeds_primary_alpha_group_metrics.csv", "seed_*/tables/primary_alpha_group_metrics.csv")
    return pd.read_csv(p)


def standardize_temperature_columns(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()

    rename_map = {}
    aliases = {
        "temperature": "temperature_value",
        "temp": "temperature_value",
        "temp_value": "temperature_value",
        "sweep_temperature": "temperature_value",
        "group": "temperature_group",
        "group_idx": "temperature_group",
        "temperature_group_idx": "temperature_group",
    }
    for src, dst in aliases.items():
        if src in out.columns and dst not in out.columns:
            rename_map[src] = dst
    if rename_map:
        out = out.rename(columns=rename_map)

    if "temperature_value" not in out.columns:
        return pd.DataFrame()

    if "temperature_group" not in out.columns:
        tg = metadata.get("temperature_sweep_group", np.nan)
        out["temperature_group"] = tg

    if "temperature_group_name" not in out.columns:
        out["temperature_group_name"] = ""

    mask_blank = out["temperature_group_name"].astype(str).str.strip().isin(["", "nan", "None"])
    if mask_blank.any() and "temperature_group" in out.columns:
        out.loc[mask_blank, "temperature_group_name"] = out.loc[mask_blank, "temperature_group"].apply(
            lambda x: facet_group_name_from_index(int(x)) if pd.notna(x) else ""
        )

    # Repair notebook leftovers such as gender-based labels.
    allowed_names = set(FACET_GROUP_ORDER)
    bad_mask = ~out["temperature_group_name"].astype(str).isin(allowed_names)
    if bad_mask.any() and "temperature_group" in out.columns:
        out.loc[bad_mask, "temperature_group_name"] = out.loc[bad_mask, "temperature_group"].apply(
            lambda x: facet_group_name_from_index(int(x)) if pd.notna(x) else ""
        )

    return out


def is_preaggregated_temperature_summary(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    expected = {
        "sigma_delta_mean",
        "m_eff_segment_times_sigma_delta_mean",
        "rms_cov_pooled_mean",
        "rms_size_from_groupwise_mean",
        "rms_cov_from_equalized_size_mean",
    }
    return len(expected.intersection(df.columns)) >= 3


def normalize_preaggregated_temperature_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename_map = {}
    for base in [
        "sigma_delta",
        "m_eff_segment_times_sigma_delta",
        "rms_cov_pooled",
        "rms_size_from_groupwise",
        "rms_cov_from_equalized_size",
    ]:
        if base in out.columns and f"{base}_mean" not in out.columns:
            rename_map[base] = f"{base}_mean"
    if rename_map:
        out = out.rename(columns=rename_map)
    return out


def _read_temperature_csv_candidate(path: Path, metadata: dict) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = standardize_temperature_columns(df, metadata)
    if df.empty:
        return pd.DataFrame()
    if is_preaggregated_temperature_summary(df):
        df = normalize_preaggregated_temperature_summary(df)
    return df


def load_temperature_summary_optional(run_root: Path, metadata: dict) -> pd.DataFrame:
    direct_candidates = [
        run_root / "all_seeds_temperature_sweep_summary.csv",
        run_root / "aggregate_temperature_sweep_summary.csv",
        run_root / "appendix_reporting_exports" / "tables" / "temperature_sweep_summary.csv",
    ]
    for p in direct_candidates:
        if p.exists():
            try:
                df = _read_temperature_csv_candidate(p, metadata)
                if not df.empty:
                    return df
            except Exception:
                pass

    appendix_glob_patterns = [
        "appendix_reporting_exports/summary_temperature_sweep_group*_mean_over_seeds.csv",
        "appendix_reporting_exports/all_seeds_temperature_sweep_group*.csv",
        "appendix_reporting_exports/temperature_plot_source_csv/*temperature*.csv",
    ]
    appendix_candidates = []
    for pattern in appendix_glob_patterns:
        appendix_candidates.extend(sorted(run_root.glob(pattern)))

    for p in appendix_candidates:
        try:
            df = _read_temperature_csv_candidate(p, metadata)
            if not df.empty:
                return df
        except Exception:
            pass

    per_seed_patterns = [
        "seed_*/tables/temperature_sweep_summary.csv",
        "seed_*/temperature_sweep_summary.csv",
        "seed_*/appendix_reporting_exports/tables/temperature_sweep_summary.csv",
        "seed_*/temperature sweep/*.csv",
    ]
    dfs = []
    preagg_dfs = []
    for pattern in per_seed_patterns:
        for c in sorted(run_root.glob(pattern)):
            try:
                df = _read_temperature_csv_candidate(c, metadata)
                if df.empty:
                    continue
                if is_preaggregated_temperature_summary(df):
                    preagg_dfs.append(df)
                    continue
                if "seed" not in df.columns:
                    try:
                        seed_name = next((part for part in c.parts if part.startswith("seed_")), "")
                        if seed_name.startswith("seed_"):
                            df["seed"] = int(seed_name.split("seed_")[-1])
                    except Exception:
                        pass
                dfs.append(df)
            except Exception:
                pass

    if preagg_dfs:
        return pd.concat(preagg_dfs, ignore_index=True)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()



# -----------------------------------------------------------------------------
# Figure builders
# -----------------------------------------------------------------------------
def build_primary_overview_figure(
    df_group: pd.DataFrame,
    out_base: Path,
    dataset_name: str,
    score_name: str,
    alpha: float,
    group_axis_label: str = "",
) -> None:
    order = df_group.sort_values("group").copy().reset_index(drop=True)
    labels = order["group_name"].tolist()
    y = np.arange(len(order))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(6.9, 3.0),
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0], "wspace": 0.26},
    )

    panel_specs = [
        {
            "ax": axes[0],
            "letter": "A",
            "title": "Pooled threshold",
            "xlabel": "Coverage distortion",
            "mean_col": "epsilon_pooled_mean",
            "std_col": "epsilon_pooled_std",
            "color": PANEL_COLORS["gap"],
        },
        {
            "ax": axes[1],
            "letter": "B",
            "title": "Equalized coverage",
            "xlabel": "Set-size distortion",
            "mean_col": "delta_size_from_groupwise_mean",
            "std_col": "delta_size_from_groupwise_std",
            "color": PANEL_COLORS["size"],
        },
        {
            "ax": axes[2],
            "letter": "C",
            "title": "Equalized size",
            "xlabel": "Coverage distortion",
            "mean_col": "delta_cov_from_equalized_size_mean",
            "std_col": "delta_cov_from_equalized_size_std",
            "color": PANEL_COLORS["cov"],
        },
    ]

    for spec in panel_specs:
        ax = spec["ax"]
        add_row_bands(ax, len(order))
        vals, errs = get_series(order, spec["mean_col"], spec["std_col"])
        lim = symmetric_limit(vals, errs, pad=0.04)

        ax.axvline(0.0, color=GRAY_ZERO, linestyle="--", linewidth=1.0, zorder=1)
        ax.grid(axis="x", color="0.88", linewidth=0.6)

        pos_color = spec["color"]
        neg_color = lighten_color(spec["color"], factor=0.62)
        facecolors = [pos_color if v >= 0 else neg_color for v in vals]

        ax.barh(
            y,
            vals,
            height=0.56,
            color=facecolors,
            edgecolor=spec["color"],
            linewidth=1.25,
            zorder=3,
        )

        if np.any(errs > 0):
            ax.errorbar(
                vals,
                y,
                xerr=errs,
                fmt="none",
                ecolor="0.25",
                elinewidth=0.9,
                capsize=2.2,
                zorder=4,
            )

        ax.set_xlim(-lim, lim)
        ax.xaxis.set_major_locator(FixedLocator(symmetric_three_ticks(lim)))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        panel_label_fs = plt.rcParams["axes.labelsize"] - 1
        ax.set_xlabel(spec["xlabel"], fontsize=panel_label_fs - 1.0)
        if spec["letter"] in {"A", "B"}:
            ax.xaxis.set_label_coords(0.50, -0.25)
        if spec["letter"] == "C":
            # Keep Panel C on the same baseline while preserving a slight left shift.
            ax.xaxis.set_label_coords(0.40, -0.25)
        ax.set_title(spec["title"], pad=5, fontsize=panel_label_fs)
        ax.text(
            0.01,
            1.10,
            spec["letter"],
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=15,
            fontweight="bold",
        )
        ax.tick_params(axis="y", length=0)
        panel_tick_fs = plt.rcParams["xtick.labelsize"] - 2.0
        if spec["letter"] == "C":
            ax.tick_params(axis="x", rotation=45, labelsize=panel_tick_fs, pad=1.0)
        else:
            ax.tick_params(axis="x", rotation=45, labelsize=panel_tick_fs)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    if str(group_axis_label).strip():
        axes[0].set_ylabel(group_axis_label)
    else:
        axes[0].set_ylabel("")
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False, labelleft=False)
    axes[0].invert_yaxis()
    savefig(fig, out_base, rect=(0.0, 0.0, 0.94, 1.0))


def build_primary_floor_figure(
    primary_row: pd.Series,
    out_base: Path,
    dataset_name: str,
    score_name: str,
    alpha: float,
) -> None:
    lhs = float(primary_row["rms_cov_pooled"])
    rhs = float(primary_row["m_eff_segment_times_sigma_delta"])
    ratio = lhs / rhs if abs(rhs) > 1e-12 else np.nan
    score_disp = score_display_name(score_name)

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    labels = ["Lower-bound proxy", "Empirical pooled RMS gap"]
    vals = [rhs, lhs]
    colors = [PANEL_COLORS["bound"], PANEL_COLORS["floor"]]
    xpos = np.arange(2)

    ax.bar(xpos, vals, width=0.58, color=colors)
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Magnitude")
    ax.set_title("Empirical lower-bound check at the primary point", pad=8)
    ax.grid(axis="y", color="0.90", linewidth=0.6)
    ymax = max(vals) * 1.28 if max(vals) > 0 else 1.0
    ax.set_ylim(0.0, ymax)
    for x, y in zip(xpos, vals):
        ax.text(x, y + 0.02 * ymax, format_num_plain(y), ha="center", va="bottom", fontsize=10.8)
    ax.text(
        0.5,
        0.96,
        f"ratio = {format_ratio(ratio)}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.985,
        rf"{dataset_name}, {score_disp} score, $\alpha={alpha:.2f}$",
        ha="center",
        va="bottom",
        fontsize=11.4,
    )
    savefig(fig, out_base, rect=(0.0, 0.0, 1.0, 0.96))


def build_support_magnitude_bar_figure(
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    title: str,
    out_base: Path,
) -> None:
    x_values = df[x_col].tolist()
    x_labels = [f"{x:.3f}".rstrip("0").rstrip(".") if x_col == "alpha" else f"{x:.2f}" for x in x_values]
    xpos = np.arange(len(x_values))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(16.2, 3.4),
        gridspec_kw={"width_ratios": [1.05, 0.97, 0.97], "wspace": 0.50},
    )

    # Panel A: floor ratio.
    ax = axes[0]
    ratio = df["ratio_floor_mean"].to_numpy(float)
    ax.grid(axis="y", color="0.90", linewidth=0.6)
    ax.axhline(1.0, color=GRAY_ZERO, linestyle="--", linewidth=1.0)
    ax.bar(xpos, ratio, width=0.52, color=PANEL_COLORS["floor"])
    for x, yv in zip(xpos, ratio):
        ax.text(x, yv + 0.012, format_ratio(yv), ha="center", va="bottom", fontsize=11.8)
    ymin = min(0.97, float(np.nanmin(ratio)) - 0.03) if np.isfinite(ratio).any() else 0.97
    ymax = max(1.06, float(np.nanmax(ratio)) + 0.07) if np.isfinite(ratio).any() else 1.06
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(xpos)
    panel_label_fs = plt.rcParams["axes.labelsize"] - 1
    ax.set_xticklabels(x_labels)
    ax.tick_params(axis="x", labelsize=12, rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.set_xlabel(x_label, fontsize=panel_label_fs)
    ax.set_ylabel("Empirical / lower-bound proxy", labelpad=4)
    ax.set_title("Lower-bound ratio", pad=6, fontsize=panel_label_fs)
    ax.text(-0.10, 1.07, "A", transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")

    # Panel B: size distortion.
    ax = axes[1]
    vals = df["rms_size_from_groupwise_mean"].to_numpy(float)
    ax.grid(axis="y", color="0.90", linewidth=0.6)
    ax.bar(xpos, vals, width=0.52, color=PANEL_COLORS["size"])
    ax.set_xticks(xpos)
    ax.set_xticklabels(x_labels)
    ax.tick_params(axis="x", labelsize=12, rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.set_xlabel(x_label, fontsize=panel_label_fs)
    ax.set_ylabel("RMS change in expected set size", labelpad=2)
    ax.set_title("Set-size gap", pad=6, fontsize=panel_label_fs)
    ax.text(-0.10, 1.07, "B", transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")

    # Panel C: coverage distortion.
    ax = axes[2]
    vals = df["rms_cov_from_equalized_size_mean"].to_numpy(float)
    ax.grid(axis="y", color="0.90", linewidth=0.6)
    ax.bar(xpos, vals, width=0.52, color=PANEL_COLORS["cov"])
    ax.set_xticks(xpos)
    ax.set_xticklabels(x_labels)
    ax.tick_params(axis="x", labelsize=12, rotation=45)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.set_xlabel(x_label, fontsize=panel_label_fs)
    ax.set_ylabel("RMS induced coverage gap", labelpad=2)
    ax.set_title("Coverage distortion", pad=6, fontsize=panel_label_fs)
    ax.text(-0.10, 1.07, "C", transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")

    savefig(fig, out_base, rect=(0.0, 0.02, 1.0, 0.94))


# -----------------------------------------------------------------------------
# Table builders
# -----------------------------------------------------------------------------
def build_primary_summary_table(
    primary_row: pd.Series,
    out_dir: Path,
    dataset_name: str,
    score_name: str,
    alpha: float,
) -> None:
    rhs = float(primary_row["m_eff_segment_times_sigma_delta"])
    lhs = float(primary_row["rms_cov_pooled"])
    ratio = lhs / rhs if abs(rhs) > 1e-12 else np.nan

    pretty = pd.DataFrame(
        [
            {
                r"$q$": format_num_plain(primary_row["q_pooled"]),
                r"$\sigma_\Delta$": format_num_plain(primary_row["sigma_delta"]),
                "Lower-bound proxy": format_num_plain(rhs),
                "Pooled RMS gap": format_num_plain(lhs),
                "Ratio": format_ratio(ratio),
                r"$\lambda$": format_num_plain(primary_row["lambda_target"]),
                "RMS set-size gap": format_num_plain(primary_row["rms_size_from_groupwise"]),
                r"$\sigma_\lambda$": format_num_plain(primary_row["sigma_lambda"]),
                "RMS coverage gap": format_num_plain(primary_row["rms_cov_from_equalized_size"]),
            }
        ]
    )
    pretty.to_csv(out_dir / "table_primary_summary.csv", index=False)
    tex = latex_table_from_df(
        pretty,
        caption=(
            rf"Primary-point summary for {dataset_name} ({score_display_name(score_name)} score, $\alpha={alpha:.2f}$). "
            "The lower-bound proxy reports the empirical Section~3 reference already exported by the notebook. "
            "The remaining columns summarize the pooled RMS coverage gap, the set-size gap after group-wise thresholds, "
            "and the induced coverage gap after expected-size equalization."
        ),
        label="tab:facet-primary-summary",
        column_format="ccccccccc",
    )
    (out_dir / "table_primary_summary.tex").write_text(tex, encoding="utf-8")


def build_primary_group_full_table(
    df_group: pd.DataFrame,
    out_dir: Path,
    dataset_name: str,
    score_name: str,
) -> None:
    order = df_group.sort_values("group").copy()
    pretty = pd.DataFrame(
        {
            "Age group": order["group_name"],
            r"$p_g$": order["p_g_mean"].map(format_num_plain),
            r"$q_g$": order["q_g_mean"].map(format_num_plain),
            r"$\widehat\varepsilon_g(q)$": order["epsilon_pooled_mean"].map(format_num_plain),
            r"$\lambda_g$": order["lambda_g_mean"].map(format_num_plain),
            r"$\lambda_g-\ell_g(q)$": order["delta_size_from_groupwise_mean"].map(format_num_plain),
            r"$\tau_g$": order["tau_g_mean"].map(format_num_plain),
            r"$\widehat F_g(\tau_g)-\widehat F_g(q_g)$": order["delta_cov_from_equalized_size_mean"].map(format_num_plain),
        }
    )
    pretty.to_csv(out_dir / "table_primary_group_full.csv", index=False)
    tex = latex_table_from_df(
        pretty,
        caption=f"Per-group summary for {dataset_name} at the primary operating point ({score_display_name(score_name)} score).",
        label="tab:facet-primary-groups-full",
        column_format="l" + "c" * (len(pretty.columns) - 1),
    )
    (out_dir / "table_primary_group_full.tex").write_text(tex, encoding="utf-8")


def build_support_table(
    df: pd.DataFrame,
    x_col: str,
    x_label_tex: str,
    out_csv: Path,
    out_tex: Path,
    caption: str,
    label: str,
) -> None:
    pretty = pd.DataFrame(
        {
            x_label_tex: df[x_col].map(format_num_plain),
            r"$\sigma_\Delta$": df["sigma_delta_mean"].map(format_num_plain),
            "Lower-bound proxy": df["m_eff_segment_times_sigma_delta_mean"].map(format_num_plain),
            "Pooled RMS gap": df["rms_cov_pooled_mean"].map(format_num_plain),
            "Ratio": df["ratio_floor_mean"].map(format_ratio),
            "RMS set-size gap": df["rms_size_from_groupwise_mean"].map(format_num_plain),
            "RMS coverage gap": df["rms_cov_from_equalized_size_mean"].map(format_num_plain),
        }
    )
    pretty.to_csv(out_csv, index=False)
    tex = latex_table_from_df(
        pretty,
        caption=caption,
        label=label,
        column_format="l" + "c" * (len(pretty.columns) - 1),
    )
    out_tex.write_text(tex, encoding="utf-8")


def build_score_config_table(
    score_config: dict,
    out_dir: Path,
    dataset_name: str,
    score_name: str,
    alpha: float,
) -> None:
    rows = [
        ("Dataset", dataset_name),
        ("Score", score_display_name(score_name)),
        (r"Primary $\alpha$", format_num_plain(alpha)),
        ("Score temperature", format_num_plain(score_config.get("score_temperature", np.nan))),
    ]
    if score_name == "raps":
        rows.append((r"RAPS $\lambda$", format_num_plain(score_config.get("raps_lambda", np.nan))))
        rows.append((r"RAPS $k_{\mathrm{reg}}$", format_num_plain(score_config.get("raps_k_reg", np.nan))))
    if score_name == "saps":
        rows.append((r"SAPS $\lambda$", format_num_plain(score_config.get("saps_lambda", np.nan))))
    tg_name = score_config.get("temperature_sweep_group_name", "")
    if tg_name:
        rows.append(("Temperature-perturbed group", str(tg_name)))
    tg_vals = score_config.get("temperature_sweep_values", "")
    if tg_vals:
        rows.append(("Temperature values", str(tg_vals)))
    seeds = score_config.get("seeds", "")
    if seeds:
        rows.append(("Seeds", str(seeds)))

    pretty = pd.DataFrame(rows, columns=["Setting", "Value"])
    pretty.to_csv(out_dir / "table_run_config.csv", index=False)
    tex = latex_table_from_df(
        pretty,
        caption=f"Run configuration summary for the {dataset_name} paper-ready export.",
        label="tab:facet-run-config",
        column_format="ll",
    )
    (out_dir / "table_run_config.tex").write_text(tex, encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready FACET figures/tables from an existing output directory.")
    parser.add_argument("run_root", type=str, help="Existing FACET output directory produced by the notebook.")
    args = parser.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    if not run_root.exists():
        raise FileNotFoundError(f"Run root does not exist: {run_root}")

    paper_root = ensure_dir(run_root / "paper_ready")
    main_fig_dir = clear_generated_dir(paper_root / "main_text" / "figures")
    main_tab_dir = clear_generated_dir(paper_root / "main_text" / "tables")
    app_fig_dir = clear_generated_dir(paper_root / "appendix" / "figures")
    app_tab_dir = clear_generated_dir(paper_root / "appendix" / "tables")

    metadata = find_metadata(run_root)
    dataset_name = read_dataset_name(run_root)
    score_name = read_score_name(run_root)

    df_alpha_raw = load_alpha_summary(run_root)
    df_primary_summary_raw = load_primary_summary(run_root)
    df_primary_groups_raw = load_primary_group_metrics(run_root)
    df_temp_raw = load_temperature_summary_optional(run_root, metadata)

    primary_alpha = read_primary_alpha(run_root, df_primary_summary_raw)
    primary_row = summarize_primary_seed_table(df_primary_summary_raw)
    df_primary_groups = collapse_primary_group_metrics(df_primary_groups_raw)

    if "alpha" in df_alpha_raw.columns:
        df_alpha = ratio_columns(aggregate_numeric(df_alpha_raw, ["alpha"]).sort_values("alpha").reset_index(drop=True))
    else:
        df_alpha = pd.DataFrame()

    if not df_temp_raw.empty and "temperature_value" in df_temp_raw.columns:
        df_temp_use = df_temp_raw.copy()
        target_group = metadata.get("temperature_sweep_group", np.nan)
        if pd.notna(target_group) and "temperature_group" in df_temp_use.columns:
            mask = pd.to_numeric(df_temp_use["temperature_group"], errors="coerce") == float(target_group)
            if mask.any():
                df_temp_use = df_temp_use.loc[mask].copy()

        if is_preaggregated_temperature_summary(df_temp_use):
            df_temp = ratio_columns(normalize_preaggregated_temperature_summary(df_temp_use).sort_values("temperature_value").reset_index(drop=True))
        else:
            group_cols = ["temperature_value"]
            if "temperature_group" in df_temp_use.columns:
                group_cols = ["temperature_group", "temperature_value"]
            if "temperature_group_name" in df_temp_use.columns:
                group_cols.insert(1 if "temperature_group" in group_cols else 0, "temperature_group_name")
            df_temp = ratio_columns(aggregate_numeric(df_temp_use, group_cols).sort_values("temperature_value").reset_index(drop=True))
    else:
        df_temp = pd.DataFrame()

    score_config = read_score_config(run_root, df_primary_summary_raw)

    # Main text outputs
    build_primary_overview_figure(
        df_primary_groups,
        main_fig_dir / "figure_primary_overview",
        dataset_name=dataset_name,
        score_name=score_name,
        alpha=primary_alpha,
        group_axis_label="",
    )
    build_primary_floor_figure(
        primary_row,
        main_fig_dir / "figure_primary_floor_check",
        dataset_name=dataset_name,
        score_name=score_name,
        alpha=primary_alpha,
    )
    build_primary_summary_table(
        primary_row,
        main_tab_dir,
        dataset_name=dataset_name,
        score_name=score_name,
        alpha=primary_alpha,
    )

    # Appendix outputs
    if not df_alpha.empty:
        build_support_magnitude_bar_figure(
            df_alpha,
            x_col="alpha",
            x_label=r"$\alpha$",
            title=rf"{dataset_name}, {score_display_name(score_name)} score: support across $\alpha$",
            out_base=app_fig_dir / "figure_alpha_support_summary",
        )
        build_support_table(
            df_alpha,
            x_col="alpha",
            x_label_tex=r"$\alpha$",
            out_csv=app_tab_dir / "table_alpha_support.csv",
            out_tex=app_tab_dir / "table_alpha_support.tex",
            caption=(
                rf"Support across $\alpha$ for {dataset_name} ({score_display_name(score_name)} score). "
                "The empirical lower-bound proxy is compared directly against the pooled RMS coverage gap, "
                "and the corresponding set-size and expected-size-equalized coverage gaps are reported alongside it."
            ),
            label="tab:facet-alpha-support",
        )

    if not df_temp.empty:
        temp_group_title = ""
        if "temperature_group_name" in df_temp.columns and len(df_temp) > 0:
            temp_group_title = str(df_temp["temperature_group_name"].iloc[0])
        temp_title_suffix = f" ({temp_group_title})" if temp_group_title else ""
        build_support_magnitude_bar_figure(
            df_temp,
            x_col="temperature_value",
            x_label="Temperature",
            title=f"{dataset_name}, {score_display_name(score_name)} score: controlled temperature perturbation{temp_title_suffix}",
            out_base=app_fig_dir / "figure_temperature_support_summary",
        )
        build_support_table(
            df_temp,
            x_col="temperature_value",
            x_label_tex="Temperature",
            out_csv=app_tab_dir / "table_temperature_support.csv",
            out_tex=app_tab_dir / "table_temperature_support.tex",
            caption=(
                f"Controlled temperature-perturbation summary for {dataset_name} ({score_display_name(score_name)} score). "
                "Only the designated age group is temperature-perturbed; the table reports how the lower-bound proxy, "
                "the pooled RMS coverage gap, and the induced set-size and coverage gaps move with the perturbation."
            ),
            label="tab:facet-temperature-support",
        )

    build_primary_group_full_table(df_primary_groups, app_tab_dir, dataset_name=dataset_name, score_name=score_name)
    build_score_config_table(score_config, app_tab_dir, dataset_name=dataset_name, score_name=score_name, alpha=primary_alpha)

    manifest = {
        "run_root": str(run_root),
        "dataset": dataset_name,
        "score": score_name,
        "primary_alpha": float(primary_alpha),
        "main_text": {
            "figures": sorted(p.name for p in main_fig_dir.iterdir()),
            "tables": sorted(p.name for p in main_tab_dir.iterdir()),
            "suggested_usage": [
                "Use figure_primary_overview as the main Section 4.4 trade-off figure.",
                "Use figure_primary_floor_check if you want a separate visual for the lower-bound check.",
                "Use table_primary_summary as the compact main-text table.",
            ],
        },
        "appendix": {
            "figures": sorted(p.name for p in app_fig_dir.iterdir()),
            "tables": sorted(p.name for p in app_tab_dir.iterdir()),
            "notes": [
                "Panel A reports the ratio between the empirical pooled RMS gap and the notebook's lower-bound proxy.",
                "Panel B reports the RMS set-size gap after replacing the pooled threshold by group-wise thresholds.",
                "Panel C reports the RMS coverage gap after expected-size equalization.",
                "For FACET, the preferred lower-bound reference is the notebook-exported empirical proxy.",
            ],
        },
    }
    (paper_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[done] wrote paper-ready outputs to: {paper_root}")


if __name__ == "__main__":
    main()
