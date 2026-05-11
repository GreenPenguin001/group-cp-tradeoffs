#!/usr/bin/env python3
"""
Post-process an existing MultiNLI conformal output directory into
figures and tables, with a verification strategy aligned to the strengths of
real-data MultiNLI evidence.

Usage:
    python data_plot_process_figures.py /path/to/output_root

Design principles:
- Keep a single main-text figure (per-group signed trade-off overview).
- For Section 3, do NOT force every relation into the same lower-bound ratio
  template. On MultiNLI, the pooled-threshold floor is the most stable
  quantitative check; the equalized-coverage / equalized-size relations are
  better validated via sign-consistency and induced-disparity evidence.
- Replace appendix line plots with grouped bar plots because alpha and
  temperature are discrete design choices.
"""

from __future__ import annotations

import argparse
import json
import math
import re
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
        "font.size": 18,
        "axes.titlesize": 23,
        "axes.labelsize": 18,
        "xtick.labelsize": 15.0,
        "ytick.labelsize": 18.0,
        "legend.fontsize": 13.0,
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
    "positive": "#4E79A7",
    "negative": "#C44E52",
    "floor": "#4E79A7",
    "thm6": "#E15759",
    "thm7": "#59A14F",
}

GRAY_STEM = "0.72"
GRAY_ZERO = "0.35"
GRAY_BAND = "0.975"

DEFAULT_RAPS_LAMBDA = 0.2
DEFAULT_RAPS_K_REG = 1
DEFAULT_SAPS_LAMBDA = 0.2
DEFAULT_SCORE_TEMPERATURE = 1.0
DEFAULT_RANDOM_U = 0.5
DEFAULT_GRID_STEP = 0.001
DEFAULT_TEMP_EPS = 1e-12

# Sign-check tolerances. These are not used to force success; they only exclude
# effectively-zero, non-informative groups from a sign comparison.
SIGN_TOL_Q = 5e-4
SIGN_TOL_SIZE = 5e-3
SIGN_TOL_COV = 5e-3
SIGN_TOL_EPS = 5e-3


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


def savefig(fig: plt.Figure, path_base: Path, rect: Optional[tuple[float, float, float, float]] = None) -> None:
    if rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=rect)
    fig.savefig(str(path_base.with_suffix(".png")), bbox_inches="tight")
    fig.savefig(str(path_base.with_suffix(".pdf")), bbox_inches="tight")
    plt.close(fig)


def format_num_plain(x: float) -> str:
    if pd.isna(x):
        return ""
    ax = abs(float(x))
    if ax != 0 and (ax < 1e-4 or ax >= 1e4):
        return f"{x:.2e}"
    return f"{x:.4f}"


def format_slack(x: float) -> str:
    if pd.isna(x):
        return ""
    ax = abs(float(x))
    if ax != 0 and ax < 1e-4:
        return f"{x:.2e}"
    return f"{x:+.4f}"

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


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def score_name_from_run_name(name: str) -> str:
    name = name.lower()
    for score in ("simple", "saps", "raps"):
        if score in name:
            return score
    return "unknown"


def read_score_name(run_root: Path) -> str:
    report_candidates = sorted(run_root.glob("seed_*/experiment_report.json"))
    for p in report_candidates:
        try:
            report = read_json(p)
            run_name = str(report.get("run_name", ""))
            score = score_name_from_run_name(run_name)
            if score != "unknown":
                return score
        except Exception:
            pass
    return score_name_from_run_name(run_root.name)


def read_run_config(run_root: Path) -> dict:
    report_candidates = sorted(run_root.glob("seed_*/experiment_report.json"))
    if not report_candidates:
        return {}
    report = read_json(report_candidates[0])
    return dict(report.get("config", {}))


def candidate_primary_summary(run_root: Path) -> Path:
    p = run_root / "all_seeds_primary_alpha_summary.csv"
    if p.exists():
        return p
    candidates = sorted(run_root.glob("seed_*/tables/primary_alpha_summary.csv"))
    if not candidates:
        raise FileNotFoundError("Could not find primary alpha summary CSV")
    return candidates[0]


def candidate_primary_group_metrics(run_root: Path) -> Path:
    p = run_root / "all_seeds_primary_alpha_group_metrics.csv"
    if p.exists():
        return p
    candidates = sorted(run_root.glob("seed_*/tables/primary_alpha_group_metrics.csv"))
    if not candidates:
        raise FileNotFoundError("Could not find primary-alpha group metrics CSV")
    return candidates[0]


def collapse_primary_group_metrics(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        c for c in df.columns if c not in {"group", "group_name", "seed"} and pd.api.types.is_numeric_dtype(df[c])
    ]
    grouped = df.groupby(["group", "group_name"], as_index=False)[numeric_cols].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        col if isinstance(col, str) else (col[0] if col[1] == "" else f"{col[0]}_{col[1]}")
        for col in grouped.columns
    ]
    return grouped.sort_values("group").reset_index(drop=True)


def get_series(df: pd.DataFrame, mean_col: str, std_col: str) -> tuple[np.ndarray, np.ndarray]:
    vals = df[mean_col].to_numpy(dtype=float)
    if std_col in df.columns:
        errs = np.nan_to_num(df[std_col].to_numpy(dtype=float), nan=0.0)
    else:
        errs = np.zeros_like(vals)
    return vals, errs


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


def _safe_ratio(num: float, den: float) -> float:
    if abs(den) < 1e-12:
        return np.nan
    return float(num / den)


def _safe_div(num: float, den: float) -> float:
    if abs(den) < 1e-12:
        return 0.0
    return float(num / den)


def sign_with_tol(x: float, tol: float) -> int:
    if x > tol:
        return 1
    if x < -tol:
        return -1
    return 0


def sign_match_summary(expected_by_group: dict, observed_by_group: dict, tol_expected: float, tol_observed: float) -> dict:
    match = 0
    informative = 0
    mismatches = []
    for g in sorted(expected_by_group):
        s_exp = sign_with_tol(expected_by_group[g], tol_expected)
        s_obs = sign_with_tol(observed_by_group[g], tol_observed)
        if s_exp == 0 or s_obs == 0:
            continue
        informative += 1
        if s_exp == s_obs:
            match += 1
        else:
            mismatches.append(int(g))
    frac = float(match / informative) if informative > 0 else np.nan
    return {
        "match": int(match),
        "informative": int(informative),
        "frac": frac,
        "mismatched_groups": mismatches,
    }


# -----------------------------------------------------------------------------
# Score + conformal helpers
# -----------------------------------------------------------------------------
def _apply_temperature_global(probs: np.ndarray, T: float, eps: float = DEFAULT_TEMP_EPS) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    safe = np.clip(probs, eps, 1.0)
    if abs(T - 1.0) < 1e-12:
        return safe / safe.sum(axis=1, keepdims=True)
    powered = safe ** (1.0 / float(T))
    return powered / powered.sum(axis=1, keepdims=True)


def apply_group_temperature(probs: np.ndarray, groups: np.ndarray, temperature_map: dict, eps: float = DEFAULT_TEMP_EPS) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    groups = np.asarray(groups)
    adjusted = np.empty_like(probs)
    for g in np.unique(groups):
        mask = groups == g
        T = float(temperature_map[int(g)])
        adjusted[mask] = _apply_temperature_global(probs[mask], T, eps=eps)
    return adjusted


def all_label_scores(
    probs: np.ndarray,
    method: str,
    score_temperature: float = DEFAULT_SCORE_TEMPERATURE,
    raps_lambda: float = DEFAULT_RAPS_LAMBDA,
    raps_k_reg: int = DEFAULT_RAPS_K_REG,
    saps_lambda: float = DEFAULT_SAPS_LAMBDA,
    random_u: float = DEFAULT_RANDOM_U,
) -> np.ndarray:
    method = (method or "simple").strip().lower()
    probs = _apply_temperature_global(probs, float(score_temperature))
    probs = np.asarray(probs, dtype=float)

    if method in ("simple", "1-p", "1p"):
        return 1.0 - probs

    n, K = probs.shape
    u = np.full((n, 1), float(random_u), dtype=float)

    order = np.argsort(-probs, axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(n)[:, None], order] = np.arange(1, K + 1)

    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumsum = np.cumsum(sorted_probs, axis=1)
    rho_sorted = np.concatenate([np.zeros((n, 1)), cumsum[:, :-1]], axis=1)
    rho = rho_sorted[np.arange(n)[:, None], ranks - 1]

    if method == "raps":
        return rho + u * probs + float(raps_lambda) * np.maximum(ranks - int(raps_k_reg), 0)

    if method == "saps":
        pmax = probs.max(axis=1, keepdims=True)
        base = pmax + float(saps_lambda) * (ranks - 2 + u)
        top = u * probs
        return np.where(ranks == 1, top, base)

    raise ValueError(f"Unknown score method: {method!r}")


def true_label_scores(score_matrix: np.ndarray, y: np.ndarray) -> np.ndarray:
    return score_matrix[np.arange(len(y)), y]


def split_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    k = int(math.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.partition(scores, k - 1)[k - 1])


def group_thresholds(scores: np.ndarray, groups: np.ndarray, alpha: float) -> dict:
    out = {}
    for g in sorted(np.unique(groups)):
        out[int(g)] = split_conformal_quantile(scores[groups == g], alpha)
    return out


def empirical_cdf_at_threshold(scores_group: np.ndarray, threshold: float) -> float:
    return float(np.mean(scores_group <= threshold))


def empirical_group_coverage(scores: np.ndarray, groups: np.ndarray, threshold_by_group) -> dict:
    out = {}
    for g in sorted(np.unique(groups)):
        mask = groups == g
        threshold = threshold_by_group[g] if isinstance(threshold_by_group, dict) else threshold_by_group
        out[int(g)] = empirical_cdf_at_threshold(scores[mask], threshold)
    return out


def average_set_size_at_threshold(scores_all: np.ndarray, threshold: float) -> float:
    return float(np.mean((scores_all <= threshold).sum(axis=1)))


def average_group_set_size(scores_all: np.ndarray, groups: np.ndarray, threshold_by_group) -> dict:
    out = {}
    for g in sorted(np.unique(groups)):
        mask = groups == g
        threshold = threshold_by_group[g] if isinstance(threshold_by_group, dict) else threshold_by_group
        out[int(g)] = average_set_size_at_threshold(scores_all[mask], threshold)
    return out


def group_weights(groups: np.ndarray) -> dict:
    unique, counts = np.unique(groups, return_counts=True)
    weights = counts / counts.sum()
    return {int(g): float(w) for g, w in zip(unique, weights)}


def make_score_grid(max_score: float, step: float, max_points: int = 5001) -> np.ndarray:
    max_score = float(max_score)
    step = float(step)
    if max_score <= 0:
        return np.array([0.0, 1.0], dtype=float)
    n_points = int(max_score / max(step, 1e-9)) + 1
    if n_points > max_points:
        return np.linspace(0.0, max_score, num=max_points, dtype=float)
    return np.arange(0.0, max_score + step / 2.0, step, dtype=float)


def size_curve_from_scores(scores_all_group: np.ndarray, grid: np.ndarray) -> np.ndarray:
    cutoffs = np.sort(scores_all_group.reshape(-1))
    n_examples = scores_all_group.shape[0]
    return np.searchsorted(cutoffs, grid, side="right") / max(n_examples, 1)


def find_threshold_for_target_size(scores_all_group: np.ndarray, target_size: float, grid_step: float) -> float:
    max_score = float(np.max(scores_all_group))
    grid = make_score_grid(max_score=max_score, step=grid_step)
    curve = size_curve_from_scores(scores_all_group, grid)
    idx = int(np.argmin(np.abs(curve - float(target_size))))
    return float(grid[idx])


def weighted_rms(values_by_group: dict, weights_by_group: dict) -> float:
    groups = sorted(values_by_group.keys())
    vals = np.array([values_by_group[g] for g in groups], dtype=float)
    w = np.array([weights_by_group[g] for g in groups], dtype=float)
    return float(np.sqrt(np.sum(w * vals ** 2)))


def weighted_sd(values_by_group: dict, weights_by_group: dict) -> float:
    groups = sorted(values_by_group.keys())
    vals = np.array([values_by_group[g] for g in groups], dtype=float)
    w = np.array([weights_by_group[g] for g in groups], dtype=float)
    mu = float(np.sum(w * vals))
    return float(np.sqrt(np.sum(w * (vals - mu) ** 2)))


# -----------------------------------------------------------------------------
# Calibration-derived pooled floor + qualitative Section-3 checks
# -----------------------------------------------------------------------------
def _segment_slope(y1: float, y0: float, x1: float, x0: float) -> float:
    dx = abs(float(x1) - float(x0))
    if dx < 1e-12:
        return 0.0
    return abs(float(y1) - float(y0)) / dx


def _aggregate_bound_from_group_slopes(slopes: dict, deltas: dict, weights: dict) -> float:
    total = 0.0
    for g in sorted(weights):
        total += weights[g] * (float(slopes[g]) ** 2) * (float(deltas[g]) ** 2)
    return float(np.sqrt(total))


def section3_support_metrics_from_scores(
    score_mat_cal: np.ndarray,
    score_mat_test: np.ndarray,
    scores_cal_true: np.ndarray,
    scores_test_true: np.ndarray,
    g_cal: np.ndarray,
    g_test: np.ndarray,
    alpha: float,
    grid_step: float,
) -> dict:
    """Return a stable MultiNLI-oriented Section-3 evidence summary.

    Quantitative check:
        pooled-threshold floor only (most stable on real data)

    Qualitative / mechanism checks:
        - pooled sign pattern: sign eps_g(q) follows sign(q - q_g)
        - Thm 6 style: sign(lambda_g - l_g(q)) follows sign(q_g - q)
        - Thm 7 style: sign(F_g(tau_g)-F_g(q_g)) follows sign(lambda - lambda_g)
    """

    q = split_conformal_quantile(scores_cal_true, alpha)
    q_g = group_thresholds(scores_cal_true, g_cal, alpha)

    p_test = group_weights(g_test)
    groups = sorted(p_test.keys())

    # Pooled-threshold floor: calibration RHS, test LHS.
    F_cal_q = empirical_group_coverage(scores_cal_true, g_cal, q)
    F_cal_qg = empirical_group_coverage(scores_cal_true, g_cal, q_g)
    delta_q_for_floor = {g: float(q - q_g[g]) for g in groups}
    m_seg = {g: _segment_slope(F_cal_q[g], F_cal_qg[g], q, q_g[g]) for g in groups}
    rhs_floor = _aggregate_bound_from_group_slopes(m_seg, delta_q_for_floor, p_test)

    F_test_q = empirical_group_coverage(scores_test_true, g_test, q)
    eps_test = {g: F_test_q[g] - (1.0 - alpha) for g in groups}
    lhs_floor = weighted_rms(eps_test, p_test)
    slack_floor = lhs_floor - rhs_floor

    # Test-side mechanism quantities.
    size_test_q = average_group_set_size(score_mat_test, g_test, q)
    size_test_qg = average_group_set_size(score_mat_test, g_test, q_g)
    delta_size_test = {g: size_test_qg[g] - size_test_q[g] for g in groups}
    lhs_size = weighted_rms(delta_size_test, p_test)

    lambda_target_test = float(sum(p_test[g] * size_test_qg[g] for g in groups))
    tau_g = {
        g: find_threshold_for_target_size(score_mat_cal[g_cal == g], lambda_target_test, grid_step)
        for g in groups
    }
    F_test_qg = empirical_group_coverage(scores_test_true, g_test, q_g)
    F_test_tau = empirical_group_coverage(scores_test_true, g_test, tau_g)
    delta_cov_test = {g: F_test_tau[g] - F_test_qg[g] for g in groups}
    lhs_cov = weighted_rms(delta_cov_test, p_test)

    # Sign checks.
    pooled_expected = {g: float(q - q_g[g]) for g in groups}
    pooled_obs = eps_test
    pooled_sign = sign_match_summary(pooled_expected, pooled_obs, SIGN_TOL_Q, SIGN_TOL_EPS)

    thm6_expected = {g: float(q_g[g] - q) for g in groups}
    thm6_obs = delta_size_test
    thm6_sign = sign_match_summary(thm6_expected, thm6_obs, SIGN_TOL_Q, SIGN_TOL_SIZE)

    thm7_expected = {g: float(lambda_target_test - size_test_qg[g]) for g in groups}
    thm7_obs = delta_cov_test
    thm7_sign = sign_match_summary(thm7_expected, thm7_obs, SIGN_TOL_SIZE, SIGN_TOL_COV)

    return {
        "alpha": float(alpha),
        "q_pooled_cal": float(q),
        "lhs_floor_test": float(lhs_floor),
        "rhs_floor_cal": float(rhs_floor),
        "slack_floor": float(slack_floor),
        "ratio_floor": _safe_ratio(lhs_floor, rhs_floor),
        "sat_floor": int(slack_floor >= -1e-10),
        "pooled_sign_match": pooled_sign["match"],
        "pooled_sign_informative": pooled_sign["informative"],
        "pooled_sign_frac": pooled_sign["frac"],
        "thm6_sign_match": thm6_sign["match"],
        "thm6_sign_informative": thm6_sign["informative"],
        "thm6_sign_frac": thm6_sign["frac"],
        "thm7_sign_match": thm7_sign["match"],
        "thm7_sign_informative": thm7_sign["informative"],
        "thm7_sign_frac": thm7_sign["frac"],
        "rms_cov_pooled": float(lhs_floor),
        "rms_size_from_groupwise": float(lhs_size),
        "rms_cov_from_equalized_size": float(lhs_cov),
        "lambda_target_test": float(lambda_target_test),
        "n_groups": int(len(groups)),
    }


# -----------------------------------------------------------------------------
# Seed-wise collections
# -----------------------------------------------------------------------------
def infer_temperature_settings(run_root: Path) -> tuple[int, list[float], float]:
    config = read_run_config(run_root)
    temp_group = int(config.get("temperature_sweep_group", 0))
    temp_values = [float(x) for x in config.get("temperature_sweep_values", [1.0, 1.1, 1.25, 1.5, 1.75, 2.0])]
    grid_step = float(config.get("grid_step", DEFAULT_GRID_STEP))
    return temp_group, temp_values, grid_step


def read_score_config(run_root: Path, score_name: str) -> dict:
    config = read_run_config(run_root)
    score_temp = float(config.get("score_temperature", DEFAULT_SCORE_TEMPERATURE))
    out = {
        "score_temperature": score_temp,
        "raps_lambda": float(config.get("raps_lambda", DEFAULT_RAPS_LAMBDA)),
        "raps_k_reg": int(config.get("raps_k_reg", DEFAULT_RAPS_K_REG)),
        "saps_lambda": float(config.get("saps_lambda", DEFAULT_SAPS_LAMBDA)),
    }
    if score_name == "simple":
        out["raps_lambda"] = DEFAULT_RAPS_LAMBDA
        out["raps_k_reg"] = DEFAULT_RAPS_K_REG
        out["saps_lambda"] = DEFAULT_SAPS_LAMBDA
    return out


def _load_seed_arrays(seed_dir: Path) -> dict:
    arrays_dir = seed_dir / "arrays"
    needed = [
        arrays_dir / "probs_cal.npy",
        arrays_dir / "probs_test.npy",
        arrays_dir / "y_cal.npy",
        arrays_dir / "y_test.npy",
        arrays_dir / "g_cal.npy",
        arrays_dir / "g_test.npy",
    ]
    if not all(p.exists() for p in needed):
        raise FileNotFoundError(f"Missing arrays under: {arrays_dir}")
    return {
        "probs_cal": np.load(arrays_dir / "probs_cal.npy"),
        "probs_test": np.load(arrays_dir / "probs_test.npy"),
        "y_cal": np.load(arrays_dir / "y_cal.npy"),
        "y_test": np.load(arrays_dir / "y_test.npy"),
        "g_cal": np.load(arrays_dir / "g_cal.npy"),
        "g_test": np.load(arrays_dir / "g_test.npy"),
    }


def collect_alpha_support_by_seed(run_root: Path, score_name: str) -> pd.DataFrame:
    score_cfg = read_score_config(run_root, score_name)
    rows = []
    for seed_dir in sorted(run_root.glob("seed_*")):
        seed_match = re.search(r"seed_(\d+)", seed_dir.name)
        seed = int(seed_match.group(1)) if seed_match else -1
        arrays = _load_seed_arrays(seed_dir)

        alpha_table = seed_dir / "tables" / "alpha_summary.csv"
        if not alpha_table.exists():
            continue
        alpha_values = pd.read_csv(alpha_table)["alpha"].tolist()

        score_mat_cal = all_label_scores(arrays["probs_cal"], method=score_name, **score_cfg)
        score_mat_test = all_label_scores(arrays["probs_test"], method=score_name, **score_cfg)
        scores_cal_true = true_label_scores(score_mat_cal, arrays["y_cal"])
        scores_test_true = true_label_scores(score_mat_test, arrays["y_test"])

        grid_step = float(read_run_config(run_root).get("grid_step", DEFAULT_GRID_STEP))

        for alpha in alpha_values:
            metrics = section3_support_metrics_from_scores(
                score_mat_cal=score_mat_cal,
                score_mat_test=score_mat_test,
                scores_cal_true=scores_cal_true,
                scores_test_true=scores_test_true,
                g_cal=arrays["g_cal"],
                g_test=arrays["g_test"],
                alpha=float(alpha),
                grid_step=grid_step,
            )
            rows.append({"seed": seed, **metrics})
    if not rows:
        raise FileNotFoundError("Could not compute alpha support checks from saved arrays.")
    return pd.DataFrame(rows)


def collect_temperature_support_by_seed(run_root: Path, score_name: str) -> pd.DataFrame:
    score_cfg = read_score_config(run_root, score_name)
    temp_group, temp_values, grid_step = infer_temperature_settings(run_root)
    primary_alpha = float(read_run_config(run_root).get("primary_alpha", 0.10))
    rows = []

    for seed_dir in sorted(run_root.glob("seed_*")):
        seed_match = re.search(r"seed_(\d+)", seed_dir.name)
        seed = int(seed_match.group(1)) if seed_match else -1
        arrays = _load_seed_arrays(seed_dir)
        groups = np.unique(arrays["g_test"])

        for temp_value in temp_values:
            temp_map = {int(g): 1.0 for g in groups}
            temp_map[temp_group] = float(temp_value)

            probs_cal_adj = apply_group_temperature(arrays["probs_cal"], arrays["g_cal"], temp_map)
            probs_test_adj = apply_group_temperature(arrays["probs_test"], arrays["g_test"], temp_map)

            score_mat_cal = all_label_scores(probs_cal_adj, method=score_name, **score_cfg)
            score_mat_test = all_label_scores(probs_test_adj, method=score_name, **score_cfg)
            scores_cal_true = true_label_scores(score_mat_cal, arrays["y_cal"])
            scores_test_true = true_label_scores(score_mat_test, arrays["y_test"])

            metrics = section3_support_metrics_from_scores(
                score_mat_cal=score_mat_cal,
                score_mat_test=score_mat_test,
                scores_cal_true=scores_cal_true,
                scores_test_true=scores_test_true,
                g_cal=arrays["g_cal"],
                g_test=arrays["g_test"],
                alpha=primary_alpha,
                grid_step=grid_step,
            )
            rows.append({"seed": seed, "temperature_value": float(temp_value), **metrics})

    if not rows:
        raise FileNotFoundError("Could not compute temperature support checks from saved arrays.")
    return pd.DataFrame(rows)


def aggregate_support(df: pd.DataFrame, by: str) -> pd.DataFrame:
    numeric_cols = [c for c in df.columns if c not in {by, "seed"}]
    grouped = df.groupby(by, as_index=False)[numeric_cols].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        col if isinstance(col, str) else (col[0] if col[1] == "" else f"{col[0]}_{col[1]}")
        for col in grouped.columns
    ]

    extra = df.groupby(by).agg(
        sat_floor_count=("sat_floor", "sum"),
        n_seeds=("seed", "nunique"),
        min_slack_floor=("slack_floor", "min"),
        pooled_sign_match_sum=("pooled_sign_match", "sum"),
        pooled_sign_informative_sum=("pooled_sign_informative", "sum"),
        thm6_sign_match_sum=("thm6_sign_match", "sum"),
        thm6_sign_informative_sum=("thm6_sign_informative", "sum"),
        thm7_sign_match_sum=("thm7_sign_match", "sum"),
        thm7_sign_informative_sum=("thm7_sign_informative", "sum"),
    ).reset_index()

    out = grouped.merge(extra, on=by, how="left")
    out["sat_floor_rate"] = out["sat_floor_count"] / out["n_seeds"].replace(0, np.nan)
    out["pooled_sign_rate"] = out["pooled_sign_match_sum"] / out["pooled_sign_informative_sum"].replace(0, np.nan)
    out["thm6_sign_rate"] = out["thm6_sign_match_sum"] / out["thm6_sign_informative_sum"].replace(0, np.nan)
    out["thm7_sign_rate"] = out["thm7_sign_match_sum"] / out["thm7_sign_informative_sum"].replace(0, np.nan)
    return out


# -----------------------------------------------------------------------------
# Figure builders
# -----------------------------------------------------------------------------
def build_primary_overview_figure(
    df_group: pd.DataFrame,
    out_base: Path,
    score_name: str,
    alpha: float,
) -> None:
    order = df_group.sort_values("epsilon_pooled_mean", ascending=False).copy().reset_index(drop=True)
    labels = order["group_name"].tolist()
    y = np.arange(len(order))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(8.6, 6.5),
        sharey=True,
        gridspec_kw={"width_ratios": [0.92, 0.92, 0.92], "wspace": 0.26},
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
        lim = symmetric_limit(vals, errs, pad=0.035)

        ax.axvline(0.0, color=GRAY_ZERO, linestyle="--", linewidth=1.0, zorder=1)
        ax.grid(axis="x", color="0.88", linewidth=0.6)

        pos_color = spec["color"]
        neg_color = lighten_color(spec["color"], factor=0.62)
        facecolors = [pos_color if v >= 0 else neg_color for v in vals]

        bars = ax.barh(
            y,
            vals,
            height=0.82,
            color=facecolors,
            edgecolor=spec["color"],
            linewidth=1.3,
            zorder=3,
        )

        if np.any(errs > 0):
            ax.errorbar(
                vals,
                y,
                xerr=errs,
                fmt="none",
                ecolor="0.30",
                elinewidth=0.9,
                capsize=2.2,
                zorder=4,
            )

        ax.set_xlim(-lim, lim)
        ax.xaxis.set_major_locator(FixedLocator(symmetric_three_ticks(lim)))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
        ax.set_xlabel(spec["xlabel"], fontsize=16.0)
        if spec["letter"] == "C":
            ax.xaxis.label.set_x(0.43)
        ax.set_title(spec["title"], pad=8, fontsize=plt.rcParams["axes.labelsize"])
        ax.text(
            0.01,
            1.10,
            spec["letter"],
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=18,
            fontweight="bold",
        )
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", rotation=45, labelsize=14.0)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    #axes[0].set_ylabel("Genre")
    for ax in axes[1:]:
        ax.tick_params(axis="y", left=False, labelleft=False)
    axes[0].invert_yaxis()

    savefig(fig, out_base, rect=(0.0, 0.05, 0.95, 1.0))


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
    xtick_fontsize = 13.5

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.8, 4.9),
        gridspec_kw={"width_ratios": [1.02, 0.98, 0.98], "wspace": 0.56},
    )

    # Panel A: pooled-threshold floor ratio.
    ax = axes[0]
    ratio = df["ratio_floor_mean"].to_numpy(float)
    ratio_err = np.nan_to_num(df.get("ratio_floor_std", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float), nan=0.0)
    ymin = 0.4
    ymax = max(1.06, float(np.nanmax(ratio + ratio_err)) + 0.05)
    label_offset = 0.012 * (ymax - ymin)

    ax.grid(axis="y", color="0.90", linewidth=0.6)
    ax.axhline(1.0, color=GRAY_ZERO, linestyle="--", linewidth=1.0)
    ax.bar(
        xpos,
        ratio,
        width=0.52,
        color=PANEL_COLORS["floor"],
        zorder=2,
    )
    if np.any(ratio_err > 0):
        ax.errorbar(
            xpos,
            ratio,
            yerr=ratio_err,
            fmt="none",
            ecolor="0.25",
            elinewidth=0.8,
            capsize=2.0,
            zorder=3,
        )
    for x, yv in zip(xpos, ratio):
        ax.text(x, yv + label_offset, format_ratio(yv), ha="center", va="bottom", fontsize=11.0)

    ax.set_ylim(ymin, ymax)
    ax.set_xticks(xpos)
    ax.set_xticklabels(x_labels, fontsize=xtick_fontsize)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Empirical / lower-bound proxy", labelpad=4)
    ax.set_title("Pooled floor", pad=6, fontsize=14.0)
    ax.text(-0.10, 1.04, "A", transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")

    # Panel B: induced size distortion under q_G.
    ax = axes[1]
    vals = df["rms_size_from_groupwise_mean"].to_numpy(float)
    errs = np.nan_to_num(df.get("rms_size_from_groupwise_std", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float), nan=0.0)
    ax.grid(axis="y", color="0.90", linewidth=0.6)
    ax.bar(xpos, vals, width=0.52, color=PANEL_COLORS["size"])
    if np.any(errs > 0):
        ax.errorbar(xpos, vals, yerr=errs, fmt="none", ecolor="0.25", elinewidth=0.8, capsize=2.0, zorder=3)
    ax.set_xticks(xpos)
    ax.set_xticklabels(x_labels, fontsize=xtick_fontsize)
    ax.set_xlabel(x_label)
    ax.set_ylabel("RMS set-size distortion", labelpad=2)
    ax.set_title("Size distortion", pad=6, fontsize=14.0)
    ax.text(-0.10, 1.04, "B", transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")

    # Panel C: induced coverage distortion under lambda.
    ax = axes[2]
    vals = df["rms_cov_from_equalized_size_mean"].to_numpy(float)
    errs = np.nan_to_num(df.get("rms_cov_from_equalized_size_std", pd.Series(np.zeros(len(df)))).to_numpy(dtype=float), nan=0.0)
    ax.grid(axis="y", color="0.90", linewidth=0.6)
    ax.bar(xpos, vals, width=0.52, color=PANEL_COLORS["cov"])
    if np.any(errs > 0):
        ax.errorbar(xpos, vals, yerr=errs, fmt="none", ecolor="0.25", elinewidth=0.8, capsize=2.0, zorder=3)
    ax.set_xticks(xpos)
    ax.set_xticklabels(x_labels, fontsize=xtick_fontsize)
    ax.set_xlabel(x_label)
    ax.set_ylabel("RMS coverage distortion", labelpad=2)
    ax.set_title("Coverage distortion", pad=6, fontsize=14.0)
    ax.text(-0.10, 1.04, "C", transform=ax.transAxes, ha="left", va="bottom", fontweight="bold")

    savefig(fig, out_base, rect=(0.0, 0.02, 1.0, 0.94))


# -----------------------------------------------------------------------------
# Table builders
# -----------------------------------------------------------------------------
def summarize_primary_seed_table(df: pd.DataFrame) -> pd.Series:
    if "seed" in df.columns and len(df) > 1:
        s = df.mean(numeric_only=True)
        if "alpha" not in s and "alpha" in df.columns:
            s["alpha"] = df["alpha"].iloc[0]
        return s
    return df.iloc[0]


def build_primary_mechanism_table(primary_summary_row: pd.Series, out_dir: Path, score_name: str, alpha_val: float) -> None:
    pretty = pd.DataFrame(
        [
            {
                r"$\sigma_\Delta$": format_num_plain(primary_summary_row["sigma_delta"]),
                r"$R_\varepsilon(q)$": format_num_plain(primary_summary_row["rms_cov_pooled"]),
                r"$\lambda$": format_num_plain(primary_summary_row["lambda_target"]),
                r"$R_\lambda(q)$": format_num_plain(primary_summary_row["rms_size_from_groupwise"]),
                r"$\sigma_\lambda$": format_num_plain(primary_summary_row["sigma_lambda"]),
                r"$R_{\rm cov}(\lambda)$": format_num_plain(primary_summary_row["rms_cov_from_equalized_size"]),
            }
        ]
    )
    pretty.to_csv(out_dir / "table_primary_mechanism_summary.csv", index=False)
    tex = latex_table_from_df(
        pretty,
        caption=(
            f"Primary-point mechanism summary for MultiNLI ({score_name} score, $\\alpha={alpha_val:.2f}$). "
            "The table reports the pooled-threshold heterogeneity scale, the induced pooled coverage distortion, "
            "the common expected-size target under equalized coverage, the resulting size disparity, and the "
            "coverage disparity reintroduced under equalized expected size."
        ),
        label="tab:mnli-primary-mechanism-summary",
        column_format="cccccc",
    )
    (out_dir / "table_primary_mechanism_summary.tex").write_text(tex)


def build_primary_section3_support_table(df_primary: pd.DataFrame, out_dir: Path, score_name: str, alpha_val: float) -> None:
    row = df_primary.iloc[0]
    pretty = pd.DataFrame(
        [
            {
                "Check": "Pooled-threshold floor",
                "Empirical summary": r"$R_\varepsilon^{\rm test}\ge B_\varepsilon^{\rm cal}$",
                "Evidence": f"ratio {format_ratio(row['ratio_floor_mean'])}; slack {format_slack(row['slack_floor_mean'])}; sat. {int(row['sat_floor_count'])}/{int(row['n_seeds'])}",
            },
            {
                "Check": "Pooled sign pattern",
                "Empirical summary": r"sign$\,\widehat\varepsilon_g(q)$ follows sign$(q-q_g)$",
                "Evidence": f"{int(row['pooled_sign_match_sum'])}/{int(row['pooled_sign_informative_sum'])} informative groups",
            },
            {
                "Check": r"Equalized coverage $\Rightarrow$ size disparity",
                "Empirical summary": r"sign$(\lambda_g-\ell_g(q))$ follows sign$(q_g-q)$",
                "Evidence": f"{int(row['thm6_sign_match_sum'])}/{int(row['thm6_sign_informative_sum'])} informative groups; RMS {format_num_plain(row['rms_size_from_groupwise_mean'])}",
            },
            {
                "Check": r"Equalized size $\Rightarrow$ coverage disparity",
                "Empirical summary": r"sign$(\widehat F_g(\tau_g)-\widehat F_g(q_g))$ follows sign$(\lambda-\lambda_g)$",
                "Evidence": f"{int(row['thm7_sign_match_sum'])}/{int(row['thm7_sign_informative_sum'])} informative groups; RMS {format_num_plain(row['rms_cov_from_equalized_size_mean'])}",
            },
        ]
    )
    pretty.to_csv(out_dir / "table_primary_section3_support.csv", index=False)
    tex = latex_table_from_df(
        pretty,
        caption=(
            f"Section~3 support summary for MultiNLI at the primary operating point ({score_name} score, "
            f"$\\alpha={alpha_val:.2f}$). The pooled-threshold floor is checked quantitatively, while the equalized-coverage "
            "and equalized-size claims are validated through theorem-aligned sign consistency and nonzero induced disparity."
        ),
        label="tab:mnli-primary-section3-support",
        column_format="p{3.4cm}p{5.2cm}p{4.0cm}",
    )
    (out_dir / "table_primary_section3_support.tex").write_text(tex)


def build_primary_group_full_table(df_group: pd.DataFrame, out_dir: Path, score_name: str) -> None:
    order = df_group.sort_values("epsilon_pooled_mean", ascending=False).copy()
    pretty = pd.DataFrame(
        {
            "Genre": order["group_name"],
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
        caption=f"Full per-genre summary for MultiNLI at the primary operating point ({score_name} score).",
        label="tab:mnli-primary-groups-full",
        column_format="l" + "c" * (len(pretty.columns) - 1),
    )
    (out_dir / "table_primary_group_full.tex").write_text(tex)


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
            r"$R_\varepsilon^{\rm test}$": df["lhs_floor_test_mean"].map(format_num_plain),
            r"$B_\varepsilon^{\rm cal}$": df["rhs_floor_cal_mean"].map(format_num_plain),
            "ratio": df["ratio_floor_mean"].map(format_ratio),
            r"slack$_\varepsilon$": df["slack_floor_mean"].map(format_slack),
            r"sat$_\varepsilon$": [f"{int(a)}/{int(b)}" for a, b in zip(df["sat_floor_count"], df["n_seeds"])],
            r"pooled sign": [f"{int(a)}/{int(b)}" for a, b in zip(df["pooled_sign_match_sum"], df["pooled_sign_informative_sum"])],
            r"Thm.6 sign": [f"{int(a)}/{int(b)}" for a, b in zip(df["thm6_sign_match_sum"], df["thm6_sign_informative_sum"])],
            r"Thm.7 sign": [f"{int(a)}/{int(b)}" for a, b in zip(df["thm7_sign_match_sum"], df["thm7_sign_informative_sum"])],
            r"$R_\lambda$": df["rms_size_from_groupwise_mean"].map(format_num_plain),
            r"$R_{\rm cov}$": df["rms_cov_from_equalized_size_mean"].map(format_num_plain),
        }
    )
    pretty.to_csv(out_csv, index=False)
    tex = latex_table_from_df(
        pretty,
        caption=caption,
        label=label,
        column_format="l" + "c" * (len(pretty.columns) - 1),
    )
    out_tex.write_text(tex)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NeurIPS-ready MultiNLI figures/tables from an existing output directory.")
    parser.add_argument("run_root", type=str, help="Existing output directory produced by the notebook.")
    args = parser.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    if not run_root.exists():
        raise FileNotFoundError(f"Run root does not exist: {run_root}")

    paper_root = ensure_dir(run_root / "paper_ready")
    main_fig_dir = clear_generated_dir(paper_root / "main_text" / "figures")
    main_tab_dir = clear_generated_dir(paper_root / "main_text" / "tables")
    app_fig_dir = clear_generated_dir(paper_root / "appendix" / "figures")
    app_tab_dir = clear_generated_dir(paper_root / "appendix" / "tables")

    score_name = read_score_name(run_root)

    df_primary_summary = pd.read_csv(candidate_primary_summary(run_root))
    primary_summary_row = summarize_primary_seed_table(df_primary_summary)

    df_primary_groups_raw = pd.read_csv(candidate_primary_group_metrics(run_root))
    if "seed" in df_primary_groups_raw.columns:
        df_primary_groups = collapse_primary_group_metrics(df_primary_groups_raw)
    else:
        rename_map = {c: f"{c}_mean" for c in df_primary_groups_raw.columns if c not in {"group", "group_name"}}
        df_primary_groups = df_primary_groups_raw.rename(columns=rename_map)
        if "group_mean" in df_primary_groups.columns:
            df_primary_groups = df_primary_groups.rename(columns={"group_mean": "group"})

    df_alpha_seed = collect_alpha_support_by_seed(run_root, score_name=score_name)
    df_alpha = aggregate_support(df_alpha_seed, by="alpha").sort_values("alpha")

    df_temp_seed = collect_temperature_support_by_seed(run_root, score_name=score_name)
    df_temp = aggregate_support(df_temp_seed, by="temperature_value").sort_values("temperature_value")

    alpha_val = float(primary_summary_row["alpha"])
    df_primary = df_alpha[df_alpha["alpha"] == alpha_val].copy()
    if df_primary.empty:
        raise RuntimeError(f"Could not find primary alpha {alpha_val} in recomputed support summary.")

    # Main text: 1 figure + 2 tables.
    build_primary_overview_figure(
        df_primary_groups,
        main_fig_dir / "figure_primary_overview",
        score_name=score_name,
        alpha=alpha_val,
    )
    build_primary_mechanism_table(primary_summary_row, main_tab_dir, score_name=score_name, alpha_val=alpha_val)
    build_primary_section3_support_table(df_primary, main_tab_dir, score_name=score_name, alpha_val=alpha_val)

    # Appendix: bar-plot robustness summaries + detailed tables.
    build_support_magnitude_bar_figure(
        df_alpha,
        x_col="alpha",
        x_label=r"$\alpha$",
        title=f"Alpha sweep ({score_name})",
        out_base=app_fig_dir / "figure_alpha_section3_summary",
    )
    build_support_magnitude_bar_figure(
        df_temp,
        x_col="temperature_value",
        x_label="Temperature",
        title=f"Temperature sweep ({score_name})",
        out_base=app_fig_dir / "figure_temperature_section3_summary",
    )
    build_primary_group_full_table(df_primary_groups, app_tab_dir, score_name=score_name)
    build_support_table(
        df_alpha,
        x_col="alpha",
        x_label_tex=r"$\alpha$",
        out_csv=app_tab_dir / "table_alpha_section3_support.csv",
        out_tex=app_tab_dir / "table_alpha_section3_support.tex",
        caption=(
            f"Alpha-robustness support summary for Section~3 on MultiNLI ({score_name} score). "
            "The pooled-threshold floor is reported quantitatively; the equalized-coverage and equalized-size claims "
            "are summarized through sign-consistency counts together with the induced distortion magnitudes."
        ),
        label="tab:mnli-alpha-section3-support",
    )
    build_support_table(
        df_temp,
        x_col="temperature_value",
        x_label_tex="Temperature",
        out_csv=app_tab_dir / "table_temperature_section3_support.csv",
        out_tex=app_tab_dir / "table_temperature_section3_support.tex",
        caption=(
            f"Temperature-sweep support summary for Section~3 on MultiNLI ({score_name} score). "
            "The pooled-threshold floor is reported quantitatively; the equalized-coverage and equalized-size claims "
            "are summarized through sign-consistency counts together with the induced distortion magnitudes."
        ),
        label="tab:mnli-temperature-section3-support",
    )

    manifest = {
        "run_root": str(run_root),
        "score": score_name,
        "main_text": {
            "figures": sorted(p.name for p in main_fig_dir.iterdir()),
            "tables": sorted(p.name for p in main_tab_dir.iterdir()),
            "suggested_usage": [
                "Use figure_primary_overview as the single main-text figure.",
                "Use table_primary_mechanism_summary and table_primary_section3_support as the two main-text tables.",
            ],
        },
        "appendix": {
            "figures": sorted(p.name for p in app_fig_dir.iterdir()),
            "tables": sorted(p.name for p in app_tab_dir.iterdir()),
            "notes": [
                "Appendix Panel A reports the pooled-threshold lower-bound ratio against the reference line at 1.",
                r"Appendix Panel B reports the induced size distortion $\sqrt{E[(\ell_G(q_G)-\ell_G(q))^2]}$.",
                r"Appendix Panel C reports the induced coverage distortion $\sqrt{E[(\widehat F_G(\tau_G)-\widehat F_G(q_G))^2]}$.",
            ],
        },
    }
    (paper_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[done] wrote paper-ready outputs to: {paper_root}")


if __name__ == "__main__":
    main()
