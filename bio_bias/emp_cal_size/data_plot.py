#!/usr/bin/env python3
"""
Process Bias-in-Bios conformal result zip files and generate calibration-size
analysis for uncertainty-floor detectability.

Place this script in the same directory as result_simple.zip, result_saps.zip,
and result_raps.zip, then run:

    python data_plot.py

It reads the arrays directly from each zip, performs stratified calibration
subsampling, evaluates pooled-threshold group-wise miscoverage on the fixed test
split, and writes plots + CSV summaries to ./data_plot_output.

Main outputs:
  - data_plot_output/full_metrics_summary.csv
  - data_plot_output/subsample_summary.csv
  - data_plot_output/detectability_scaling.csv
  - data_plot_output/figure_floor_vs_n.pdf
  - data_plot_output/figure_snr_vs_n.pdf
  - data_plot_output/figure_floor_inflation_vs_n.pdf
  - data_plot_output/figure_sigma_inflation_vs_n.pdf
  - data_plot_output/figure_detectability_4panel.pdf
  - data_plot_output/figure_normalized_collapse_snr.pdf
"""

from __future__ import annotations

import argparse
import io
import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCORE_ORDER = {"simple": 0, "SAPS": 1, "RAPS": 2}

PLOT_RC = {
    "figure.dpi": 180,
    "savefig.dpi": 400,
    "font.size": 18.0,
    "axes.titlesize": 18.5,
    "axes.labelsize": 18.0,
    "xtick.labelsize": 18.5,
    "ytick.labelsize": 18.5,
    "legend.fontsize": 13.7,
    "lines.linewidth": 4.5,
    "lines.markersize": 12.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "mathtext.fontset": "dejavusans",
}

plt.rcParams.update(PLOT_RC)


def style_axis(ax, *, xlabel: str, ylabel: str, title: str, xscale: str = "log") -> None:
    if xscale:
        ax.set_xscale(xscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.grid(True, alpha=0.28, linewidth=0.8)
    ax.set_axisbelow(True)



def ordered_group_items(df: pd.DataFrame):
    items = list(df.groupby("score"))
    items.sort(key=lambda kv: SCORE_ORDER.get(str(kv[0]), 999))
    return items


# ---------- helpers ----------

def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mean = float(np.sum(weights * values) / np.sum(weights))
    var = float(np.sum(weights * (values - mean) ** 2) / np.sum(weights))
    return math.sqrt(max(var, 0.0))


def conformal_index(n: int, alpha: float) -> int:
    # split conformal style: ceil((n+1)(1-alpha)) clamped to [1, n]
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return k - 1


def conformal_quantile(values: np.ndarray, alpha: float) -> float:
    values = np.asarray(values)
    idx = conformal_index(len(values), alpha)
    # O(n) selection instead of full sort
    return float(np.partition(values, idx)[idx])


def quantile_from_sorted(sorted_values: np.ndarray, alpha: float) -> float:
    idx = conformal_index(len(sorted_values), alpha)
    return float(sorted_values[idx])


def local_density_estimate(sorted_values: np.ndarray, q: float) -> float:
    """Simple local density proxy around q using a symmetric count window."""
    x = np.asarray(sorted_values, dtype=float)
    n = len(x)
    if n < 10:
        return float("nan")
    std = np.std(x, ddof=1)
    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    robust_sigma = std if iqr <= 0 else min(std, iqr / 1.34)
    if not np.isfinite(robust_sigma) or robust_sigma <= 0:
        robust_sigma = max((x[-1] - x[0]) / 10.0, 1e-6)
    # Silverman-like bandwidth, with mild lower bound for numerical stability
    h = 0.9 * robust_sigma * (n ** (-1.0 / 5.0))
    h = max(float(h), 1e-6)
    left = np.searchsorted(x, q - h, side="left")
    right = np.searchsorted(x, q + h, side="right")
    count = right - left
    return float(count / (n * 2.0 * h))


def make_stratified_counts(total_n: int, probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    probs = probs / probs.sum()
    raw = total_n * probs
    counts = np.floor(raw).astype(int)

    # Ensure at least one observation per group when possible.
    if total_n >= len(probs):
        for i in range(len(counts)):
            if counts[i] == 0:
                counts[i] = 1

    diff = int(total_n - counts.sum())
    if diff > 0:
        frac = raw - np.floor(raw)
        order = np.argsort(-frac)
        for j in range(diff):
            counts[order[j % len(order)]] += 1
    elif diff < 0:
        frac = raw - np.floor(raw)
        order = np.argsort(frac)
        need = -diff
        for idx in order:
            while need > 0 and counts[idx] > 1:
                counts[idx] -= 1
                need -= 1
        if need > 0:
            for idx in np.argsort(-counts):
                while need > 0 and counts[idx] > 0:
                    counts[idx] -= 1
                    need -= 1
                if need == 0:
                    break
    if counts.sum() != total_n:
        raise RuntimeError(f"Failed to allocate stratified counts for n={total_n}.")
    return counts


@dataclass
class ScoreBundle:
    score_name: str
    zip_path: Path
    alpha: float
    cal_scores: np.ndarray
    test_scores: np.ndarray
    g_cal: np.ndarray
    g_test: np.ndarray
    group_ids: np.ndarray
    test_group_probs: np.ndarray
    cal_group_probs: np.ndarray
    cal_scores_by_group: Dict[int, np.ndarray]
    sorted_test_by_group: Dict[int, np.ndarray]
    sorted_test_all: np.ndarray

    @classmethod
    def from_zip(cls, zip_path: Path, alpha_override: float | None = None) -> "ScoreBundle":
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()

            def pick(suffix: str) -> str:
                hits = [n for n in names if n.endswith(suffix)]
                if not hits:
                    raise FileNotFoundError(f"Could not find {suffix} inside {zip_path.name}")
                return hits[0]

            cal_scores = np.load(io.BytesIO(zf.read(pick("/arrays/cal_scores.npy"))))
            test_scores = np.load(io.BytesIO(zf.read(pick("/arrays/test_scores.npy"))))
            g_cal = np.load(io.BytesIO(zf.read(pick("/arrays/g_cal.npy"))))
            g_test = np.load(io.BytesIO(zf.read(pick("/arrays/g_test.npy"))))
            meta = json.loads(zf.read(pick("/split_meta.json")).decode("utf-8"))

        alpha = float(alpha_override) if alpha_override is not None else float(meta.get("primary_alpha", 0.1))

        lower_name = zip_path.name.lower()
        if "simple" in lower_name:
            score_name = "simple"
        elif "saps" in lower_name:
            score_name = "SAPS"
        elif "raps" in lower_name:
            score_name = "RAPS"
        else:
            score_name = zip_path.stem

        group_ids = np.array(sorted(set(np.unique(g_cal)).union(set(np.unique(g_test)))))
        cal_scores_by_group = {int(g): np.asarray(cal_scores[g_cal == g], dtype=float) for g in group_ids}
        sorted_test_by_group = {int(g): np.sort(np.asarray(test_scores[g_test == g], dtype=float)) for g in group_ids}
        sorted_test_all = np.sort(np.asarray(test_scores, dtype=float))

        cal_group_probs = np.array([(g_cal == g).mean() for g in group_ids], dtype=float)
        test_group_probs = np.array([(g_test == g).mean() for g in group_ids], dtype=float)

        return cls(
            score_name=score_name,
            zip_path=zip_path,
            alpha=alpha,
            cal_scores=np.asarray(cal_scores, dtype=float),
            test_scores=np.asarray(test_scores, dtype=float),
            g_cal=np.asarray(g_cal),
            g_test=np.asarray(g_test),
            group_ids=group_ids,
            test_group_probs=test_group_probs,
            cal_group_probs=cal_group_probs,
            cal_scores_by_group=cal_scores_by_group,
            sorted_test_by_group=sorted_test_by_group,
            sorted_test_all=sorted_test_all,
        )

    def coverage_by_group_on_test(self, q: float) -> np.ndarray:
        cov = []
        for g in self.group_ids:
            sorted_scores = self.sorted_test_by_group[int(g)]
            count = np.searchsorted(sorted_scores, q, side="right")
            cov.append(count / len(sorted_scores))
        return np.asarray(cov, dtype=float)

    def floor_on_test(self, q: float) -> Tuple[float, np.ndarray, np.ndarray]:
        cov = self.coverage_by_group_on_test(q)
        eps = cov - (1.0 - self.alpha)
        floor = math.sqrt(float(np.sum(self.test_group_probs * eps**2)))
        return floor, cov, eps

    def groupwise_quantiles(self, score_map: Dict[int, np.ndarray], probs: np.ndarray) -> Tuple[np.ndarray, float]:
        qg = np.array([conformal_quantile(score_map[int(g)], self.alpha) for g in self.group_ids], dtype=float)
        sigma = weighted_std(qg, probs)
        return qg, sigma

    def full_metrics(self) -> Dict[str, float]:
        q_true = quantile_from_sorted(self.sorted_test_all, self.alpha)
        true_floor, _, _ = self.floor_on_test(q_true)
        q_full = conformal_quantile(self.cal_scores, self.alpha)
        obs_floor_full, _, _ = self.floor_on_test(q_full)
        qg_true = np.array([
            quantile_from_sorted(self.sorted_test_by_group[int(g)], self.alpha) for g in self.group_ids
        ], dtype=float)
        sigma_true = weighted_std(qg_true, self.test_group_probs)
        qg_full, sigma_full = self.groupwise_quantiles(self.cal_scores_by_group, self.cal_group_probs)
        m_eff = local_density_estimate(self.sorted_test_all, q_true)
        return {
            "score": self.score_name,
            "zip_file": self.zip_path.name,
            "alpha": self.alpha,
            "K": int(len(self.group_ids)),
            "n_cal_total": int(len(self.cal_scores)),
            "n_test_total": int(len(self.test_scores)),
            "q_true_test_proxy": q_true,
            "q_full_cal": q_full,
            "sigma_delta_true_test_proxy": sigma_true,
            "sigma_delta_full_cal": sigma_full,
            "true_floor_test_proxy": true_floor,
            "observed_floor_full_cal_on_test": obs_floor_full,
            "m_eff_density_proxy_at_q": m_eff,
            **{f"p_test_group_{int(g)}": float(p) for g, p in zip(self.group_ids, self.test_group_probs)},
            **{f"p_cal_group_{int(g)}": float(p) for g, p in zip(self.group_ids, self.cal_group_probs)},
            **{f"q_true_group_{int(g)}": float(q) for g, q in zip(self.group_ids, qg_true)},
            **{f"q_full_group_{int(g)}": float(q) for g, q in zip(self.group_ids, qg_full)},
        }


def subsample_analysis(bundle: ScoreBundle, n_grid: Sequence[int], reps: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    probs = bundle.cal_group_probs
    group_arrays = {int(g): np.asarray(bundle.cal_scores_by_group[int(g)], dtype=float) for g in bundle.group_ids}

    full = bundle.full_metrics()
    true_floor = float(full["true_floor_test_proxy"])
    sigma_true = float(full["sigma_delta_true_test_proxy"])

    rows: List[Dict[str, float]] = []
    full_n = len(bundle.cal_scores)

    for n_total in n_grid:
        if n_total > full_n:
            continue

        counts = make_stratified_counts(n_total, probs)

        floor_vals: List[float] = []
        sigma_vals: List[float] = []
        q_vals: List[float] = []

        # Full-size case is deterministic under the current single-seed zip.
        local_reps = 1 if n_total == full_n else reps

        for _ in range(local_reps):
            sampled_groups: Dict[int, np.ndarray] = {}
            pooled_parts = []
            for g, n_g in zip(bundle.group_ids, counts):
                arr = group_arrays[int(g)]
                idx = rng.choice(len(arr), size=int(n_g), replace=False)
                sampled = arr[idx]
                sampled_groups[int(g)] = sampled
                pooled_parts.append(sampled)
            pooled = np.concatenate(pooled_parts)

            q_sub = conformal_quantile(pooled, bundle.alpha)
            q_vals.append(q_sub)
            floor_sub, _, _ = bundle.floor_on_test(q_sub)
            floor_vals.append(floor_sub)

            qg_sub = np.array(
                [conformal_quantile(sampled_groups[int(g)], bundle.alpha) for g in bundle.group_ids],
                dtype=float,
            )
            sigma_sub = weighted_std(qg_sub, bundle.test_group_probs)
            sigma_vals.append(sigma_sub)

        floor_vals_np = np.asarray(floor_vals, dtype=float)
        sigma_vals_np = np.asarray(sigma_vals, dtype=float)
        q_vals_np = np.asarray(q_vals, dtype=float)

        floor_sd = float(np.std(floor_vals_np, ddof=1)) if len(floor_vals_np) > 1 else 0.0
        sigma_sd = float(np.std(sigma_vals_np, ddof=1)) if len(sigma_vals_np) > 1 else 0.0
        q_sd = float(np.std(q_vals_np, ddof=1)) if len(q_vals_np) > 1 else 0.0

        rows.append(
            {
                "score": bundle.score_name,
                "n_cal": int(n_total),
                "reps": int(local_reps),
                "mean_floor_eval_on_test": float(np.mean(floor_vals_np)),
                "sd_floor_eval_on_test": floor_sd,
                "median_floor_eval_on_test": float(np.median(floor_vals_np)),
                "mean_sigma_hat": float(np.mean(sigma_vals_np)),
                "sd_sigma_hat": sigma_sd,
                "mean_q_sub": float(np.mean(q_vals_np)),
                "sd_q_sub": q_sd,
                "true_floor_test_proxy": true_floor,
                "sigma_delta_true_test_proxy": sigma_true,
                "snr_true_floor_over_sd": (true_floor / floor_sd) if floor_sd > 0 else np.inf,
                "floor_inflation_ratio": float(np.mean(floor_vals_np) / true_floor) if true_floor > 0 else np.nan,
                "sigma_inflation_ratio": float(np.mean(sigma_vals_np) / sigma_true) if sigma_true > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def detectability_scaling(full_df: pd.DataFrame, subsample_df: pd.DataFrame, eta: float) -> pd.DataFrame:
    rows = []
    for _, r in full_df.iterrows():
        score = r["score"]
        sub = subsample_df[subsample_df["score"] == score].sort_values("n_cal")
        detect_row = sub[sub["snr_true_floor_over_sd"] >= 1.0].head(1)
        n_detect = float(detect_row["n_cal"].iloc[0]) if len(detect_row) else np.nan
        K = float(r["K"])
        m_eff = float(r["m_eff_density_proxy_at_q"])
        sigma_true = float(r["sigma_delta_true_test_proxy"])
        denom = K * math.log(K / eta)
        scaled_n_detect = (n_detect * (m_eff**2) * (sigma_true**2) / denom) if np.isfinite(n_detect) else np.nan
        rows.append(
            {
                "score": score,
                "eta": eta,
                "K": int(K),
                "m_eff_density_proxy_at_q": m_eff,
                "sigma_delta_true_test_proxy": sigma_true,
                "n_detect_snr_ge_1": n_detect,
                "scaled_n_detect_constant": scaled_n_detect,
                "true_floor_test_proxy": float(r["true_floor_test_proxy"]),
                "observed_floor_full_cal_on_test": float(r["observed_floor_full_cal_on_test"]),
            }
        )
    return pd.DataFrame(rows)



def _draw_floor_vs_n(ax, df: pd.DataFrame) -> None:
    handles = []
    labels = []
    for score, sub in ordered_group_items(df):
        sub = sub.sort_values("n_cal")
        x = sub["n_cal"].to_numpy()
        y = sub["mean_floor_eval_on_test"].to_numpy()
        s = sub["sd_floor_eval_on_test"].to_numpy()
        tf = float(sub["true_floor_test_proxy"].iloc[0])

        line, = ax.plot(x, y, marker="o")
        color = line.get_color()
        ax.fill_between(x, np.maximum(y - s, 0.0), y + s, color=color, alpha=0.18)
        floor_line = ax.axhline(tf, linestyle="--", linewidth=1.5, color=color)

        handles.extend([line, floor_line])
        labels.extend([f"{score}: mean empirical floor", f"{score}: true-floor proxy"])

    style_axis(
        ax,
        xlabel="Calibration size",
        ylabel="RMS group miscoverage on fixed test split",
        title="A. Empirical floor vs calibration size",
    )
    ax.legend(handles, labels, frameon=False, ncol=2, loc="upper right", columnspacing=1.0, handlelength=2.6)


def plot_floor_vs_n(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    _draw_floor_vs_n(ax, df)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _draw_snr_vs_n(ax, df: pd.DataFrame) -> None:
    handles = []
    labels = []
    for score, sub in ordered_group_items(df):
        sub = sub.sort_values("n_cal")
        sub = sub[np.isfinite(sub["snr_true_floor_over_sd"])].copy()
        if len(sub) == 0:
            continue
        line, = ax.plot(sub["n_cal"], sub["snr_true_floor_over_sd"], marker="o")
        handles.append(line)
        labels.append(score)

    thresh = ax.axhline(1.0, linestyle="--", linewidth=1.4, color="black")
    handles.append(thresh)
    labels.append("SNR = 1 threshold")

    style_axis(
        ax,
        xlabel="Calibration size",
        ylabel=r"SNR = true floor / sd(empirical floor)",
        title="B. Detectability of the structural floor",
    )
    ax.legend(handles, labels, frameon=False, loc="upper left")


def plot_snr_vs_n(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    _draw_snr_vs_n(ax, df)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _draw_ratio_vs_n(ax, df: pd.DataFrame, *, ycol: str, ylabel: str, title: str) -> None:
    handles = []
    labels = []
    for score, sub in ordered_group_items(df):
        sub = sub.sort_values("n_cal")
        line, = ax.plot(sub["n_cal"], sub[ycol], marker="o")
        handles.append(line)
        labels.append(score)

    ref = ax.axhline(1.0, linestyle="--", linewidth=1.4, color="black")
    handles.append(ref)
    labels.append("ratio = 1 reference")

    style_axis(
        ax,
        xlabel="Calibration size",
        ylabel=ylabel,
        title=title,
    )
    ax.legend(handles, labels, frameon=False, loc="upper right")


def plot_ratio_vs_n(df: pd.DataFrame, ycol: str, ylabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    _draw_ratio_vs_n(ax, df, ycol=ycol, ylabel=ylabel, title=title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_detectability_4panel(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17.3, 13.2))
    axes = axes.ravel()

    _draw_floor_vs_n(axes[0], df)
    _draw_snr_vs_n(axes[1], df)
    _draw_ratio_vs_n(
        axes[2],
        df,
        ycol="floor_inflation_ratio",
        ylabel="mean empirical floor / true floor",
        title="C. Inflation of the observed floor",
    )
    _draw_ratio_vs_n(
        axes[3],
        df,
        ycol="sigma_inflation_ratio",
        ylabel=r"mean $\hat\sigma_\Delta$ / true $\sigma_\Delta$",
        title=r"D. Inflation of empirical heterogeneity $\hat\sigma_\Delta$",
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_normalized_collapse(sub_df: pd.DataFrame, full_df: pd.DataFrame, eta: float, out_path: Path) -> None:
    merged = sub_df.merge(
        full_df[["score", "K", "m_eff_density_proxy_at_q"]],
        on="score",
        how="left",
    )
    denom = merged["K"] * np.log(merged["K"] / eta)
    merged["normalized_n"] = (
        merged["n_cal"]
        * merged["m_eff_density_proxy_at_q"] ** 2
        * merged["sigma_delta_true_test_proxy"] ** 2
        / denom
    )

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    handles = []
    labels = []
    for score, sub in ordered_group_items(merged):
        sub = sub.sort_values("normalized_n")
        sub = sub[np.isfinite(sub["snr_true_floor_over_sd"])].copy()
        if len(sub) == 0:
            continue
        line, = ax.plot(sub["normalized_n"], sub["snr_true_floor_over_sd"], marker="o")
        handles.append(line)
        labels.append(score)
    thresh = ax.axhline(1.0, linestyle="--", linewidth=1.4, color="black")
    handles.append(thresh)
    labels.append("SNR = 1 threshold")

    style_axis(
        ax,
        xlabel=r"Normalized calibration size: $n m_{eff}(q)^2 \sigma_\Delta^2 / (K\log(K/\eta))$",
        ylabel=r"SNR = true floor / sd(empirical floor)",
        title="Approximate scaling collapse across score choices",
    )
    ax.legend(handles, labels, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze conformal result zip files and plot calibration-size floor detectability.")
    parser.add_argument(
        "--zip-pattern",
        type=str,
        default="result*.zip",
        help="Glob pattern used to find result zip files in the current directory.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="data_plot_output",
        help="Directory where CSVs and figures will be written.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override alpha. By default, uses primary_alpha from the zip metadata.",
    )
    parser.add_argument(
        "--n-grid",
        type=int,
        nargs="*",
        default=[200, 500, 1000, 2000, 4000, 7000, 15000, 20000, 25000, 28000],
        help="Calibration sizes used for stratified subsampling. Full calibration size is appended automatically.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=300,
        help="Number of subsampling repetitions for each calibration size except the full-size deterministic case.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for subsampling.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.05,
        help="Eta used in the normalized detectability scaling constant.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cwd = Path(".").resolve()
    zip_paths = sorted(cwd.glob(args.zip_pattern))
    if not zip_paths:
        raise FileNotFoundError(
            f"No zip files matched pattern {args.zip_pattern!r} in {cwd}. "
            "Place this script next to result_simple.zip / result_saps.zip / result_raps.zip."
        )

    outdir = cwd / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    bundles: List[ScoreBundle] = []
    for zip_path in zip_paths:
        try:
            bundle = ScoreBundle.from_zip(zip_path, alpha_override=args.alpha)
            bundles.append(bundle)
        except Exception as exc:
            print(f"[skip] {zip_path.name}: {exc}")

    if not bundles:
        raise RuntimeError("No valid result zip files were found.")

    print("Found zip files:")
    for b in bundles:
        print(f"  - {b.zip_path.name} -> score={b.score_name}, alpha={b.alpha}")

    full_rows = []
    subsample_frames = []

    for bundle in bundles:
        full = bundle.full_metrics()
        full_rows.append(full)
        n_grid = sorted(set([n for n in args.n_grid if n < len(bundle.cal_scores)] + [len(bundle.cal_scores)]))
        print(f"Running subsampling for {bundle.score_name}: n_grid={n_grid}, reps={args.reps}")
        sub_df = subsample_analysis(bundle, n_grid=n_grid, reps=args.reps, seed=args.seed)
        subsample_frames.append(sub_df)

    full_df = pd.DataFrame(full_rows).sort_values("score")
    subsample_df = pd.concat(subsample_frames, ignore_index=True).sort_values(["score", "n_cal"])
    detect_df = detectability_scaling(full_df, subsample_df, eta=args.eta).sort_values("score")

    full_df.to_csv(outdir / "full_metrics_summary.csv", index=False)
    subsample_df.to_csv(outdir / "subsample_summary.csv", index=False)
    detect_df.to_csv(outdir / "detectability_scaling.csv", index=False)

    config = {
        "zip_pattern": args.zip_pattern,
        "alpha_override": args.alpha,
        "n_grid_requested": args.n_grid,
        "reps": args.reps,
        "seed": args.seed,
        "eta": args.eta,
        "processed_scores": [b.score_name for b in bundles],
    }
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    plot_floor_vs_n(subsample_df, outdir / "figure_floor_vs_n.pdf")
    plot_snr_vs_n(subsample_df, outdir / "figure_snr_vs_n.pdf")
    plot_ratio_vs_n(
        subsample_df,
        ycol="floor_inflation_ratio",
        ylabel="mean empirical floor / true floor",
        title="Inflation of the observed floor at small calibration size",
        out_path=outdir / "figure_floor_inflation_vs_n.pdf",
    )
    plot_ratio_vs_n(
        subsample_df,
        ycol="sigma_inflation_ratio",
        ylabel=r"mean $\hat\sigma_\Delta$ / true $\sigma_\Delta$",
        title=r"Inflation of empirical heterogeneity $\hat\sigma_\Delta$",
        out_path=outdir / "figure_sigma_inflation_vs_n.pdf",
    )
    plot_detectability_4panel(subsample_df, outdir / "figure_detectability_4panel.pdf")
    plot_normalized_collapse(subsample_df, full_df, eta=args.eta, out_path=outdir / "figure_normalized_collapse_snr.pdf")

    # Human-readable console summary.
    print("\n=== Full-metric summary ===")
    cols = [
        "score",
        "q_true_test_proxy",
        "q_full_cal",
        "sigma_delta_true_test_proxy",
        "sigma_delta_full_cal",
        "true_floor_test_proxy",
        "observed_floor_full_cal_on_test",
        "m_eff_density_proxy_at_q",
    ]
    print(full_df[cols].to_string(index=False))

    print("\n=== Detectability summary (first n with SNR >= 1) ===")
    print(detect_df[["score", "n_detect_snr_ge_1", "scaled_n_detect_constant"]].to_string(index=False))

    print(f"\nSaved outputs to: {outdir}")


if __name__ == "__main__":
    main()
