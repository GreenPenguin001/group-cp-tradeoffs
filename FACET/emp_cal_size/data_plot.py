#!/usr/bin/env python3
"""
Process FACET conformal result zip files and generate the calibration-size
structural-floor detectability analysis.

Typical usage
-------------
Place this script next to results_simple.zip / results_saps.zip / results_raps.zip,
then run

    python data_plot_facet.py

or explicitly point to one or more zip files:

    python data_plot_facet.py --zip-files results_raps.zip

Main outputs (written to ./data_plot_facet_output by default)
-------------------------------------------------------------
  - full_metrics_summary.csv
  - group_structure_summary.csv
  - subsample_summary.csv
  - detectability_scaling.csv
  - dataset_meta_summary.csv
  - figure_floor_vs_n.pdf
  - figure_snr_vs_n.pdf
  - figure_floor_inflation_vs_n.pdf
  - figure_sigma_inflation_vs_n.pdf
  - figure_normalized_collapse_snr.pdf

Compared with the original MultiNLI version, this FACET version keeps the same
core mechanism but improves a few practical points:
  1) FACET-oriented default calibration-size grid.
  2) Better metadata extraction (group names, FACET variant, chosen temperature).
  3) Numeric log-scale x-axis ticks so the calibration sizes are explicitly shown.
  4) Direct --zip-files support in addition to glob patterns.
  5) More robust score-name inference from zip metadata.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "figure.dpi": 180,
        "savefig.dpi": 320,
        "font.size": 15,
        "axes.titlesize": 25,
        "axes.labelsize": 18,
        "xtick.labelsize": 13,
        "ytick.labelsize": 18,
        "legend.fontsize": 22,
        "lines.linewidth": 19.2,
        "lines.markersize": 16.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


SCORE_ORDER = {"simple": 0, "saps": 1, "raps": 2}
DEFAULT_FACET_N_GRID = [
    100,
    150,
    200,
    250,
    300,
    350,
    400,
    450,
    500,
    600,
    700,
    800,
    1000,
    1500,
    2000,
    3000,
    4000,
]


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def sanitize_name(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_")
    return text or "unknown"


def ordered_group_items(df: pd.DataFrame):
    items = list(df.groupby("score"))
    items.sort(key=lambda kv: SCORE_ORDER.get(str(kv[0]).lower(), 999))
    return items


def weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mean = float(np.sum(weights * values) / np.sum(weights))
    var = float(np.sum(weights * (values - mean) ** 2) / np.sum(weights))
    return math.sqrt(max(var, 0.0))


def conformal_index(n: int, alpha: float) -> int:
    # split conformal style: ceil((n + 1)(1 - alpha)) clamped to [1, n]
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    k = min(max(k, 1), n)
    return k - 1


def conformal_quantile(values: np.ndarray, alpha: float) -> float:
    values = np.asarray(values)
    idx = conformal_index(len(values), alpha)
    return float(np.partition(values, idx)[idx])


def quantile_from_sorted(sorted_values: np.ndarray, alpha: float) -> float:
    idx = conformal_index(len(sorted_values), alpha)
    return float(sorted_values[idx])


def local_density_estimate(sorted_values: np.ndarray, q: float) -> float:
    """Local density proxy around q using a symmetric count window."""
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


def apply_numeric_log_ticks(ax: plt.Axes, x_values: Sequence[int], rotate: int = 45) -> None:
    vals = sorted({int(x) for x in x_values if x is not None and np.isfinite(x) and x > 0})
    if not vals:
        return

    # Show every requested calibration size explicitly.
    ax.set_xticks(vals)
    ax.set_xticklabels([str(v) for v in vals], rotation=rotate, ha="right")
    ax.minorticks_off()


def maybe_save_png(fig: plt.Figure, pdf_path: Path, dpi: int, save_png: bool) -> None:
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    if save_png:
        fig.savefig(pdf_path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")


def find_member_by_suffix(names: Sequence[str], suffix: str) -> str:
    hits = [n for n in names if n.endswith(suffix)]
    if not hits:
        raise FileNotFoundError(f"Could not find {suffix!r} inside zip file.")
    return hits[0]


def infer_score_name(zip_path: Path, meta: Mapping[str, object]) -> str:
    lower_name = zip_path.name.lower()
    meta_score = str(meta.get("conformal_score", "")).strip().lower()
    if meta_score in {"simple", "saps", "raps"}:
        return "simple" if meta_score == "simple" else meta_score.upper()
    if "simple" in lower_name:
        return "simple"
    if "saps" in lower_name:
        return "SAPS"
    if "raps" in lower_name:
        return "RAPS"
    return zip_path.stem


def resolve_zip_paths(zip_files: Sequence[str], zip_pattern: str) -> List[Path]:
    paths: List[Path] = []
    if zip_files:
        for item in zip_files:
            p = Path(item).expanduser()
            if p.is_dir():
                paths.extend(sorted(p.glob("*.zip")))
            elif p.exists():
                paths.append(p.resolve())
            else:
                raise FileNotFoundError(f"Zip file not found: {item}")
        return sorted({p.resolve() for p in paths})

    cwd = Path(".").resolve()
    return sorted(cwd.glob(zip_pattern))


# -----------------------------------------------------------------------------
# main data container
# -----------------------------------------------------------------------------

@dataclass
class ScoreBundle:
    score_name: str
    zip_path: Path
    alpha: float
    dataset_name: str
    experiment_dataset: str
    group_ids: np.ndarray
    group_names: List[str]
    label_names: List[str]
    split_meta: Dict[str, object]
    cal_scores: np.ndarray
    test_scores: np.ndarray
    g_cal: np.ndarray
    g_test: np.ndarray
    test_group_probs: np.ndarray
    cal_group_probs: np.ndarray
    cal_scores_by_group: Dict[int, np.ndarray]
    sorted_test_by_group: Dict[int, np.ndarray]
    sorted_test_all: np.ndarray

    @classmethod
    def from_zip(cls, zip_path: Path, alpha_override: float | None = None) -> "ScoreBundle":
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()

            cal_scores = np.load(io.BytesIO(zf.read(find_member_by_suffix(names, "/arrays/cal_scores.npy"))))
            test_scores = np.load(io.BytesIO(zf.read(find_member_by_suffix(names, "/arrays/test_scores.npy"))))
            g_cal = np.load(io.BytesIO(zf.read(find_member_by_suffix(names, "/arrays/g_cal.npy"))))
            g_test = np.load(io.BytesIO(zf.read(find_member_by_suffix(names, "/arrays/g_test.npy"))))
            meta = json.loads(zf.read(find_member_by_suffix(names, "/split_meta.json")).decode("utf-8"))

        alpha = float(alpha_override) if alpha_override is not None else float(meta.get("primary_alpha", 0.1))
        score_name = infer_score_name(zip_path, meta)
        dataset_name = str(meta.get("dataset_name", "FACET"))
        experiment_dataset = str(meta.get("experiment_dataset", dataset_name))

        group_ids = np.array(sorted(set(np.unique(g_cal)).union(set(np.unique(g_test)))))
        raw_group_names = list(meta.get("group_names", []))
        if len(raw_group_names) < len(group_ids):
            raw_group_names = raw_group_names + [f"group_{g}" for g in group_ids[len(raw_group_names):]]
        group_names = [str(raw_group_names[int(g)]) if int(g) < len(raw_group_names) else f"group_{int(g)}" for g in group_ids]
        label_names = [str(x) for x in meta.get("label_names", [])]

        cal_scores_by_group = {int(g): np.asarray(cal_scores[g_cal == g], dtype=float) for g in group_ids}
        sorted_test_by_group = {int(g): np.sort(np.asarray(test_scores[g_test == g], dtype=float)) for g in group_ids}
        sorted_test_all = np.sort(np.asarray(test_scores, dtype=float))

        cal_group_probs = np.array([(g_cal == g).mean() for g in group_ids], dtype=float)
        test_group_probs = np.array([(g_test == g).mean() for g in group_ids], dtype=float)

        return cls(
            score_name=score_name,
            zip_path=zip_path.resolve(),
            alpha=alpha,
            dataset_name=dataset_name,
            experiment_dataset=experiment_dataset,
            group_ids=group_ids,
            group_names=group_names,
            label_names=label_names,
            split_meta=meta,
            cal_scores=np.asarray(cal_scores, dtype=float),
            test_scores=np.asarray(test_scores, dtype=float),
            g_cal=np.asarray(g_cal),
            g_test=np.asarray(g_test),
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
        floor = math.sqrt(float(np.sum(self.test_group_probs * eps ** 2)))
        return floor, cov, eps

    def groupwise_quantiles(self, score_map: Dict[int, np.ndarray], probs: np.ndarray) -> Tuple[np.ndarray, float]:
        qg = np.array([conformal_quantile(score_map[int(g)], self.alpha) for g in self.group_ids], dtype=float)
        sigma = weighted_std(qg, probs)
        return qg, sigma

    def full_metrics(self) -> Dict[str, float | int | str]:
        q_true = quantile_from_sorted(self.sorted_test_all, self.alpha)
        true_floor, _, _ = self.floor_on_test(q_true)
        q_full = conformal_quantile(self.cal_scores, self.alpha)
        obs_floor_full, _, _ = self.floor_on_test(q_full)
        qg_true = np.array(
            [quantile_from_sorted(self.sorted_test_by_group[int(g)], self.alpha) for g in self.group_ids],
            dtype=float,
        )
        sigma_true = weighted_std(qg_true, self.test_group_probs)
        qg_full, sigma_full = self.groupwise_quantiles(self.cal_scores_by_group, self.cal_group_probs)
        m_eff = local_density_estimate(self.sorted_test_all, q_true)

        out: Dict[str, float | int | str] = {
            "score": self.score_name,
            "zip_file": self.zip_path.name,
            "alpha": self.alpha,
            "dataset_name": self.dataset_name,
            "experiment_dataset": self.experiment_dataset,
            "K": int(len(self.group_ids)),
            "n_labels": int(len(self.label_names)),
            "n_cal_total": int(len(self.cal_scores)),
            "n_test_total": int(len(self.test_scores)),
            "q_true_test_proxy": q_true,
            "q_full_cal": q_full,
            "sigma_delta_true_test_proxy": sigma_true,
            "sigma_delta_full_cal": sigma_full,
            "true_floor_test_proxy": true_floor,
            "observed_floor_full_cal_on_test": obs_floor_full,
            "m_eff_density_proxy_at_q": m_eff,
        }

        for g, name, p_cal, p_test, q_true_g, q_full_g in zip(
            self.group_ids,
            self.group_names,
            self.cal_group_probs,
            self.test_group_probs,
            qg_true,
            qg_full,
        ):
            safe = sanitize_name(name)
            out[f"group_{int(g)}_name"] = name
            out[f"p_cal_group_{int(g)}"] = float(p_cal)
            out[f"p_test_group_{int(g)}"] = float(p_test)
            out[f"q_true_group_{int(g)}"] = float(q_true_g)
            out[f"q_full_group_{int(g)}"] = float(q_full_g)
            out[f"q_true_group_{int(g)}_{safe}"] = float(q_true_g)
            out[f"q_full_group_{int(g)}_{safe}"] = float(q_full_g)

        return out

    def group_structure_df(self) -> pd.DataFrame:
        rows = []
        for g, name, p_cal, p_test in zip(self.group_ids, self.group_names, self.cal_group_probs, self.test_group_probs):
            rows.append(
                {
                    "score": self.score_name,
                    "group": int(g),
                    "group_name": name,
                    "n_cal": int(np.sum(self.g_cal == g)),
                    "n_test": int(np.sum(self.g_test == g)),
                    "p_cal": float(p_cal),
                    "p_test": float(p_test),
                }
            )
        return pd.DataFrame(rows)

    def dataset_meta_row(self) -> Dict[str, object]:
        meta = self.split_meta
        return {
            "score": self.score_name,
            "zip_file": self.zip_path.name,
            "experiment_dataset": meta.get("experiment_dataset"),
            "dataset_name": meta.get("dataset_name"),
            "run_name": meta.get("run_name"),
            "conformal_score": meta.get("conformal_score"),
            "primary_alpha": meta.get("primary_alpha"),
            "alphas": json.dumps(meta.get("alphas", [])),
            "n_groups": len(meta.get("group_names", [])),
            "group_names": json.dumps(meta.get("group_names", [])),
            "n_labels": len(meta.get("label_names", [])),
            "facet_group_variant": meta.get("facet_group_variant"),
            "facet_split_stratify_mode": meta.get("facet_split_stratify_mode"),
            "facet_split_size_mode": meta.get("facet_split_size_mode"),
            "facet_image_mode": meta.get("facet_image_mode"),
            "temperature_sweep_group": meta.get("temperature_sweep_group"),
            "temperature_sweep_group_name": meta.get("temperature_sweep_group_name"),
            "temperature_sweep_values": json.dumps(meta.get("temperature_sweep_values", [])),
            "chosen_score_temperature": meta.get("chosen_score_temperature"),
            "auto_tune_score_temperature": meta.get("auto_tune_score_temperature"),
            "clip_model_name": meta.get("clip_model_name") or meta.get("model_name"),
            "split_sizes": json.dumps(meta.get("split_sizes", {})),
        }


# -----------------------------------------------------------------------------
# subsampling analysis
# -----------------------------------------------------------------------------

def subsample_analysis(bundle: ScoreBundle, n_grid: Sequence[int], reps: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    probs = bundle.cal_group_probs
    group_arrays = {int(g): np.asarray(bundle.cal_scores_by_group[int(g)], dtype=float) for g in bundle.group_ids}

    full = bundle.full_metrics()
    true_floor = float(full["true_floor_test_proxy"])
    sigma_true = float(full["sigma_delta_true_test_proxy"])
    full_n = len(bundle.cal_scores)

    rows: List[Dict[str, float | int | str]] = []
    for n_total in n_grid:
        if n_total > full_n:
            continue

        counts = make_stratified_counts(n_total, probs)
        floor_vals: List[float] = []
        sigma_vals: List[float] = []
        q_vals: List[float] = []

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

        snr = (true_floor / floor_sd) if floor_sd > 0 else np.inf
        rows.append(
            {
                "score": bundle.score_name,
                "dataset_name": bundle.dataset_name,
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
                "snr_true_floor_over_sd": snr,
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
        scaled_n_detect = (n_detect * (m_eff ** 2) * (sigma_true ** 2) / denom) if np.isfinite(n_detect) else np.nan
        rows.append(
            {
                "score": score,
                "dataset_name": r["dataset_name"],
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


# -----------------------------------------------------------------------------
# plotting
# -----------------------------------------------------------------------------

def plot_floor_vs_n(df: pd.DataFrame, out_path: Path, dpi: int, save_png: bool) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    handles = []
    labels = []

    all_x = []
    for score, sub in ordered_group_items(df):
        sub = sub.sort_values("n_cal")
        x = sub["n_cal"].to_numpy()
        y = sub["mean_floor_eval_on_test"].to_numpy()
        s = sub["sd_floor_eval_on_test"].to_numpy()
        tf = float(sub["true_floor_test_proxy"].iloc[0])
        all_x.extend(x.tolist())

        line, = ax.plot(x, y, marker="o", markersize=6.5, linewidth=2.4)
        color = line.get_color()
        ax.fill_between(x, np.maximum(y - s, 0.0), y + s, color=color, alpha=0.18)
        floor_line = ax.axhline(tf, linestyle="--", linewidth=1.3, color=color)
        handles.extend([line, floor_line])
        labels.extend([f"{score}: mean empirical floor", f"{score}: true-floor proxy"])

    ax.set_xscale("log")
    apply_numeric_log_ticks(ax, all_x)
    ax.tick_params(axis="x", labelsize=11.5)
    ax.tick_params(axis="y", labelsize=11.5)
    ax.set_xlabel("Calibration size")
    ax.set_ylabel("RMS group miscoverage on fixed test split")
    ax.set_title("FACET: empirical floor vs calibration size")
    ax.grid(True, alpha=0.3)
    ax.legend(handles, labels, frameon=False, fontsize=12.5, ncol=2, handlelength=2.3, columnspacing=1.2)
    fig.tight_layout()
    maybe_save_png(fig, out_path, dpi=dpi, save_png=save_png)
    plt.close(fig)


def plot_snr_vs_n(df: pd.DataFrame, out_path: Path, dpi: int, save_png: bool) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    handles = []
    labels = []
    all_x = []

    for score, sub in ordered_group_items(df):
        sub = sub.sort_values("n_cal")
        sub = sub[np.isfinite(sub["snr_true_floor_over_sd"])].copy()
        if len(sub) == 0:
            continue
        all_x.extend(sub["n_cal"].tolist())
        line, = ax.plot(sub["n_cal"], sub["snr_true_floor_over_sd"], marker="o", markersize=8.5, linewidth=4.4)
        handles.append(line)
        labels.append(score)

    thresh = ax.axhline(1.0, linestyle="--", linewidth=1.2, color="black")
    handles.append(thresh)
    labels.append("SNR = 1 threshold")

    ax.set_xscale("log")
    apply_numeric_log_ticks(ax, all_x)
    ax.tick_params(axis="x", labelsize=11.5)
    ax.tick_params(axis="y", labelsize=11.5)
    ax.set_xlabel("Calibration size")
    ax.set_ylabel(r"SNR = true floor / sd(empirical floor)")
    ax.set_title("FACET: detectability of the structural floor", fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.legend(handles, labels, frameon=False, fontsize=15, handlelength=2.3)
    fig.tight_layout()
    maybe_save_png(fig, out_path, dpi=dpi, save_png=save_png)
    plt.close(fig)


def plot_ratio_vs_n(
    df: pd.DataFrame,
    ycol: str,
    ylabel: str,
    title: str,
    out_path: Path,
    dpi: int,
    save_png: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    handles = []
    labels = []
    all_x = []

    for score, sub in ordered_group_items(df):
        sub = sub.sort_values("n_cal")
        all_x.extend(sub["n_cal"].tolist())
        line, = ax.plot(sub["n_cal"], sub[ycol], marker="o", markersize=6.5, linewidth=2.4)
        handles.append(line)
        labels.append(score)

    ref = ax.axhline(1.0, linestyle="--", linewidth=1.2, color="black")
    handles.append(ref)
    labels.append("ratio = 1 reference")

    ax.set_xscale("log")
    apply_numeric_log_ticks(ax, all_x)
    ax.tick_params(axis="x", labelsize=11.5)
    ax.tick_params(axis="y", labelsize=11.5)
    ax.set_xlabel("Calibration size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(handles, labels, frameon=False, fontsize=12.5, handlelength=2.3)
    fig.tight_layout()
    maybe_save_png(fig, out_path, dpi=dpi, save_png=save_png)
    plt.close(fig)


def plot_normalized_collapse(
    sub_df: pd.DataFrame,
    full_df: pd.DataFrame,
    eta: float,
    out_path: Path,
    dpi: int,
    save_png: bool,
) -> None:
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

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    handles = []
    labels = []

    for score, sub in ordered_group_items(merged):
        sub = sub.sort_values("normalized_n")
        sub = sub[np.isfinite(sub["snr_true_floor_over_sd"])].copy()
        if len(sub) == 0:
            continue
        line, = ax.plot(sub["normalized_n"], sub["snr_true_floor_over_sd"], marker="o", markersize=6.5, linewidth=2.4)
        handles.append(line)
        labels.append(score)

    thresh = ax.axhline(1.0, linestyle="--", linewidth=1.2, color="black")
    handles.append(thresh)
    labels.append("SNR = 1 threshold")

    ax.set_xscale("log")
    ax.tick_params(axis="x", labelsize=11.5)
    ax.tick_params(axis="y", labelsize=11.5)
    ax.set_xlabel(r"Normalized calibration size: $n m_{\mathrm{eff}}(q)^2 \sigma_\Delta^2 / (K\log(K/\eta))$")
    ax.set_ylabel(r"SNR = true floor / sd(empirical floor)")
    ax.set_title("FACET: approximate scaling collapse across score choices")
    ax.grid(True, alpha=0.3)
    ax.legend(handles, labels, frameon=False, fontsize=12.5, handlelength=2.3)
    fig.tight_layout()
    maybe_save_png(fig, out_path, dpi=dpi, save_png=save_png)
    plt.close(fig)


# -----------------------------------------------------------------------------
# cli
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze FACET conformal result zip files and plot calibration-size structural-floor detectability."
    )
    parser.add_argument(
        "--zip-files",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit zip files. If omitted, --zip-pattern is used in the current directory.",
    )
    parser.add_argument(
        "--zip-pattern",
        type=str,
        default="results*.zip",
        help="Glob pattern used to find FACET result zip files in the current directory when --zip-files is not given.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="data_plot_facet_output",
        help="Directory where CSVs and figures will be written.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override alpha. By default, uses primary_alpha from each zip metadata.",
    )
    parser.add_argument(
        "--n-grid",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Calibration sizes used for stratified subsampling. "
            "If omitted, this FACET script uses a denser default grid around the detectability region."
        ),
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
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI for saved outputs.",
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Also save PNG copies next to the PDF figures.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    zip_paths = resolve_zip_paths(args.zip_files or [], args.zip_pattern)
    if not zip_paths:
        cwd = Path(".").resolve()
        raise FileNotFoundError(
            f"No zip files matched. Checked explicit --zip-files or pattern {args.zip_pattern!r} in {cwd}."
        )

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    bundles: List[ScoreBundle] = []
    for zip_path in zip_paths:
        try:
            bundle = ScoreBundle.from_zip(zip_path, alpha_override=args.alpha)
            bundles.append(bundle)
        except Exception as exc:
            print(f"[skip] {zip_path.name}: {exc}")

    if not bundles:
        raise RuntimeError("No valid FACET result zip files were found.")

    print("Found zip files:")
    for b in bundles:
        print(
            f"  - {b.zip_path.name} -> score={b.score_name}, dataset={b.dataset_name}, "
            f"alpha={b.alpha}, n_cal={len(b.cal_scores)}, groups={b.group_names}"
        )

    n_grid_requested = args.n_grid if args.n_grid else DEFAULT_FACET_N_GRID

    full_rows = []
    group_rows = []
    meta_rows = []
    subsample_frames = []

    for bundle in bundles:
        full_rows.append(bundle.full_metrics())
        group_rows.append(bundle.group_structure_df())
        meta_rows.append(bundle.dataset_meta_row())

        n_grid = sorted({int(n) for n in n_grid_requested if int(n) < len(bundle.cal_scores)} | {len(bundle.cal_scores)})
        print(f"Running subsampling for {bundle.score_name}: n_grid={n_grid}, reps={args.reps}")
        sub_df = subsample_analysis(bundle, n_grid=n_grid, reps=args.reps, seed=args.seed)
        subsample_frames.append(sub_df)

    full_df = pd.DataFrame(full_rows).sort_values("score")
    group_df = pd.concat(group_rows, ignore_index=True).sort_values(["score", "group"])
    meta_df = pd.DataFrame(meta_rows).sort_values("score")
    subsample_df = pd.concat(subsample_frames, ignore_index=True).sort_values(["score", "n_cal"])
    detect_df = detectability_scaling(full_df, subsample_df, eta=args.eta).sort_values("score")

    full_df.to_csv(outdir / "full_metrics_summary.csv", index=False)
    group_df.to_csv(outdir / "group_structure_summary.csv", index=False)
    meta_df.to_csv(outdir / "dataset_meta_summary.csv", index=False)
    subsample_df.to_csv(outdir / "subsample_summary.csv", index=False)
    detect_df.to_csv(outdir / "detectability_scaling.csv", index=False)

    config = {
        "zip_files": [str(p) for p in zip_paths],
        "zip_pattern": args.zip_pattern,
        "alpha_override": args.alpha,
        "n_grid_requested": list(n_grid_requested),
        "reps": args.reps,
        "seed": args.seed,
        "eta": args.eta,
        "dpi": args.dpi,
        "save_png": bool(args.save_png),
        "processed_scores": [b.score_name for b in bundles],
    }
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    plot_floor_vs_n(subsample_df, outdir / "figure_floor_vs_n.pdf", dpi=args.dpi, save_png=args.save_png)
    plot_snr_vs_n(subsample_df, outdir / "figure_snr_vs_n.pdf", dpi=args.dpi, save_png=args.save_png)
    plot_ratio_vs_n(
        subsample_df,
        ycol="floor_inflation_ratio",
        ylabel="mean empirical floor / true floor",
        title="FACET: inflation of the observed floor at small calibration size",
        out_path=outdir / "figure_floor_inflation_vs_n.pdf",
        dpi=args.dpi,
        save_png=args.save_png,
    )
    plot_ratio_vs_n(
        subsample_df,
        ycol="sigma_inflation_ratio",
        ylabel=r"mean $\hat\sigma_\Delta$ / true $\sigma_\Delta$",
        title=r"FACET: inflation of empirical heterogeneity $\hat\sigma_\Delta$",
        out_path=outdir / "figure_sigma_inflation_vs_n.pdf",
        dpi=args.dpi,
        save_png=args.save_png,
    )
    plot_normalized_collapse(
        subsample_df,
        full_df,
        eta=args.eta,
        out_path=outdir / "figure_normalized_collapse_snr.pdf",
        dpi=args.dpi,
        save_png=args.save_png,
    )

    print("\n=== Full-metric summary ===")
    cols = [
        "score",
        "dataset_name",
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
