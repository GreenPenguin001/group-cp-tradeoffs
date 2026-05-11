from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .distributions import GroupDistribution, mixture_quantile

# compute RMS
def weighted_l2(values: Sequence[float], weights: Sequence[float]) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    return float(np.sqrt(np.sum(w * v * v)))

# compute heterogeneity: sigma_delta for calibration quantiles or sigma_lambda for coverage-calibrated set sizes
def sigma_weighted(values: Sequence[float], weights: Sequence[float]) -> float:
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mu = np.sum(w * x)
    return float(np.sqrt(np.sum(w * (x - mu) ** 2)))

# proxy of sensitivity quantity
def average_segment_density(dist: GroupDistribution, a: float, b: float) -> float:
    if np.isclose(a, b):
        return float(dist.pdf(a))
    return float(abs(dist.cdf(a) - dist.cdf(b)) / abs(a - b))

# proxy of set size quantity (monotone)
def linear_size_params(k: int) -> tuple[np.ndarray, np.ndarray]:
    a = 2.0 + 0.15 * np.arange(k, dtype=float)
    b = 0.8 + 0.1 * np.arange(k, dtype=float)
    return a, b


def ell_linear(t, a, b):
    return a + b * t


def sample_quantile(x: np.ndarray, p: float) -> float:
    return float(np.quantile(x, p, method="higher"))

# pooled-threshold distortion
def oracle_pooled_metrics(dists: Sequence[GroupDistribution], weights: Sequence[float], alpha: float):
    w = np.asarray(weights, dtype=float)
    q_g = np.array([d.ppf(1 - alpha) for d in dists], dtype=float)
    q = mixture_quantile(alpha, dists, w)
    eps = np.array([d.cdf(q) - (1 - alpha) for d in dists], dtype=float)
    seg = np.array([average_segment_density(d, q, qgi) for d, qgi in zip(dists, q_g)], dtype=float)
    sigma_delta = sigma_weighted(q_g, w)
    denom = max(np.sum(w * (q - q_g) ** 2), 1e-12)
    m_eff = float(np.sqrt(np.sum(w * (seg ** 2) * (q - q_g) ** 2) / denom))
    return {
        "q_pooled_oracle": q,
        "sigma_delta_oracle": sigma_delta,
        "distortion_l2_oracle": weighted_l2(eps, w),
        "m_eff_oracle": m_eff,
        "bound_oracle": m_eff * sigma_delta,
        "q_g_oracle": q_g,
    }

# groupwise threshold leads to groupwise coverage
# coverage and set size tradeoff
def oracle_coverage_to_size(dists: Sequence[GroupDistribution], weights: Sequence[float], alpha: float):
    base = oracle_pooled_metrics(dists, weights, alpha)
    q = base["q_pooled_oracle"]
    q_g = np.asarray(base["q_g_oracle"], dtype=float)
    w = np.asarray(weights, dtype=float)
    a, b = linear_size_params(len(dists))
    size_gap = ell_linear(q_g, a, b) - ell_linear(q, a, b)
    denom = max(np.sum(w * (q_g - q) ** 2), 1e-12)
    v_eff = float(np.sqrt(np.sum(w * (b ** 2) * (q_g - q) ** 2) / denom))
    return {
        "sigma_delta_oracle": base["sigma_delta_oracle"],
        "size_disparity_oracle": weighted_l2(size_gap, w),
        "v_eff_oracle": v_eff,
        "bound_oracle": v_eff * base["sigma_delta_oracle"],
    }

# equalize expected set size yields coverage distortion
def oracle_size_to_coverage(dists: Sequence[GroupDistribution], weights: Sequence[float], alpha: float):
    w = np.asarray(weights, dtype=float)
    q_g = np.array([d.ppf(1 - alpha) for d in dists], dtype=float)
    a, b = linear_size_params(len(dists))
    lambda_g = ell_linear(q_g, a, b)
    lambda_common = float(np.sum(w * lambda_g))
    tau_g = (lambda_common - a) / b
    cov_shift = np.array([d.cdf(tau) - d.cdf(qgi) for d, tau, qgi in zip(dists, tau_g, q_g)], dtype=float)
    seg = np.array([average_segment_density(d, tau, qgi) for d, tau, qgi in zip(dists, tau_g, q_g)], dtype=float)
    sigma_lambda = sigma_weighted(lambda_g, w)
    denom = max(np.sum(w * (lambda_g - lambda_common) ** 2), 1e-12)
    kappa_eff = float(np.sqrt(np.sum(w * ((seg / b) ** 2) * (lambda_g - lambda_common) ** 2) / denom))
    return {
        "sigma_lambda_oracle": sigma_lambda,
        "coverage_distortion_oracle": weighted_l2(cov_shift, w),
        "kappa_eff_oracle": kappa_eff,
        "bound_oracle": kappa_eff * sigma_lambda,
    }

# miscoverage in test set vs oracle lower bound
def mc_run_pooled_metrics(
    dists: Sequence[GroupDistribution],
    weights: Sequence[float],
    alpha: float,
    rng: np.random.Generator,
    n_cal: int,
    n_test: int,
):
    w = np.asarray(weights, dtype=float)
    cal = [d.rvs(n_cal, rng) for d in dists]
    test = [d.rvs(n_test, rng) for d in dists]
    q_g_hat = np.array([sample_quantile(x, 1 - alpha) for x in cal], dtype=float)
    pooled_cal = np.concatenate(cal)
    q_hat = sample_quantile(pooled_cal, 1 - alpha)
    coverage_hat = np.array([np.mean(x <= q_hat) for x in test], dtype=float)
    eps_hat = coverage_hat - (1 - alpha)
    sigma_delta_hat = sigma_weighted(q_g_hat, w)
    oracle = oracle_pooled_metrics(dists, weights, alpha)

    group_df = pd.DataFrame({
        "group": [d.name for d in dists],
        "weight": w,
        "q_g_hat": q_g_hat,
        "q_pooled_hat": q_hat,
        "coverage_hat": coverage_hat,
        "eps_hat": eps_hat,
        "q_g_oracle": oracle["q_g_oracle"],
        "q_pooled_oracle": oracle["q_pooled_oracle"],
    })
    summary = {
        "q_pooled_hat": q_hat,
        "sigma_delta_hat": sigma_delta_hat,
        "distortion_l2": weighted_l2(eps_hat, w),
        "distortion_l1": float(np.sum(w * np.abs(eps_hat))),
        "distortion_linf": float(np.max(np.abs(eps_hat))),
        "sigma_delta_oracle": oracle["sigma_delta_oracle"],
        "bound": oracle["bound_oracle"],
        "bound_oracle": oracle["bound_oracle"],
        "m_eff_oracle": oracle["m_eff_oracle"],
        "distortion_l2_oracle": oracle["distortion_l2_oracle"],
        "ratio_to_bound": weighted_l2(eps_hat, w) / max(oracle["bound_oracle"], 1e-12),
    }
    return group_df, summary

# size RMS (depending on calibration sample) vs oracle lower bound
def mc_run_coverage_to_size(dists, weights, alpha, rng, n_cal, n_test):
    pooled_df, pooled_summary = mc_run_pooled_metrics(dists, weights, alpha, rng, n_cal, n_test)
    q_hat = float(pooled_summary["q_pooled_hat"])
    q_g_hat = pooled_df["q_g_hat"].to_numpy(dtype=float)
    w = pooled_df["weight"].to_numpy(dtype=float)
    a, b = linear_size_params(len(dists))
    size_pooled = ell_linear(q_hat, a, b)
    size_groupwise = ell_linear(q_g_hat, a, b)
    size_gap = size_groupwise - size_pooled
    oracle = oracle_coverage_to_size(dists, weights, alpha)
    group_df = pooled_df.copy()
    group_df["a"] = a
    group_df["b"] = b
    group_df["size_pooled"] = size_pooled
    group_df["size_groupwise"] = size_groupwise
    group_df["size_gap"] = size_gap
    summary = {
        "sigma_delta_hat": pooled_summary["sigma_delta_hat"],
        "sigma_delta_oracle": oracle["sigma_delta_oracle"],
        "size_disparity": weighted_l2(size_gap, w),
        "v_eff_oracle": oracle["v_eff_oracle"],
        "bound": oracle["bound_oracle"],
        "bound_oracle": oracle["bound_oracle"],
        "size_disparity_oracle": oracle["size_disparity_oracle"],
        "ratio_to_bound": weighted_l2(size_gap, w) / max(oracle["bound_oracle"], 1e-12),
    }
    return group_df, summary

# coverage distortion of size-calibrated thresholds vs oracle lower bound
# LHS depends on calibrations quantiles and coverage disparity depends on the test set
def mc_run_size_to_coverage(dists, weights, alpha, rng, n_cal, n_test):
    w = np.asarray(weights, dtype=float)
    cal = [d.rvs(n_cal, rng) for d in dists]
    test = [d.rvs(n_test, rng) for d in dists]
    q_g_hat = np.array([sample_quantile(x, 1 - alpha) for x in cal], dtype=float)
    a, b = linear_size_params(len(dists))
    lambda_g_hat = ell_linear(q_g_hat, a, b)
    lambda_common_hat = float(np.sum(w * lambda_g_hat))
    tau_hat = (lambda_common_hat - a) / b
    coverage_at_qg = np.array([np.mean(x <= qg) for x, qg in zip(test, q_g_hat)], dtype=float)
    coverage_at_tau = np.array([np.mean(x <= t) for x, t in zip(test, tau_hat)], dtype=float)
    coverage_shift = coverage_at_tau - coverage_at_qg
    sigma_lambda_hat = sigma_weighted(lambda_g_hat, w)
    oracle = oracle_size_to_coverage(dists, weights, alpha)
    group_df = pd.DataFrame({
        "group": [d.name for d in dists],
        "weight": w,
        "q_g_hat": q_g_hat,
        "a": a,
        "b": b,
        "lambda_g_hat": lambda_g_hat,
        "lambda_common_hat": lambda_common_hat,
        "tau_hat": tau_hat,
        "coverage_shift": coverage_shift,
        "coverage_at_qg": coverage_at_qg,
        "coverage_at_tau": coverage_at_tau,
    })
    summary = {
        "sigma_lambda_hat": sigma_lambda_hat,
        "sigma_lambda_oracle": oracle["sigma_lambda_oracle"],
        "coverage_distortion": weighted_l2(coverage_shift, w),
        "kappa_eff_oracle": oracle["kappa_eff_oracle"],
        "bound": oracle["bound_oracle"],
        "bound_oracle": oracle["bound_oracle"],
        "coverage_distortion_oracle": oracle["coverage_distortion_oracle"],
        "ratio_to_bound": weighted_l2(coverage_shift, w) / max(oracle["bound_oracle"], 1e-12),
    }
    return group_df, summary
