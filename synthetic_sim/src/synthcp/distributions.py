from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats
from scipy.optimize import brentq

class MixtureDistribution:
    def __init__(self, components, weights):
        self.components = list(components)
        w = np.asarray(weights, dtype=float)
        self.weights = w / np.sum(w)

    def cdf(self, x):
        return float(sum(w * comp.cdf(x) for w, comp in zip(self.weights, self.components)))

    def pdf(self, x):
        return float(sum(w * comp.pdf(x) for w, comp in zip(self.weights, self.components)))

    def ppf(self, u: float):
        lo = min(comp.ppf(1e-6) for comp in self.components) - 5.0
        hi = max(comp.ppf(1 - 1e-6) for comp in self.components) + 5.0
        return float(brentq(lambda z: self.cdf(z) - u, lo, hi))

    # rvs is for sampling purposes.
    def rvs(self, size: int, random_state):
        rng = random_state
        idx = rng.choice(len(self.components), size=size, p=self.weights)
        out = np.empty(size, dtype=float)
        for i, comp in enumerate(self.components):
            mask = idx == i
            if np.any(mask):
                out[mask] = comp.rvs(size=int(np.sum(mask)), random_state=rng)
        return out


@dataclass(frozen=True)
class GroupDistribution:
    name: str
    dist: object

    def cdf(self, x):
        return self.dist.cdf(x)

    def pdf(self, x):
        return self.dist.pdf(x)

    def ppf(self, u: float) -> float:
        return float(self.dist.ppf(u))

    def rvs(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return np.asarray(self.dist.rvs(size=size, random_state=rng), dtype=float)


# Simulation for two group case with Gaussian distributions, where the difference in means is controlled by the delta parameter.
def make_two_group_gaussian(delta: float) -> list[GroupDistribution]:
    return [
        GroupDistribution("g0", stats.norm(loc=0.0, scale=1.0)),
        GroupDistribution("g1", stats.norm(loc=delta, scale=1.0)),
    ]


# multiple group case with Gaussian distributions, where the difference in means is controlled by the scale parameter.
def make_multigroup_gaussian(scale: float) -> list[GroupDistribution]:
    mus = scale * np.array([-1.5, -0.5, 0.5, 1.5], dtype=float)
    return [GroupDistribution(f"g{i}", stats.norm(loc=float(mu), scale=1.0)) for i, mu in enumerate(mus)]


# multiple group case with Gaussian distributions, where the difference in variances is controlled by the scale parameter.
def make_multigroup_wide_gaussian(scale: float) -> list[GroupDistribution]:
    scales = 1.0 + scale * np.array([0.0, 0.2, 0.4, 0.6], dtype=float)
    return [GroupDistribution(f"g{i}", stats.norm(loc=0.0, scale=float(sd))) for i, sd in enumerate(scales)]


# introduce skewness by using gamma distributions, where the difference in scales is controlled by the scale parameter.
def make_multigroup_gamma(scale: float) -> list[GroupDistribution]:
    scales = 0.55 + scale * np.array([0.00, 0.06, 0.12, 0.18], dtype=float)
    return [GroupDistribution(f"g{i}", stats.gamma(a=4.0, loc=0.0, scale=float(sc))) for i, sc in enumerate(scales)]


# introduce heavy tails by using t-distributions, where the difference in scales is controlled by the scale parameter.
def make_multigroup_t_heavy(scale: float) -> list[GroupDistribution]:
    scales = 0.9 + scale * np.array([0.0, 0.12, 0.24, 0.36], dtype=float)
    return [GroupDistribution(f"g{i}", stats.t(df=6, loc=0.0, scale=float(sc))) for i, sc in enumerate(scales)]


# 8 asymmetric gaussian groups with different means and variances,
# where the difference in means and variances is controlled by the scale parameter.
def make_multigroup_coverage_to_size_hard(scale: float) -> list[GroupDistribution]:
    centers = scale * np.array([-1.85, -1.35, -0.95, -0.55, 0.05, 0.55, 1.05, 1.65], dtype=float)
    left_offsets = np.array([-1.30, -1.10, -1.55, -0.85, -0.60, -1.25, -0.75, -1.45], dtype=float)
    right_offsets = np.array([0.30, 0.65, 0.10, 0.85, 1.10, 0.55, 1.25, 0.95], dtype=float)
    left_scales = np.array([0.24, 0.34, 0.28, 0.38, 0.22, 0.30, 0.26, 0.35], dtype=float)
    right_scales = np.array([1.45, 0.95, 1.25, 0.78, 1.55, 0.92, 1.38, 1.08], dtype=float)
    weights = [
        [0.14, 0.86],
        [0.34, 0.66],
        [0.22, 0.78],
        [0.47, 0.53],
        [0.58, 0.42],
        [0.29, 0.71],
        [0.62, 0.38],
        [0.18, 0.82],
    ]
    out = []
    for i, c in enumerate(centers):
        components = [
            stats.norm(loc=float(c + left_offsets[i]), scale=float(left_scales[i])),
            stats.norm(loc=float(c + right_offsets[i]), scale=float(right_scales[i])),
        ]
        out.append(GroupDistribution(f"g{i}", MixtureDistribution(components, weights[i])))
    return out


# eight asymmetric groups with different means, variances, and tail behaviors
# one gaussian and one t-distribution component
'''
def make_multigroup_size_to_coverage_hard(scale: float) -> list[GroupDistribution]:
    centers = scale * np.array(
        [-2.15, -1.55, -1.10, -0.60, -0.05, 0.55, 1.15, 1.95],
        dtype=float,
    )

    # heavier and more heterogeneous tails
    dfs = [3, 5, 4, 7, 3, 6, 4, 8]

    # pull the left t component further left
    t_offsets = np.array(
        [-1.55, -1.25, -1.45, -1.00, -1.20, -0.90, -1.10, -1.45],
        dtype=float,
    )

    # move the right Gaussian further right to create a stronger shoulder / gap
    n_offsets = np.array(
        [1.10, 0.95, 1.25, 1.00, 1.30, 0.92, 1.42, 1.10],
        dtype=float,
    )

    # narrow some t components so the left mass is concentrated
    t_scales = np.array(
        [0.18, 0.27, 0.21, 0.30, 0.17, 0.28, 0.20, 0.31],
        dtype=float,
    )

    # make the Gaussian components moderately wide, but not too dominant
    n_scales = np.array(
        [1.20, 0.92, 1.10, 0.82, 1.28, 0.88, 1.18, 0.98],
        dtype=float,
    )

    # put more groups near the sensitive shoulder regime
    # [weight on t, weight on Gaussian]
    mix_weights = [
        [0.82, 0.18],
        [0.74, 0.26],
        [0.68, 0.32],
        [0.60, 0.40],
        [0.72, 0.28],
        [0.58, 0.42],
        [0.66, 0.34],
        [0.78, 0.22],
    ]

    out = []
    for i, c in enumerate(centers):
        components = [
            stats.t(df=dfs[i], loc=float(c + t_offsets[i]), scale=float(t_scales[i])),
            stats.norm(loc=float(c + n_offsets[i]), scale=float(n_scales[i])),
        ]
        out.append(GroupDistribution(f"g{i}", MixtureDistribution(components, mix_weights[i])))
    return out
'''
def make_multigroup_size_to_coverage_hard(scale: float) -> list[GroupDistribution]:
    centers = scale * np.array([-2.00, -1.45, -1.00, -0.55, 0.00, 0.55, 1.10, 1.75], dtype=float)
    dfs = [4, 6, 5, 8, 4, 7, 5, 9]
    t_offsets = np.array([-1.35, -1.05, -1.25, -0.85, -1.10, -0.70, -0.95, -1.30], dtype=float)
    n_offsets = np.array([0.95, 0.70, 1.10, 0.88, 1.18, 0.62, 1.28, 0.98], dtype=float)
    t_scales = np.array([0.22, 0.32, 0.25, 0.36, 0.20, 0.33, 0.24, 0.37], dtype=float)
    n_scales = np.array([1.60, 1.08, 1.42, 0.92, 1.72, 0.98, 1.55, 1.22], dtype=float)
    mix_weights = [
        [0.10, 0.90],
        [0.26, 0.74],
        [0.18, 0.82],
        [0.42, 0.58],
        [0.52, 0.48],
        [0.28, 0.72],
        [0.60, 0.40],
        [0.12, 0.88],
    ]
    out = []
    for i, c in enumerate(centers):
        components = [
            stats.t(df=dfs[i], loc=float(c + t_offsets[i]), scale=float(t_scales[i])),
            stats.norm(loc=float(c + n_offsets[i]), scale=float(n_scales[i])),
        ]
        out.append(GroupDistribution(f"g{i}", MixtureDistribution(components, mix_weights[i])))
    return out


# F_mix is the CDF of the mixture distribution, which is a weighted sum of the CDFs of the component distributions.
def mixture_cdf(x: float, dists: Sequence[GroupDistribution], weights: Sequence[float]) -> float:
    return float(sum(w * d.cdf(x) for w, d in zip(weights, dists)))


# q_mix is the quantile function of the mixture distribution,
# which can be computed by finding the value of x such that F_mix(x) = 1 - alpha.
def mixture_quantile(alpha: float, dists: Sequence[GroupDistribution], weights: Sequence[float]) -> float:
    target = 1.0 - alpha
    lo = min(d.ppf(1e-6) for d in dists) - 2.0
    hi = max(d.ppf(1 - 1e-6) for d in dists) + 2.0
    return float(brentq(lambda x: mixture_cdf(x, dists, weights) - target, lo, hi))
