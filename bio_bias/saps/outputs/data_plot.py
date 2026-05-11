
from __future__ import annotations

import argparse
import json
import math
import tempfile
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

METHOD = "saps"
METHOD_DISPLAY = "SAPS"
DEFAULT_TEMPERATURE = 1.0
DEFAULT_LAMBDA = 0.2
DEFAULT_RANDOMIZE = False
DEFAULT_RANDOM_SEED = 0
DEFAULT_K_REG = 5

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

plt.rcParams.update(
    {
        "figure.dpi": 180,
        "savefig.dpi": 400,
        "font.size": 20.0,
        "axes.titlesize": 19.0,
        "axes.labelsize": 19.0,
        "xtick.labelsize": 19.5,
        "ytick.labelsize": 20.5,
        "legend.fontsize": 16.5,
        "lines.linewidth": 3.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "mathtext.fontset": "dejavusans",
    }
)


def _looks_like_result_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "split_meta.json").exists() and any(path.glob("seed_*")):
        return True
    if any((child / "split_meta.json").exists() for child in path.glob("seed_*")):
        return True
    return False


def _extract_zip(path: Path) -> tuple[Path, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory(prefix="bias_bios_extract_")
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(tmp.name)
    entries = list(Path(tmp.name).iterdir())
    extracted = entries[0] if len(entries) == 1 else Path(tmp.name)
    return extracted.resolve(), tmp


def discover_result_root(script_dir: Path, user_result_root: str | None = None) -> tuple[Path, tempfile.TemporaryDirectory | None]:
    if user_result_root:
        root = Path(user_result_root).expanduser().resolve()
        if root.is_dir():
            return root, None
        if root.is_file() and root.suffix == ".zip":
            return _extract_zip(root)
        raise FileNotFoundError(f"Could not use --result-root={user_result_root}")

    script_dir = script_dir.resolve()
    candidates: list[Path] = []
    search_dirs = [script_dir, script_dir.parent, Path.cwd().resolve(), Path.cwd().resolve().parent]
    seen: set[Path] = set()

    method_tokens = [f"bias_in_bios_{METHOD}", f"{METHOD}_full_primary", METHOD]

    for base in search_dirs:
        if base in seen or not base.exists():
            continue
        seen.add(base)

        if _looks_like_result_root(base):
            return base, None

        for child in base.iterdir():
            name = child.name.lower()
            if child.is_dir() and _looks_like_result_root(child):
                if any(tok in name for tok in method_tokens):
                    return child.resolve(), None
                candidates.append(child.resolve())
            if child.is_file() and child.suffix == ".zip":
                if any(tok in name for tok in method_tokens):
                    return _extract_zip(child.resolve())
                candidates.append(child.resolve())

    for cand in candidates:
        name = cand.name.lower()
        if "bias_in_bios" in name:
            if cand.is_dir():
                return cand.resolve(), None
            if cand.is_file() and cand.suffix == ".zip":
                return _extract_zip(cand.resolve())

    raise FileNotFoundError(
        "Could not find a Bias-in-Bios result folder or zip. "
        "Put plot.py near the result folder, or pass --result-root explicitly."
    )


def load_arrays(seed_dir: Path) -> dict[str, np.ndarray]:
    arrays_dir = seed_dir / "arrays"
    out = {}
    for stem in ["probs_cal", "probs_test", "y_cal", "y_test", "g_cal", "g_test"]:
        path = arrays_dir / f"{stem}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing array file: {path}")
        out[stem] = np.load(path)
    return out


def standardize_group_name(name: str) -> str:
    raw = str(name).strip()
    lower = raw.lower()
    if lower == "male":
        return "Male"
    if lower == "female":
        return "Female"
    if raw == "0":
        return "Male"
    if raw == "1":
        return "Female"
    return raw.title() if raw else raw


def get_group_name(g: int, group_names: list[str]) -> str:
    if 0 <= int(g) < len(group_names):
        return standardize_group_name(group_names[int(g)])
    return f"Group {g}"


def get_group_color(g: int) -> str:
    return PALETTE[int(g) % len(PALETTE)]


def lighten_color(color: str, factor: float = 0.62) -> str:
    rgb = np.array(mcolors.to_rgb(color))
    mixed = rgb + (1.0 - rgb) * float(factor)
    return mcolors.to_hex(np.clip(mixed, 0.0, 1.0))


def style_axis(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, alpha=0.18, linewidth=0.7)
    ax.set_axisbelow(True)


def apply_temperature_global(probs: np.ndarray, T: float, eps: float = 1e-12) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    safe = np.clip(probs, eps, 1.0)
    if abs(T - 1.0) < 1e-12:
        return safe / safe.sum(axis=1, keepdims=True)
    powered = safe ** (1.0 / float(T))
    return powered / powered.sum(axis=1, keepdims=True)


def random_u(n: int, randomize: bool, random_seed: int) -> np.ndarray:
    if randomize:
        rng = np.random.default_rng(int(random_seed))
        return rng.random(n)
    return np.full(n, 0.5, dtype=float)


def all_label_scores(
    probs: np.ndarray,
    *,
    temperature: float,
    randomize: bool,
    random_seed: int,
    lam: float,
    k_reg: int,
) -> np.ndarray:
    probs = apply_temperature_global(probs, temperature)
    probs = np.asarray(probs, dtype=float)
    n, K = probs.shape
    u = random_u(n, randomize, random_seed).reshape(-1, 1)

    order = np.argsort(-probs, axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(n)[:, None], order] = np.arange(1, K + 1)

    if METHOD == "raps":
        sorted_probs = np.take_along_axis(probs, order, axis=1)
        cumsum = np.cumsum(sorted_probs, axis=1)
        rho_sorted = np.concatenate([np.zeros((n, 1)), cumsum[:, :-1]], axis=1)
        rho = rho_sorted[np.arange(n)[:, None], ranks - 1]
        return rho + u * probs + float(lam) * np.maximum(ranks - int(k_reg), 0)

    if METHOD == "saps":
        pmax = probs.max(axis=1, keepdims=True)
        base = pmax + float(lam) * (ranks - 2 + u)
        top = u * probs
        return np.where(ranks == 1, top, base)

    raise ValueError(f"Unsupported method: {METHOD}")


def true_label_scores(scores_all: np.ndarray, y: np.ndarray) -> np.ndarray:
    return scores_all[np.arange(len(y)), y]


def split_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    k = int(math.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.partition(scores, k - 1)[k - 1])


def group_thresholds(scores: np.ndarray, groups: np.ndarray, alpha: float) -> dict[int, float]:
    out = {}
    for g in sorted(np.unique(groups)):
        g = int(g)
        out[g] = split_conformal_quantile(scores[groups == g], alpha)
    return out


def empirical_cdf_at_threshold(scores_group: np.ndarray, threshold: float) -> float:
    return float(np.mean(scores_group <= threshold))


def empirical_group_cdf(scores: np.ndarray, groups: np.ndarray, thresholds: dict[int, float] | float) -> dict[int, float]:
    out = {}
    for g in sorted(np.unique(groups)):
        g = int(g)
        t = thresholds[g] if isinstance(thresholds, dict) else float(thresholds)
        out[g] = empirical_cdf_at_threshold(scores[groups == g], t)
    return out


def average_set_size_at_threshold(scores_all: np.ndarray, threshold: float) -> float:
    return float(np.mean((scores_all <= threshold).sum(axis=1)))


def average_group_set_size(scores_all: np.ndarray, groups: np.ndarray, thresholds: dict[int, float] | float) -> dict[int, float]:
    out = {}
    for g in sorted(np.unique(groups)):
        g = int(g)
        t = thresholds[g] if isinstance(thresholds, dict) else float(thresholds)
        out[g] = average_set_size_at_threshold(scores_all[groups == g], t)
    return out


def empirical_ecdf(scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(scores)
    ys = np.arange(1, len(xs) + 1, dtype=float) / len(xs)
    return xs, ys


def size_curve_from_scores(scores_all_group: np.ndarray, grid: np.ndarray) -> np.ndarray:
    cutoffs = np.sort(scores_all_group.reshape(-1))
    n_examples = max(scores_all_group.shape[0], 1)
    return np.searchsorted(cutoffs, grid, side="right") / n_examples


def group_weights(groups: np.ndarray) -> dict[int, float]:
    values, counts = np.unique(groups, return_counts=True)
    counts = counts.astype(float)
    counts /= counts.sum()
    return {int(v): float(c) for v, c in zip(values, counts)}


def weighted_mean(values_by_group: dict[int, float], weights_by_group: dict[int, float]) -> float:
    return float(sum(weights_by_group[g] * values_by_group[g] for g in values_by_group))


def find_tau_from_size_curve(grid: np.ndarray, size_curve: np.ndarray, lambda_target: float) -> float:
    idx = int(np.argmin(np.abs(size_curve - lambda_target)))
    return float(grid[idx])


def compute_metrics(
    probs_cal: np.ndarray,
    y_cal: np.ndarray,
    g_cal: np.ndarray,
    probs_test: np.ndarray,
    y_test: np.ndarray,
    g_test: np.ndarray,
    alpha: float,
    *,
    temperature: float,
    randomize: bool,
    random_seed: int,
    lam: float,
    k_reg: int,
    grid_step: float = 0.001,
) -> dict:
    cal_scores_all = all_label_scores(
        probs_cal,
        temperature=temperature,
        randomize=randomize,
        random_seed=random_seed,
        lam=lam,
        k_reg=k_reg,
    )
    test_scores_all = all_label_scores(
        probs_test,
        temperature=temperature,
        randomize=randomize,
        random_seed=random_seed,
        lam=lam,
        k_reg=k_reg,
    )

    cal_scores = true_label_scores(cal_scores_all, y_cal)
    test_scores = true_label_scores(test_scores_all, y_test)

    q = split_conformal_quantile(cal_scores, alpha)
    q_g = group_thresholds(cal_scores, g_cal, alpha)

    F_pooled = empirical_group_cdf(test_scores, g_test, q)
    F_groupwise = empirical_group_cdf(test_scores, g_test, q_g)
    eps_pooled = {g: F_pooled[g] - (1.0 - alpha) for g in F_pooled}

    l_pooled = average_group_set_size(test_scores_all, g_test, q)
    l_groupwise = average_group_set_size(test_scores_all, g_test, q_g)
    delta_size = {g: l_groupwise[g] - l_pooled[g] for g in l_groupwise}

    weights = group_weights(g_test)
    lambda_target = weighted_mean(l_groupwise, weights)

    max_score_grid = float(np.max(test_scores_all))
    n_points = int(max_score_grid / max(grid_step, 1e-9)) + 1
    if n_points > 5001:
        grid = np.linspace(0.0, max_score_grid, num=5001, dtype=float)
    else:
        grid = np.arange(0.0, max_score_grid + grid_step / 2.0, grid_step)

    l_curve = {}
    tau_g = {}
    l_equalized = {}
    F_equalized = {}
    delta_cov = {}

    for g in sorted(np.unique(g_test)):
        g = int(g)
        scores_group_all = test_scores_all[g_test == g]
        l_curve[g] = size_curve_from_scores(scores_group_all, grid)
        tau = find_tau_from_size_curve(grid, l_curve[g], lambda_target)
        tau_g[g] = tau
        l_equalized[g] = average_set_size_at_threshold(scores_group_all, tau)
        F_equalized[g] = empirical_cdf_at_threshold(test_scores[g_test == g], tau)
        delta_cov[g] = F_equalized[g] - F_groupwise[g]

    q_bar = weighted_mean(q_g, weights)
    sigma_delta = float(np.sqrt(weighted_mean({g: (q_g[g] - q_bar) ** 2 for g in q_g}, weights)))
    sigma_lambda = float(np.sqrt(weighted_mean({g: (l_groupwise[g] - lambda_target) ** 2 for g in l_groupwise}, weights)))

    rms_cov_pooled = float(np.sqrt(weighted_mean({g: eps_pooled[g] ** 2 for g in eps_pooled}, weights)))
    rms_size_groupwise = float(np.sqrt(weighted_mean({g: delta_size[g] ** 2 for g in delta_size}, weights)))
    rms_cov_equalized = float(np.sqrt(weighted_mean({g: delta_cov[g] ** 2 for g in delta_cov}, weights)))

    return {
        "alpha": float(alpha),
        "cal_scores": cal_scores,
        "test_scores": test_scores,
        "q": q,
        "q_g": q_g,
        "F_pooled": F_pooled,
        "F_groupwise": F_groupwise,
        "eps_pooled": eps_pooled,
        "l_pooled": l_pooled,
        "l_groupwise": l_groupwise,
        "delta_size": delta_size,
        "lambda_target": lambda_target,
        "grid": grid,
        "l_curve": l_curve,
        "tau_g": tau_g,
        "l_equalized": l_equalized,
        "F_equalized": F_equalized,
        "delta_cov": delta_cov,
        "weights": weights,
        "sigma_delta": sigma_delta,
        "sigma_lambda": sigma_lambda,
        "rms_cov_pooled": rms_cov_pooled,
        "rms_size_groupwise": rms_size_groupwise,
        "rms_cov_equalized": rms_cov_equalized,
        "temperature": float(temperature),
        "lambda_penalty": float(lam),
        "randomize": bool(randomize),
        "random_seed": int(random_seed),
        "k_reg": int(k_reg),
        "method": METHOD_DISPLAY,
    }


def make_group_handles(groups: list[int], group_names: list[str]) -> list[Line2D]:
    return [
        Line2D([0], [0], color=get_group_color(g), lw=2.2, label=get_group_name(g, group_names))
        for g in groups
    ]


def add_vertical_math_label(ax, x: float, text: str, color: str, side: str = "left", y: float = 0.98) -> None:
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    xpad = -3 if side == "left" else 3
    ha = "right" if side == "left" else "left"
    ax.annotate(
        text,
        xy=(x, y),
        xycoords=trans,
        xytext=(xpad, 0),
        textcoords="offset points",
        rotation=90,
        va="bottom",
        ha=ha,
        color=color,
        fontsize=18.0,
        bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.85),
        clip_on=False,
        zorder=6,
    )


def add_curve_reference_points_B1(ax, metrics: dict, groups: list[int]) -> None:
    q = metrics["q"]
    q_g = metrics["q_g"]
    for g in groups:
        color = get_group_color(g)
        ax.scatter([q], [metrics["l_pooled"][g]], s=28, color=color, edgecolor="white", linewidth=0.7, zorder=5)
        ax.scatter([q_g[g]], [metrics["l_groupwise"][g]], s=34, marker="D", color=color, edgecolor="white", linewidth=0.7, zorder=5)


def add_curve_reference_points_C1(ax, metrics: dict, groups: list[int]) -> None:
    tau_g = metrics["tau_g"]
    lam = metrics["lambda_target"]
    for g in groups:
        color = get_group_color(g)
        ax.scatter([tau_g[g]], [lam], s=34, marker="o", color=color, edgecolor="white", linewidth=0.7, zorder=5)


def _score_axis_label() -> str:
    return f"True-label {METHOD_DISPLAY} nonconformity score"


def draw_panel_A1(ax, metrics: dict, g_cal: np.ndarray, group_names: list[str]) -> None:
    q = metrics["q"]
    q_g = metrics["q_g"]
    alpha = metrics["alpha"]
    target = 1.0 - alpha

    groups = sorted(np.unique(g_cal))
    for g in groups:
        g = int(g)
        xs, ys = empirical_ecdf(metrics["cal_scores"][g_cal == g])
        ax.step(xs, ys, where="post", color=get_group_color(g), label=get_group_name(g, group_names))
        ax.axvline(q_g[g], color=get_group_color(g), linestyle="--", linewidth=2.6)

    ax.axvline(q, color="black", linewidth=1.9, label="Pooled threshold")
    ax.axhline(target, color="black", linestyle=":", linewidth=1.4, alpha=0.85, label=f"Target coverage = {target:.2f}")
    style_axis(ax, grid_axis="both")
    xmax = max(float(metrics["cal_scores"].max()), q, max(q_g.values())) + 0.03
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(0.0, 1.01)
    ax.set_xlabel(_score_axis_label())
    ax.set_ylabel("Empirical CDF")
    ax.set_title("A. Calibration ECDFs and thresholds")

    all_q = [q] + [q_g[int(g)] for g in groups]
    x0 = max(0.0, min(all_q) - 0.03)
    x1 = max(all_q) + 0.03
    y0 = max(0.0, target - 0.05)
    y1 = min(1.0, target + 0.05)

    axins = inset_axes(ax, width="40%", height="44%", loc="lower left", bbox_to_anchor=(0.06, 0.05, 1, 1), bbox_transform=ax.transAxes, borderpad=0.7)
    for g in groups:
        g = int(g)
        xs, ys = empirical_ecdf(metrics["cal_scores"][g_cal == g])
        axins.step(xs, ys, where="post", color=get_group_color(g))
        axins.axvline(q_g[g], color=get_group_color(g), linestyle="--", linewidth=2.8)
    axins.axvline(q, color="black", linewidth=1.25)
    axins.axhline(target, color="black", linestyle=":", linewidth=1.0, alpha=0.85)
    axins.set_xlim(x0, x1)
    axins.set_ylim(y0, y1)
    axins.grid(True, axis="both", alpha=0.14, linewidth=0.55)
    for spine in axins.spines.values():
        spine.set_linewidth(0.8)
    mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec="0.6", lw=0.6)
    axins.set_xticks([])
    axins.set_yticks([])

    ax.legend(frameon=False, loc="lower right")


def _local_threshold_window(values: list[float], margin: float = 0.018) -> tuple[float, float]:
    return max(0.0, min(values) - margin), max(values) + margin


def draw_panel_B1(ax, metrics: dict, group_names: list[str]) -> None:
    q = metrics["q"]
    q_g = metrics["q_g"]
    groups = sorted(metrics["q_g"])
    x0, x1 = _local_threshold_window([q] + [q_g[g] for g in groups])

    y_refs = []
    for g in groups:
        y_refs.extend([metrics["l_pooled"][g], metrics["l_groupwise"][g]])

    for g in groups:
        ax.plot(metrics["grid"], metrics["l_curve"][g], color=get_group_color(g), label=get_group_name(g, group_names))
        ax.axvline(q_g[g], color=get_group_color(g), linestyle="--", linewidth=2.6)

    ax.axvline(q, color="black", linewidth=1.9, label="Pooled threshold")
    add_curve_reference_points_B1(ax, metrics, groups)
    ax.set_xlim(x0, x1)
    ax.set_ylim(min(y_refs) - 0.004, max(y_refs) + 0.004)
    style_axis(ax, grid_axis="both")
    ax.set_xlabel(r"Threshold $t$")
    ax.set_ylabel("Expected set size")
    ax.set_title("B. Size curves near pooled and group thresholds")

    for g in groups:
        add_vertical_math_label(ax, q_g[g], rf"$q_{{{g}}}$", get_group_color(g), side="right", y=0.08)
    add_vertical_math_label(ax, q, r"$q$", "black", side="right", y=0.08)

    handles = [
        *make_group_handles(groups, group_names),
        Line2D([0], [0], color="black", lw=1.9, label="Pooled threshold"),
        Line2D([0], [0], marker="o", color="0.55", markerfacecolor="0.55", markersize=6.0, lw=0, label="Point at pooled threshold"),
        Line2D([0], [0], marker="D", color="0.55", markerfacecolor="0.55", markersize=6.0, lw=0, label="Point at group threshold"),
    ]
    ax.legend(handles=handles, frameon=True, facecolor="white", edgecolor="0.85", framealpha=0.96, loc="upper left", borderpad=0.28, labelspacing=0.28, handletextpad=0.45)


def draw_panel_C1(ax, metrics: dict, group_names: list[str]) -> None:
    q = metrics["q"]
    q_g = metrics["q_g"]
    tau_g = metrics["tau_g"]
    lam = metrics["lambda_target"]
    groups = sorted(q_g)
    x0, x1 = _local_threshold_window([q] + [q_g[g] for g in groups] + [tau_g[g] for g in groups], margin=0.018)

    y_refs = [lam]
    for g in groups:
        y_refs.extend([metrics["l_groupwise"][g], metrics["l_equalized"][g]])

    for g in groups:
        ax.plot(metrics["grid"], metrics["l_curve"][g], color=get_group_color(g), label=get_group_name(g, group_names))
        ax.axvline(q_g[g], color=get_group_color(g), linestyle="--", linewidth=2.6)
        ax.axvline(tau_g[g], color=get_group_color(g), linestyle=":", linewidth=2.6)

    ax.axvline(q, color="black", linewidth=1.9, label="Pooled threshold")
    ax.axhline(lam, color="black", linestyle="-.", linewidth=1.55, label="Common target size")
    add_curve_reference_points_C1(ax, metrics, groups)
    ax.set_xlim(x0, x1)
    ax.set_ylim(min(y_refs) - 0.004, max(y_refs) + 0.004)
    style_axis(ax, grid_axis="both")
    ax.set_xlabel(r"Threshold $t$")
    ax.set_ylabel("Expected set size")
    ax.set_title("C. Equalized-size target and induced thresholds")

    for j, g in enumerate(groups):
        add_vertical_math_label(ax, q_g[g], rf"$q_{{{g}}}$", get_group_color(g), side="right", y=0.08)
        tau_side = "left" if j % 2 == 0 else "right"
        add_vertical_math_label(ax, tau_g[g], rf"$\tau_{{{g}}}$", get_group_color(g), side=tau_side, y=0.18)
    add_vertical_math_label(ax, q, r"$q$", "black", side="right", y=0.28)

    handles = [
        *make_group_handles(groups, group_names),
        Line2D([0], [0], color="black", lw=1.9, label="Pooled threshold"),
        Line2D([0], [0], color="0.35", lw=1.2, linestyle="--", label="Group thresholds"),
        Line2D([0], [0], color="black", lw=1.55, linestyle="-.", label="Common target size"),
        Line2D([0], [0], marker="o", color="0.55", markerfacecolor="0.55", markersize=6.0, lw=0, label="Equalized-size points"),
    ]
    ax.legend(handles=handles, frameon=True, facecolor="white", edgecolor="0.85", framealpha=0.96, loc="upper left", borderpad=0.28, labelspacing=0.28, handletextpad=0.45)


def draw_panel_D(ax, metrics: dict, group_names: list[str]) -> None:
    groups = sorted(metrics["eps_pooled"])
    x = np.arange(len(groups), dtype=float)
    width = 0.22

    vals1 = [metrics["eps_pooled"][g] for g in groups]
    vals2 = [metrics["delta_size"][g] for g in groups]
    vals3 = [metrics["delta_cov"][g] for g in groups]
    base1 = "#4E79A7"
    base2 = "#E15759"
    base3 = "#59A14F"

    face1 = [base1 if v >= 0 else lighten_color(base1) for v in vals1]
    face2 = [base2 if v >= 0 else lighten_color(base2) for v in vals2]
    face3 = [base3 if v >= 0 else lighten_color(base3) for v in vals3]

    ax.bar(x - width, vals1, width=width, color=face1, edgecolor=base1, linewidth=1.0)
    ax.bar(x, vals2, width=width, color=face2, edgecolor=base2, linewidth=1.0)
    ax.bar(x + width, vals3, width=width, color=face3, edgecolor=base3, linewidth=1.0)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    style_axis(ax, grid_axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels([get_group_name(g, group_names) for g in groups])
    ax.set_ylabel("Distortion value")
    ax.set_title("D. Group-level distortions")
    handles = [
        Patch(facecolor=base1, edgecolor=base1, label="Pooled coverage distortion"),
        Patch(facecolor=base2, edgecolor=base2, label="Set-size distortion"),
        Patch(facecolor=base3, edgecolor=base3, label="Coverage distortion after size equalization"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left")


def _annotate_bars(ax, bars, values: list[float]) -> None:
    max_abs = max(max(abs(v) for v in values), 1e-8)
    pad = 0.10 * max_abs
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            v + (pad if v >= 0 else -pad),
            f"{v:+.4f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=18.0,
        )


def draw_panel_B2(ax, metrics: dict, group_names: list[str]) -> None:
    groups = sorted(metrics["delta_size"])
    labels = [get_group_name(g, group_names) for g in groups]
    vals = [metrics["delta_size"][g] for g in groups]
    bars = ax.bar(labels, vals, color=[get_group_color(g) for g in groups], width=0.62)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    style_axis(ax, grid_axis="y")
    ax.set_ylabel(r"$\ell_g(q_g)-\ell_g(q)$")
    ax.set_title(r"B2. Set-size change after group-wise calibration")
    _annotate_bars(ax, bars, vals)
    lim = max(max(abs(v) for v in vals), 1e-8) * 1.32
    ax.set_ylim(-lim, lim)


def draw_panel_C2(ax, metrics: dict, group_names: list[str]) -> None:
    groups = sorted(metrics["delta_cov"])
    labels = [get_group_name(g, group_names) for g in groups]
    vals = [metrics["delta_cov"][g] for g in groups]
    bars = ax.bar(labels, vals, color=[get_group_color(g) for g in groups], width=0.62)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    style_axis(ax, grid_axis="y")
    ax.set_ylabel(r"$\widehat F_g(\tau_g)-\widehat F_g(q_g)$")
    ax.set_title(r"C2. Coverage change after equalized-size adjustment")
    _annotate_bars(ax, bars, vals)
    lim = max(max(abs(v) for v in vals), 1e-8) * 1.32
    ax.set_ylim(-lim, lim)


def save_panel(draw_fn, path: Path, metrics: dict, *extra_args) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    draw_fn(ax, metrics, *extra_args)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.14, top=0.96)
    fig.savefig(path)
    plt.close(fig)


def assemble_main_figure(metrics: dict, g_cal: np.ndarray, group_names: list[str], outpath: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17.3, 12.2))
    axes = axes.ravel()

    draw_panel_A1(axes[0], metrics, g_cal, group_names)
    draw_panel_B1(axes[1], metrics, group_names)
    draw_panel_C1(axes[2], metrics, group_names)
    draw_panel_D(axes[3], metrics, group_names)

    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.07, top=0.935, wspace=0.24, hspace=0.32)
    fig.savefig(outpath)
    plt.close(fig)


def write_manifest(outdir: Path, metrics: dict) -> None:
    lines = [
        f"method,{metrics['method']}",
        f"alpha,{metrics['alpha']}",
        f"q,{metrics['q']:.10f}",
        f"sigma_delta,{metrics['sigma_delta']:.10f}",
        f"lambda_target,{metrics['lambda_target']:.10f}",
        f"sigma_lambda,{metrics['sigma_lambda']:.10f}",
        f"rms_cov_pooled,{metrics['rms_cov_pooled']:.10f}",
        f"rms_size_groupwise,{metrics['rms_size_groupwise']:.10f}",
        f"rms_cov_equalized,{metrics['rms_cov_equalized']:.10f}",
        f"temperature,{metrics['temperature']:.10f}",
        f"lambda_penalty,{metrics['lambda_penalty']:.10f}",
        f"randomize,{metrics['randomize']}",
        f"random_seed,{metrics['random_seed']}",
    ]
    if METHOD == "raps":
        lines.append(f"k_reg,{metrics['k_reg']}")
    (outdir / "biobias_plot_manifest.csv").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Redraw BioBias experiment figures directly from arrays/tables using {METHOD_DISPLAY} scores.")
    parser.add_argument("--result-root", type=str, default=None, help="Path to the result directory or zip.")
    parser.add_argument("--seed", type=int, default=4, help="Seed id to use (default: 4).")
    parser.add_argument("--alpha", type=float, default=None, help="Alpha value. Defaults to split_meta primary_alpha.")
    parser.add_argument("--grid-step", type=float, default=0.001, help="Threshold grid step for set-size curves.")
    parser.add_argument("--outdir", type=str, default=".", help="Directory to save the pdf outputs. Default: current directory.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help=f"Temperature used before scoring (default: {DEFAULT_TEMPERATURE}).")
    parser.add_argument("--lambda-penalty", type=float, default=DEFAULT_LAMBDA, help=f"Penalty parameter for {METHOD_DISPLAY} (default: {DEFAULT_LAMBDA}).")
    if METHOD == "raps":
        parser.add_argument("--k-reg", type=int, default=DEFAULT_K_REG, help=f"RAPS k_reg (default: {DEFAULT_K_REG}).")
    parser.add_argument("--randomize", action="store_true", help="Use random u ~ Uniform[0,1] instead of fixed u=0.5.")
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED, help=f"Random seed when --randomize is used (default: {DEFAULT_RANDOM_SEED}).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    result_root, tmp = discover_result_root(script_dir, args.result_root)
    try:
        seed_dir = result_root / f"seed_{args.seed}"
        if not seed_dir.exists():
            raise FileNotFoundError(f"Could not find {seed_dir}")

        split_meta_path = seed_dir / "split_meta.json"
        with open(split_meta_path, "r", encoding="utf-8") as f:
            split_meta = json.load(f)

        alpha = float(args.alpha) if args.alpha is not None else float(split_meta["primary_alpha"])
        group_names = [standardize_group_name(x) for x in split_meta.get("gender_names", ["0", "1"])]

        arrays = load_arrays(seed_dir)
        metrics = compute_metrics(
            probs_cal=arrays["probs_cal"],
            y_cal=arrays["y_cal"],
            g_cal=arrays["g_cal"],
            probs_test=arrays["probs_test"],
            y_test=arrays["y_test"],
            g_test=arrays["g_test"],
            alpha=alpha,
            temperature=float(args.temperature),
            randomize=bool(args.randomize),
            random_seed=int(args.random_seed),
            lam=float(args.lambda_penalty),
            k_reg=int(getattr(args, "k_reg", DEFAULT_K_REG)),
            grid_step=args.grid_step,
        )

        assemble_main_figure(metrics=metrics, g_cal=arrays["g_cal"], group_names=group_names, outpath=outdir / "biobias_primary_alpha_4panel_redraw.pdf")
        save_panel(draw_panel_A1, outdir / "A1_grouped_score_distributions_redraw.pdf", metrics, arrays["g_cal"], group_names)
        save_panel(draw_panel_B1, outdir / "B1_groupwise_setsize_curves_redraw.pdf", metrics, group_names)
        save_panel(draw_panel_C1, outdir / "C1_setsize_curves_common_target_redraw.pdf", metrics, group_names)
        save_panel(draw_panel_D, outdir / "D_group_distortion_terms_redraw.pdf", metrics, group_names)

        write_manifest(outdir, metrics)
        print(f"Saved {METHOD_DISPLAY} figures to: {outdir}")
        for p in sorted(outdir.glob("*redraw.pdf")):
            print(" -", p.name)
    finally:
        if tmp is not None:
            tmp.cleanup()


if __name__ == "__main__":
    main()
