from __future__ import annotations

import argparse

from .experiments import (
    experiment_coverage_to_size,
    experiment_coverage_to_size_hard,
    experiment_multigroup,
    experiment_size_to_coverage,
    experiment_size_to_coverage_hard,
    experiment_two_group,
    run_paper,
)


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo synthetic uncertainty-relation experiments")
    parser.add_argument(
        "--experiment",
        choices=[
            "two_group",
            "multigroup_gaussian",
            "multigroup_wide",
            "multigroup_gamma",
            "multigroup_t",
            "coverage_to_size",
            "coverage_to_size_hard",
            "size_to_coverage",
            "size_to_coverage_hard",
            "paper",
        ],
        default="paper",
    )
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--n_cal", type=int, default=400)
    parser.add_argument("--n_test", type=int, default=4000)
    parser.add_argument("--outdir", type=str, default="outputs_mc_paper")
    args = parser.parse_args()

    if args.experiment == "two_group":
        experiment_two_group(args.outdir, args.alpha, args.seeds, args.n_cal, args.n_test)
    elif args.experiment == "multigroup_gaussian":
        experiment_multigroup(args.outdir, args.alpha, args.seeds, "gaussian", args.n_cal, args.n_test)
    elif args.experiment == "multigroup_wide":
        experiment_multigroup(args.outdir, args.alpha, args.seeds, "wide", args.n_cal, args.n_test)
    elif args.experiment == "multigroup_gamma":
        experiment_multigroup(args.outdir, args.alpha, args.seeds, "gamma", args.n_cal, args.n_test)
    elif args.experiment == "multigroup_t":
        experiment_multigroup(args.outdir, args.alpha, args.seeds, "t", args.n_cal, args.n_test)
    elif args.experiment == "coverage_to_size":
        experiment_coverage_to_size(args.outdir, args.alpha, args.seeds, args.n_cal, args.n_test)
    elif args.experiment == "coverage_to_size_hard":
        experiment_coverage_to_size_hard(args.outdir, args.alpha, args.seeds, args.n_cal, args.n_test)
    elif args.experiment == "size_to_coverage":
        experiment_size_to_coverage(args.outdir, args.alpha, args.seeds, args.n_cal, args.n_test)
    elif args.experiment == "size_to_coverage_hard":
        experiment_size_to_coverage_hard(args.outdir, args.alpha, args.seeds, args.n_cal, args.n_test)
    else:
        run_paper(args.outdir, args.alpha, args.seeds, args.n_cal, args.n_test)


if __name__ == "__main__":
    main()
