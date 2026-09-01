"""Stability check for the best-known GQAOA config — main experiment.

Runs gqaoa_strategy.run_job N times and reports statistics on energy_min.
"""
import argparse
import dataclasses

from gqaoa.cli._common import (
    add_model_override_args,
    add_problem_id_arg,
    add_sdp_override_args,
    apply_model_overrides,
    apply_problem_id,
    apply_sdp_override,
)
from gqaoa.config import BEST_KNOWN_CONFIG, NO_ANNEAL_CONFIG
from gqaoa.experiments.stability import run_stability


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--limit-qpu-call", type=int, default=1000)
    parser.add_argument("--device-name", default="lightning.gpu")
    parser.add_argument(
        "--no-annealing", action="store_true",
        help="Disable beta_temp cosine annealing (ablation: uses NO_ANNEAL_CONFIG "
             "instead of BEST_KNOWN_CONFIG). Default: annealing enabled.",
    )
    add_model_override_args(parser)
    add_sdp_override_args(parser)
    add_problem_id_arg(parser)
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    base = NO_ANNEAL_CONFIG if args.no_annealing else BEST_KNOWN_CONFIG
    config = dataclasses.replace(
        base,
        training=dataclasses.replace(base.training, limit_qpu_call=args.limit_qpu_call),
    )
    config = apply_model_overrides(config, args)
    config = apply_sdp_override(config, args)
    config = apply_problem_id(config, args)
    anneal_label = "no_anneal" if args.no_annealing else "anneal"
    run_stability(
        strategy="gqaoa",
        base_config=config,
        n_runs=args.n_runs,
        experiment_name="gqaoa-stability",
        device_name=args.device_name,
        run_name_prefix=f"stability_{anneal_label}_{args.limit_qpu_call}",
    )


if __name__ == "__main__":
    main()
