"""Stability check for the best-known GQAOA config — main experiment.

Runs gqaoa_strategy.run_job N times and reports statistics on energy_min.
"""
import argparse
import dataclasses

from gqaoa.cli._common import add_model_override_args, apply_model_overrides
from gqaoa.config import BEST_KNOWN_CONFIG
from gqaoa.experiments.stability import run_stability


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--limit-qpu-call", type=int, default=1000)
    parser.add_argument("--device-name", default="lightning.gpu")
    add_model_override_args(parser)
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = dataclasses.replace(
        BEST_KNOWN_CONFIG,
        training=dataclasses.replace(BEST_KNOWN_CONFIG.training, limit_qpu_call=args.limit_qpu_call),
    )
    config = apply_model_overrides(config, args)
    run_stability(
        strategy="gqaoa",
        base_config=config,
        n_runs=args.n_runs,
        experiment_name="gqaoa-stability",
        device_name=args.device_name,
        run_name_prefix=f"stability_anneal_{args.limit_qpu_call}",
    )


if __name__ == "__main__":
    main()
