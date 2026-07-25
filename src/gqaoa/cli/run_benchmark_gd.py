"""Classical baseline benchmark: PennyLane GradientDescentOptimizer + SPSA, N runs."""
import argparse
import dataclasses

from gqaoa.config import BEST_KNOWN_CONFIG
from gqaoa.experiments.stability import run_stability


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--optimizer-stepsize", type=float, default=0.01)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--limit-qpu-call", type=int, default=1000)
    parser.add_argument("--device-name", default="lightning.gpu")
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = dataclasses.replace(
        BEST_KNOWN_CONFIG,
        training=dataclasses.replace(
            BEST_KNOWN_CONFIG.training,
            optimizer_lr=args.optimizer_stepsize,
            depth=args.depth,
            limit_qpu_call=args.limit_qpu_call,
        ),
    )
    # gqaoa-benchmark-gd is a distinct experiment name from gqaoa-stability,
    # fixing the old benchmark_gd.py bug of sharing stability_check.py's experiment.
    run_stability(
        strategy="gradient_descent",
        base_config=config,
        n_runs=args.n_runs,
        experiment_name="gqaoa-benchmark-gd",
        device_name=args.device_name,
        run_name_prefix="benchmark_gd",
    )


if __name__ == "__main__":
    main()
