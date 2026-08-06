"""Classical baseline benchmark: scipy.optimize.minimize, N runs."""
import argparse
import dataclasses

from gqaoa.config import BEST_KNOWN_CONFIG
from gqaoa.experiments.stability import run_stability


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument(
        "--minimize-method", default="COBYLA",
        help="scipy.optimize.minimize method, e.g. COBYLA, Nelder-Mead, Powell, "
             "CG, BFGS, L-BFGS-B, TNC, SLSQP, trust-constr",
    )
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
            depth=args.depth,
            limit_qpu_call=args.limit_qpu_call,
        ),
    )
    run_stability(
        strategy="scipy",
        base_config=config,
        n_runs=args.n_runs,
        experiment_name="gqaoa-benchmark-scipy",
        device_name=args.device_name,
        run_name_prefix=f"benchmark_scipy_{args.minimize_method}",
        strategy_kwargs={"minimize_method": args.minimize_method},
    )


if __name__ == "__main__":
    main()
