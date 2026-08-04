"""Optuna hyperparameter search over gqaoa_strategy — thin CLI wrapper."""
import argparse

from gqaoa.experiments.hpo import main as run_hpo_main


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--device-name", default="lightning.gpu")
    parser.add_argument("--limit-epochs", type=int, default=900)
    parser.add_argument("--limit-qpu-call", type=int, default=200)
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_hpo_main(
        n_trials=args.n_trials, device_name=args.device_name,
        limit_epochs=args.limit_epochs, limit_qpu_call=args.limit_qpu_call,
    )


if __name__ == "__main__":
    main()
