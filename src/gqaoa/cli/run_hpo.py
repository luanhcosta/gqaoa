"""Optuna hyperparameter search over gqaoa_strategy — thin CLI wrapper."""
import argparse

from gqaoa.experiments.hpo import main as run_hpo_main


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--device-name", default="lightning.gpu")
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_hpo_main(n_trials=args.n_trials, device_name=args.device_name)


if __name__ == "__main__":
    main()
