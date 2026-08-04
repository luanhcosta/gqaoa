"""Bracket warm-restart strategy repeated N times — measures its stability."""
import argparse

from gqaoa.cli._common import add_model_override_args, apply_model_overrides
from gqaoa.config import BEST_KNOWN_CONFIG
from gqaoa.experiments.bracket import run_bracket


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-repetitions", type=int, default=3)
    parser.add_argument("--device-name", default="lightning.gpu")
    parser.add_argument("--no-cleanup-checkpoints", action="store_true",
                         help="Keep checkpoints/bracket/... after each run instead of deleting them")
    add_model_override_args(parser)
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = apply_model_overrides(BEST_KNOWN_CONFIG, args)
    run_bracket(
        n_repetitions=args.n_repetitions,
        cleanup_checkpoints=not args.no_cleanup_checkpoints,
        device_name=args.device_name,
        base_config=config,
    )


if __name__ == "__main__":
    main()
