"""Bracket warm-restart strategy — single run (1000 QPU calls total)."""
import argparse

from gqaoa.experiments.bracket import run_bracket


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-name", default="lightning.gpu")
    parser.add_argument("--no-cleanup-checkpoints", action="store_true",
                         help="Keep checkpoints/bracket/... after the run instead of deleting them")
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_bracket(
        n_repetitions=1,
        cleanup_checkpoints=not args.no_cleanup_checkpoints,
        device_name=args.device_name,
    )


if __name__ == "__main__":
    main()
