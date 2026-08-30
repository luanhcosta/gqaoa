"""Bracket warm-restart strategy — single run (1000 QPU calls total)."""
import argparse

from gqaoa.cli._common import (
    add_model_override_args,
    add_sdp_override_args,
    apply_model_overrides,
    apply_sdp_override,
)
from gqaoa.config import BEST_KNOWN_CONFIG
from gqaoa.experiments.bracket import run_bracket


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-name", default="lightning.gpu")
    parser.add_argument("--no-cleanup-checkpoints", action="store_true",
                         help="Keep checkpoints/bracket/... after the run instead of deleting them")
    add_model_override_args(parser)
    add_sdp_override_args(parser)
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = apply_model_overrides(BEST_KNOWN_CONFIG, args)
    config = apply_sdp_override(config, args)
    run_bracket(
        n_repetitions=1,
        cleanup_checkpoints=not args.no_cleanup_checkpoints,
        device_name=args.device_name,
        base_config=config,
    )


if __name__ == "__main__":
    main()
