"""Shared argparse helpers for CLIs built on BEST_KNOWN_CONFIG.

Extracted because run_stability_check.py, run_bracket.py, and
run_stability_bracket.py all need the same optional model-size override
flags: BEST_KNOWN_CONFIG hardcodes the "full" GPT2 architecture
(n_embd=768, n_layer=12, n_head=12) with no way to shrink it, which makes
these three experiments impractical to sanity-check on CPU (the transformer
forward/backward pass dominates cost, not the QAOA circuit itself, so
lowering --limit-qpu-call alone doesn't help). These flags are opt-in
overrides only — omitting them keeps BEST_KNOWN_CONFIG's model unchanged.
"""
import argparse
import dataclasses

from gqaoa.config import BestKnownConfig


def add_model_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--vocab-size", type=int, default=None,
                         help="Override BEST_KNOWN_CONFIG's model vocab_size (default: unchanged)")
    parser.add_argument("--n-embd", type=int, default=None,
                         help="Override BEST_KNOWN_CONFIG's model n_embd (default: unchanged)")
    parser.add_argument("--n-layer", type=int, default=None,
                         help="Override BEST_KNOWN_CONFIG's model n_layer (default: unchanged)")
    parser.add_argument("--n-head", type=int, default=None,
                         help="Override BEST_KNOWN_CONFIG's model n_head (default: unchanged)")


def apply_model_overrides(config: BestKnownConfig, args: argparse.Namespace) -> BestKnownConfig:
    overrides = {
        field: value
        for field, value in (
            ("vocab_size", args.vocab_size),
            ("n_embd", args.n_embd),
            ("n_layer", args.n_layer),
            ("n_head", args.n_head),
        )
        if value is not None
    }
    if not overrides:
        return config
    return dataclasses.replace(config, model=dataclasses.replace(config.model, **overrides))


def add_sdp_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-sdp", action="store_true",
        help="Skip the SDP compression preprocessing step (default: enabled, as in BEST_KNOWN_CONFIG)",
    )


def apply_sdp_override(config: BestKnownConfig, args: argparse.Namespace) -> BestKnownConfig:
    if not args.no_sdp:
        return config
    return dataclasses.replace(config, problem=dataclasses.replace(config.problem, sdp=False))
