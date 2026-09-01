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


def add_problem_id_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--problem-id", default=None,
        help="Load a persisted problem (see gqaoa.problem.store.save_problem) by its "
             "problem_id and use it instead of the fixed 10-asset portfolio problem. "
             "Default: unchanged (fixed problem).",
    )


def apply_problem_id(config: BestKnownConfig, args: argparse.Namespace) -> BestKnownConfig:
    """Swap in a persisted problem when --problem-id is given; no-op otherwise.

    Also overrides problem.edges_hc/edges_hb with the loaded instance's own
    topology: gqaoa_strategy/gradient_descent_strategy/scipy_strategy all build
    their QAOA instance straight from `problem.edges_hc`/`problem.edges_hb`
    (build_problem() only returns expected_value/cov_matrix/lambda_sdp, not
    edges), so this is the only place that can correct the topology for a
    persisted problem whose asset count differs from the config's fixed
    RING_TOPOLOGY_EDGES (10 nodes).
    """
    if args.problem_id is None:
        return config
    from gqaoa.problem.store import load_problem

    instance = load_problem(args.problem_id)
    return dataclasses.replace(
        config,
        problem=dataclasses.replace(
            config.problem,
            problem_id=args.problem_id,
            edges_hc=list(instance.edges_hc),
            edges_hb=list(instance.edges_hb),
        ),
    )
