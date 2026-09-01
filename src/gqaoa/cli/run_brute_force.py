"""Exact brute-force ground state of the QAOA cost Hamiltonian for a `ProblemInstance`.

Pure classical CPU search (see `gqaoa.domain.brute_force`): no MLflow, PennyLane,
or torch dependency, so it stays fast and runs in isolation from the quantum
strategies. The problem can either be loaded by id (`--problem-id`, previously
saved via `gqaoa.problem.save_problem`) or generated inline (synthetic or
yfinance-backed); an inline-generated problem is persisted via `save_problem`
before the search runs, so it is available for later comparisons.
"""
from __future__ import annotations

import argparse

from gqaoa.config import BEST_KNOWN_CONFIG
from gqaoa.domain.brute_force import brute_force_search
from gqaoa.problem import generate_synthetic_problem, generate_yfinance_problem, load_problem, save_problem
from gqaoa.problem.brute_force_store import save_brute_force_result

_DEFAULT_PROBLEM = BEST_KNOWN_CONFIG.problem


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--problem-id", default=None, help="Load a previously saved problem by id")

    synthetic_group = parser.add_argument_group("inline synthetic problem generation")
    synthetic_group.add_argument("--n-assets", type=int, default=None, help="Generate a synthetic problem with this many assets")
    synthetic_group.add_argument("--seed", type=int, default=None, help="Synthetic problem RNG seed (default: random)")
    synthetic_group.add_argument("--n-trading-days", type=int, default=None, help="Synthetic problem simulated trading days (default: generator default)")

    yfinance_group = parser.add_argument_group("inline yfinance problem generation")
    yfinance_group.add_argument("--tickers", nargs="+", default=None, help="Ticker symbols (space- or comma-separated) for a yfinance-backed problem")
    yfinance_group.add_argument("--start-date", default=None, help="yfinance start date (YYYY-MM-DD)")
    yfinance_group.add_argument("--end-date", default=None, help="yfinance end date (YYYY-MM-DD)")

    parser.add_argument("--topology", default="ring", choices=["ring", "complete"], help="Edge topology for an inline-generated problem (default: ring)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing saved problem/brute-force result with different content")

    parser.add_argument("--q", type=float, default=_DEFAULT_PROBLEM.q, help=f"Risk-aversion coefficient (default: {_DEFAULT_PROBLEM.q}, BEST_KNOWN_CONFIG)")
    parser.add_argument("--B", type=int, default=int(_DEFAULT_PROBLEM.B), help=f"Cardinality constraint (default: {int(_DEFAULT_PROBLEM.B)}, BEST_KNOWN_CONFIG)")
    parser.add_argument("--lamb", type=float, default=_DEFAULT_PROBLEM.lamb, help=f"Cardinality-penalty coefficient (default: {_DEFAULT_PROBLEM.lamb}, BEST_KNOWN_CONFIG)")
    parser.add_argument("--search-mode", choices=["full", "fixed-cardinality"], default="fixed-cardinality", help="Enumerate all bitstrings, or only those with exactly B ones (default: fixed-cardinality)")

    return parser


def _normalize_tickers(raw: list[str]) -> list[str]:
    if len(raw) == 1 and "," in raw[0]:
        return [t.strip() for t in raw[0].split(",") if t.strip()]
    return raw


def _resolve_problem(parser: argparse.ArgumentParser, args: argparse.Namespace):
    selectors = {
        "--problem-id": args.problem_id is not None,
        "--n-assets": args.n_assets is not None,
        "--tickers": args.tickers is not None,
    }
    chosen = [name for name, present in selectors.items() if present]

    if not chosen:
        parser.error(
            "one of --problem-id, --n-assets (synthetic), or --tickers (yfinance) is required"
        )
    if len(chosen) > 1:
        parser.error(
            f"options {chosen} are mutually exclusive: pick exactly one way to obtain the problem"
        )

    if args.problem_id is not None:
        return load_problem(args.problem_id)

    if args.n_assets is not None:
        kwargs = {"n_assets": args.n_assets, "seed": args.seed, "topology": args.topology}
        if args.n_trading_days is not None:
            kwargs["n_trading_days"] = args.n_trading_days
        instance = generate_synthetic_problem(**kwargs)
        save_problem(instance, overwrite=args.overwrite)
        return instance

    if not args.start_date or not args.end_date:
        parser.error("--tickers requires both --start-date and --end-date")
    tickers = _normalize_tickers(args.tickers)
    instance = generate_yfinance_problem(
        tickers=tickers, start_date=args.start_date, end_date=args.end_date, topology=args.topology
    )
    save_problem(instance, overwrite=args.overwrite)
    return instance


def main(argv=None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    instance = _resolve_problem(parser, args)

    result = brute_force_search(
        expected_value=instance.expected_value,
        cov_matrix=instance.cov_matrix,
        q=args.q,
        B=args.B,
        lamb=args.lamb,
        edges_hc=instance.edges_hc,
        n_assets=instance.n_assets,
        search_mode=args.search_mode,
    )

    save_brute_force_result(
        problem_id=instance.problem_id,
        q=args.q,
        B=args.B,
        lamb=args.lamb,
        result=result,
        overwrite=args.overwrite,
    )

    print(f"problem_id: {instance.problem_id}")
    print(f"search_mode: {result.search_mode}")
    print(f"optimal_bitstring: {result.optimal_bitstring}")
    print(f"optimal_energy: {result.optimal_energy}")
    print(f"n_candidates_evaluated: {result.n_candidates_evaluated}")
    print(f"runtime_seconds: {result.runtime_seconds:.6f}")


if __name__ == "__main__":
    main()
