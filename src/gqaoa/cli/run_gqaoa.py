"""Single GQAOA dev-run — thin CLI wrapper around gqaoa_strategy.run_job."""
import argparse

from gqaoa.cli._common import add_problem_id_arg, apply_problem_id
from gqaoa.config import (
    DEFAULT_PROBLEM_ID,
    BestKnownConfig,
    ModelConfig,
    ProblemConfig,
    RING_TOPOLOGY_EDGES,
    TrainingConfig,
)
from gqaoa.reporting.optimality import compare_single_run_to_optimal, try_load_brute_force
from gqaoa.strategies import gqaoa_strategy
from gqaoa.tracking.mlflow_utils import init_mlflow


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-lr", type=float, default=1e-5)
    parser.add_argument("--vocab-size", type=int, default=10)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--beta-temp", type=float, default=1.0)
    parser.add_argument("--limit-epochs", type=int, default=900)
    parser.add_argument("--limit-qpu-call", type=int, default=900)
    parser.add_argument("--run-name", default="gqaoa")
    parser.add_argument("--device-name", default="lightning.gpu")
    parser.add_argument("--no-sdp", action="store_true",
                         help="Skip the SDP compression preprocessing step (default: enabled)")
    add_problem_id_arg(parser)
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)

    init_mlflow("gqaoa-dev")

    problem = ProblemConfig(
        q=0.3, B=5, lamb=0,
        initial_state="dicke_state", mixture_layer="xy",
        edges_hc=RING_TOPOLOGY_EDGES, edges_hb=RING_TOPOLOGY_EDGES,
        sdp=not args.no_sdp,
        problem_id=DEFAULT_PROBLEM_ID,
    )
    model = ModelConfig(vocab_size=args.vocab_size, n_layer=args.n_layer)
    training = TrainingConfig(
        depth=args.depth,
        optimizer_lr=args.optimizer_lr,
        beta_temp=args.beta_temp,
        limit_epochs=args.limit_epochs,
        limit_qpu_call=args.limit_qpu_call,
    )

    config = apply_problem_id(BestKnownConfig(problem=problem, model=model, training=training), args)

    result = gqaoa_strategy.run_job(
        config.problem, config.training, config.model,
        device_name=args.device_name, run_name=args.run_name,
    )
    print(result)

    brute_force_result = try_load_brute_force(config.problem.problem_id)
    if brute_force_result is not None:
        comparison = compare_single_run_to_optimal(result["final_exp_val"], result["probs"], brute_force_result)
        print(f"\n{'='*45}")
        print("Optimality comparison")
        print(f"{'='*45}")
        for k, v in comparison.items():
            print(f"  {k:24s}: {v}")


if __name__ == "__main__":
    main()
