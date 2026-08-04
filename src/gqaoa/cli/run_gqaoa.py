"""Single GQAOA dev-run — thin CLI wrapper around gqaoa_strategy.run_job."""
import argparse

from gqaoa.config import ModelConfig, ProblemConfig, RING_TOPOLOGY_EDGES, TrainingConfig
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
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)

    init_mlflow("gqaoa-dev")

    problem = ProblemConfig(
        q=0.3, B=5, lamb=0,
        initial_state="dicke_state", mixture_layer="xy",
        edges_hc=RING_TOPOLOGY_EDGES, edges_hb=RING_TOPOLOGY_EDGES,
        sdp=True,
    )
    model = ModelConfig(vocab_size=args.vocab_size, n_layer=args.n_layer)
    training = TrainingConfig(
        depth=args.depth,
        optimizer_lr=args.optimizer_lr,
        beta_temp=args.beta_temp,
        limit_epochs=args.limit_epochs,
        limit_qpu_call=args.limit_qpu_call,
    )

    result = gqaoa_strategy.run_job(
        problem, training, model,
        device_name=args.device_name, run_name=args.run_name,
    )
    print(result)


if __name__ == "__main__":
    main()
