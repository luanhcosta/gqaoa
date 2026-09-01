"""Single source of truth for shared problem/model/training configuration.

Replaces the ring-topology literal and best-known-hyperparameter dicts that used
to be copy-pasted across bracket.py, stability_bracket.py, stability_check.py,
benchmark_gd.py, hpo.py and the __main__ blocks of gqaoa.py / the classical
baselines.
"""
from dataclasses import dataclass, field

RING_TOPOLOGY_EDGES = [
    (0, 4), (4, 1), (1, 7), (7, 2), (2, 8),
    (8, 5), (5, 9), (9, 6), (6, 3), (3, 0),
]


@dataclass
class ProblemConfig:
    q: float = 0.5
    B: float = 2
    lamb: float = 0
    initial_state: str = "dicke_state"
    mixture_layer: str = "xy"
    edges_hc: list | None = None
    edges_hb: list | None = None
    sdp: bool = False
    # Optional id of a persisted `gqaoa.problem.ProblemInstance` (see
    # `gqaoa.problem.store.load_problem`). When None (default), build_problem()
    # keeps using the fixed 10-asset f_return_cov() problem — 100% unchanged
    # behavior. When set, build_problem() loads that instance's
    # expected_value/cov_matrix instead.
    problem_id: str | None = None


@dataclass
class ModelConfig:
    vocab_size: int = 10
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12


@dataclass
class TrainingConfig:
    depth: int = 10
    limit_epochs: int = 100
    limit_qpu_call: int = 100
    optimizer_lr: float = 0.01
    lr_T0: int = 20
    lr_T_mult: int = 2
    beta_temp: float = 1.0
    init_scale: float = 1.0
    shots_probs_result: int = 1000


def _best_known_problem() -> ProblemConfig:
    return ProblemConfig(
        q=0.3, B=5, lamb=0,
        initial_state="dicke_state", mixture_layer="xy",
        edges_hc=RING_TOPOLOGY_EDGES, edges_hb=RING_TOPOLOGY_EDGES,
        sdp=True,
    )


def _best_known_model() -> ModelConfig:
    return ModelConfig(vocab_size=20, n_embd=768, n_layer=12, n_head=12)


def _best_known_training() -> TrainingConfig:
    return TrainingConfig(
        depth=5, limit_epochs=9999, limit_qpu_call=1000,
        optimizer_lr=3.86e-4, beta_temp=0.7817, init_scale=0.1,
    )


@dataclass
class BestKnownConfig:
    """The best hyperparameters found via HPO (see hpo.py / experiments/hpo.py)."""
    problem: ProblemConfig = field(default_factory=_best_known_problem)
    model: ModelConfig = field(default_factory=_best_known_model)
    training: TrainingConfig = field(default_factory=_best_known_training)


BEST_KNOWN_CONFIG = BestKnownConfig()
