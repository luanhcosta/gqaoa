import math

from gqaoa.config import ProblemConfig, TrainingConfig
from gqaoa.domain.data import f_return_cov


def build_problem(problem: ProblemConfig):
    """Load portfolio data and, if requested, apply SDP graph compression.

    Replaces the `if sdp and edges_hc is not None: import networkx; ...` block
    that used to be copy-pasted in gqaoa.py, pennylane_gradient_descent_optimizer.py
    and scipy_minimize.py.

    Returns:
        (expected_value, cov_matrix, lambda_sdp)
    """
    expected_value, cov_matrix = f_return_cov()
    lam = None
    if problem.sdp and problem.edges_hc is not None:
        import networkx as nx
        from gqaoa.domain.compression import compress_matrix

        G = nx.Graph()
        G.add_edges_from(problem.edges_hc)
        lam, cov_matrix = compress_matrix(cov_matrix, G)

    return expected_value, cov_matrix, lam


def beta_temp_schedule(training: TrainingConfig, count_qpu_call: int) -> float:
    """Cosine-annealed sampling temperature for the current QPU-call count.

    Anneals from `training.beta_temp_max` down to `training.beta_temp` over the
    first `training.beta_temp_anneal_frac` fraction of `training.limit_qpu_call`
    QPU calls, then holds constant at `training.beta_temp`. If
    `training.beta_temp_max is None`, annealing is disabled and the result is
    constant at `training.beta_temp` for every count_qpu_call.
    """
    beta_max = training.beta_temp_max if training.beta_temp_max is not None else training.beta_temp
    anneal_qpu = training.beta_temp_anneal_frac * training.limit_qpu_call
    t = min(count_qpu_call / max(anneal_qpu, 1), 1.0)
    return training.beta_temp + (beta_max - training.beta_temp) * 0.5 * (1 + math.cos(math.pi * t))
