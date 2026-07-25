from gqaoa.config import ProblemConfig
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
