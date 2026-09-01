from gqaoa.config import ProblemConfig
from gqaoa.domain.data import f_return_cov


def build_problem(problem: ProblemConfig):
    """Load portfolio data and, if requested, apply SDP graph compression.

    Replaces the `if sdp and edges_hc is not None: import networkx; ...` block
    that used to be copy-pasted in gqaoa.py, pennylane_gradient_descent_optimizer.py
    and scipy_minimize.py.

    If `problem.problem_id` is None, data comes straight from `f_return_cov()`
    with no `gqaoa.problem` involvement at all — a low-level escape hatch for
    ad-hoc/in-memory configs (tests, mainly), not what any CLI defaults to.
    `problem.problem_id == DEFAULT_PROBLEM_ID` (the default for `BEST_KNOWN_CONFIG`
    and every CLI) loads the same numbers via
    `gqaoa.problem.default.ensure_default_problem_persisted`, which bootstraps
    `artifacts/problems/default-n10-fixed/problem.json` on first use. Any other
    `problem_id` is loaded via `gqaoa.problem.store.load_problem` — the caller
    is responsible for having persisted it first (e.g. via `gqaoa-brute-force`).

    Returns:
        (expected_value, cov_matrix, lambda_sdp)
    """
    if problem.problem_id is None:
        expected_value, cov_matrix = f_return_cov()
        edges_for_sdp = problem.edges_hc
    else:
        from gqaoa.config import DEFAULT_PROBLEM_ID
        from gqaoa.problem.default import ensure_default_problem_persisted
        from gqaoa.problem.store import load_problem

        if problem.problem_id == DEFAULT_PROBLEM_ID:
            instance = ensure_default_problem_persisted()
        else:
            instance = load_problem(problem.problem_id)
        expected_value, cov_matrix = instance.expected_value, instance.cov_matrix
        # SDP compression graph: prefer an explicit problem.edges_hc (e.g. a CLI
        # override), but fall back to the loaded instance's own topology — the
        # config's edges_hc otherwise defaults to None (ProblemConfig) or the
        # fixed 10-node RING_TOPOLOGY_EDGES (BEST_KNOWN_CONFIG), neither of which
        # is meaningful for an arbitrary persisted problem.
        edges_for_sdp = problem.edges_hc if problem.edges_hc is not None else instance.edges_hc

    lam = None
    if problem.sdp and edges_for_sdp is not None:
        import networkx as nx
        from gqaoa.domain.compression import compress_matrix

        G = nx.Graph()
        G.add_edges_from(edges_for_sdp)
        lam, cov_matrix = compress_matrix(cov_matrix, G)

    return expected_value, cov_matrix, lam
