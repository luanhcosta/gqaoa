import numpy as np
import cvxpy as cp
import pandas as pd
import networkx as nx


def compress_matrix(C, G):
    r"""
    We are interested in approximating some symmetric matrix $C$ by a sparse symmetric matrix $X$,
    subject to restrictions of the form $X_{ij} = 0$ for all $ij \in E(\overline G)$. One way to do this is via the following SDP:
    $$
    \min \left\{ \lambda : \lambda I \succcurlyeq X - C \succcurlyeq -\lambda I, X \circ A(\overline G) = 0 \right\}
    $$
    Solutions with small lambda are good aproximations: note that for $\lambda = 0$, we get $X = C$.

    Given a matrix C and a graph of allowed entries G, returns the solution [ lambda, X ] for the above SDP.

    Args:
        C (pd.DataFrame): Input matrix.
        G (networkx.Graph): Graph of allowed (nonzero) entries.

    Returns:
        (float, pd.DataFrame): Solution [ lambda, X ] for the above SDP.
    """

    np.set_printoptions(edgeitems=3, infstr='inf', linewidth=200,
                    nanstr='nan', precision=8, suppress=True,
                    threshold=1000, formatter=None)
    n = np.shape(C)[0]

    # Variables
    X = cp.Variable((n, n), symmetric=True)
    lam = cp.Variable()

    # Constraints
    constraints = [
        X - C << lam * np.eye(n),
        X - C >> -lam * np.eye(n),
    ]
    # Linear restrictions
    G_comp = nx.complement(G)
    constraints += [X[i, j] == 0 for (i, j) in G_comp.edges()]

    # Objective
    objective = cp.Minimize(lam)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, max_iters=10000, eps=1e-9)

    response_matrix = pd.DataFrame(X.value)
    for i in response_matrix.columns: response_matrix[i][i] = np.abs(response_matrix[i][i])

    return lam.value, response_matrix
