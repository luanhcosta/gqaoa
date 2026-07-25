import networkx as nx
import numpy as np
import pandas as pd
import pytest

from gqaoa.domain.compression import compress_matrix


def test_compress_matrix_shape_and_sparsity():
    C = pd.DataFrame(np.array([
        [1.0, 0.5, 0.4, 0.3],
        [0.5, 1.0, 0.3, 0.2],
        [0.4, 0.3, 1.0, 0.1],
        [0.3, 0.2, 0.1, 1.0],
    ]))
    # ring: 0-1, 1-2, 2-3, 3-0 — disallows the diagonal pairs (0,2) and (1,3)
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

    lam, X = compress_matrix(C, G)

    assert X.shape == (4, 4)
    assert lam >= 0
    assert X.values[0, 2] == pytest.approx(0.0, abs=1e-6)
    assert X.values[1, 3] == pytest.approx(0.0, abs=1e-6)
