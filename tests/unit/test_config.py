import networkx as nx

from gqaoa.config import BEST_KNOWN_CONFIG, DEFAULT_PROBLEM_ID, RING_TOPOLOGY_EDGES


def test_ring_topology_is_a_single_10_node_cycle():
    G = nx.Graph()
    G.add_edges_from(RING_TOPOLOGY_EDGES)
    assert G.number_of_nodes() == 10
    assert nx.is_connected(G)
    assert all(degree == 2 for _, degree in G.degree())
    assert len(nx.cycle_basis(G)) == 1


def test_best_known_config_matches_documented_values():
    assert BEST_KNOWN_CONFIG.problem.q == 0.3
    assert BEST_KNOWN_CONFIG.problem.B == 5
    assert BEST_KNOWN_CONFIG.problem.sdp is True
    assert BEST_KNOWN_CONFIG.problem.initial_state == "dicke_state"
    assert BEST_KNOWN_CONFIG.problem.mixture_layer == "xy"
    assert BEST_KNOWN_CONFIG.problem.edges_hc == RING_TOPOLOGY_EDGES
    assert BEST_KNOWN_CONFIG.problem.problem_id == DEFAULT_PROBLEM_ID == "default-n10-fixed"

    assert BEST_KNOWN_CONFIG.model.vocab_size == 20
    assert BEST_KNOWN_CONFIG.model.n_embd == 768

    assert BEST_KNOWN_CONFIG.training.depth == 5
    assert BEST_KNOWN_CONFIG.training.optimizer_lr == 3.86e-4
    assert BEST_KNOWN_CONFIG.training.beta_temp == 0.7817
    assert BEST_KNOWN_CONFIG.training.init_scale == 0.1
