import math

import networkx as nx
import pytest

from gqaoa.problem.topology import complete_edges, generate_topology, ring_edges


@pytest.mark.parametrize("n", [4, 10, 50])
def test_ring_edges_has_n_edges_and_all_nodes_degree_2(n):
    edges = ring_edges(n)
    assert len(edges) == n

    G = nx.Graph()
    G.add_edges_from(edges)
    assert G.number_of_nodes() == n
    assert nx.is_connected(G)
    assert all(degree == 2 for _, degree in G.degree())
    assert len(nx.cycle_basis(G)) == 1


@pytest.mark.parametrize("n", [4, 10, 50])
def test_complete_edges_has_n_choose_2_edges(n):
    edges = complete_edges(n)
    assert len(edges) == math.comb(n, 2)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)
    assert G.number_of_nodes() == n
    assert all(degree == n - 1 for _, degree in G.degree())


def test_generate_topology_dispatches_by_name():
    assert generate_topology(6, "ring") == ring_edges(6)
    assert generate_topology(6, "complete") == complete_edges(6)


def test_generate_topology_rejects_unknown_name():
    with pytest.raises(ValueError):
        generate_topology(6, "star")


def test_ring_edges_rejects_too_small_n():
    with pytest.raises(ValueError):
        ring_edges(2)
