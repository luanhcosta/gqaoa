"""N-generic edge-topology generators for `ProblemInstance.edges_hc`/`edges_hb`.

`QAOA.__init__` (gqaoa.domain.qaoa) already falls back to the complete graph
when `edges_hc`/`edges_hb` are None, so `complete_edges` here is only needed
when a caller wants the complete-graph edge list explicit (e.g. to record it
in `provenance`) rather than relying on that fallback.
"""
from __future__ import annotations

from itertools import combinations
from typing import Literal

Edge = tuple[int, int]
Topology = Literal["ring", "complete"]


def ring_edges(n: int) -> list[Edge]:
    """A single N-node cycle, structurally equivalent to `RING_TOPOLOGY_EDGES`
    (config.py) but for a generic N: N edges, every node has degree 2.
    """
    if n < 3:
        raise ValueError(f"ring topology requires n >= 3, got n={n}")
    return [(i, (i + 1) % n) for i in range(n)]


def complete_edges(n: int) -> list[Edge]:
    """All C(n, 2) edges of the complete graph on n nodes."""
    if n < 2:
        raise ValueError(f"complete topology requires n >= 2, got n={n}")
    return list(combinations(range(n), 2))


def generate_topology(n: int, topology: Topology = "ring") -> list[Edge]:
    if topology == "ring":
        return ring_edges(n)
    if topology == "complete":
        return complete_edges(n)
    raise ValueError(f"Unknown topology: {topology!r} (expected 'ring' or 'complete')")
