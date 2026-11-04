"""Directed simplices: the "11-dimensional" structure in a wiring diagram.

A directed k-simplex is k+1 neurons with a fixed order in which every earlier neuron connects to every
later one: one source, one sink, and every path between them. A 1-simplex is a single connection, a
2-simplex a feed-forward triangle, an 11-simplex twelve neurons wired all-to-all forward (66
connections). Reimann et al. (2017, Frontiers in Computational Neuroscience, "Cliques of neurons bound
into cavities") counted these in a reconstructed rat neocortical column and found them up to dimension
11; the dimension is that of the clique's geometry, not of space.

``simplices(adj)`` counts them by dimension in any circuit; ``max_dimension`` gives the highest.
"""
from __future__ import annotations


def simplices(adj: dict[int, set[int]], top: int = 20) -> dict[int, int]:
    """Number of directed k-simplices for k = 0, 1, 2, ... (k = 0 counts neurons).

    Grows each ordered clique by a neuron that every member already connects to, so each simplex is
    built exactly once, from its source onward.
    """
    counts: dict[int, int] = {}

    def grow(members: int, common: set[int]):
        counts[members - 1] = counts.get(members - 1, 0) + 1
        if members - 1 >= top:
            return
        for n in common:
            grow(members + 1, common & adj[n])

    for n in adj:
        grow(1, set(adj[n]))
    return dict(sorted(counts.items()))


def max_dimension(adj: dict[int, set[int]]) -> int:
    c = simplices(adj)
    return max(c) if c else -1
