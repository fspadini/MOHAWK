"""Unit tests for the BCT graph-metric ports (mohawk.graph)."""

import numpy as np
import pytest

from mohawk import graph as G


def ring(n):
    A = np.zeros((n, n))
    for i in range(n):
        A[i, (i + 1) % n] = 1
        A[(i + 1) % n, i] = 1
    return A


def complete(n):
    A = np.ones((n, n)) - np.eye(n)
    return A


def star(n):
    A = np.zeros((n, n))
    A[0, 1:] = 1
    A[1:, 0] = 1
    return A


def test_degrees_und():
    assert np.allclose(G.degrees_und(ring(5)), 2)
    assert np.allclose(G.degrees_und(complete(4)), 3)
    d = G.degrees_und(star(5))
    assert d[0] == 4 and np.allclose(d[1:], 1)


def test_clustering_triangle_and_ring():
    tri = complete(3)
    assert np.allclose(G.clustering_coef_bu(tri), 1.0)
    # C4 ring has no closed triangles -> zero clustering
    assert np.allclose(G.clustering_coef_bu(ring(4)), 0.0)
    # complete graph clustering is 1 everywhere
    assert np.allclose(G.clustering_coef_bu(complete(5)), 1.0)


def test_distance_and_charpath():
    D = G.distance_bin(ring(4))
    expected = np.array([[0, 1, 2, 1], [1, 0, 1, 2], [2, 1, 0, 1], [1, 2, 1, 0]])
    assert np.allclose(D, expected)
    # mean of off-diagonal finite distances
    assert G.charpath(D) == pytest.approx(16 / 12)


def test_distance_disconnected_is_inf():
    A = np.zeros((4, 4))
    A[0, 1] = A[1, 0] = 1  # component {0,1}; {2,3} isolated
    D = G.distance_bin(A)
    assert np.isinf(D[0, 2])
    assert D[0, 1] == 1


def test_global_efficiency():
    # complete graph: all distances 1 -> efficiency 1
    assert G.efficiency_bin(complete(5)) == pytest.approx(1.0)
    # ring C4: inverse distances mean
    assert G.efficiency_bin(ring(4)) == pytest.approx(10 / 12)


def test_betweenness_star_hub():
    bc = G.betweenness_bin(star(5))
    # the hub lies on every shortest path between leaves
    assert bc[0] == bc.max()
    assert np.allclose(bc[1:], 0)


def test_threshold_proportional_keeps_strongest():
    W = np.array(
        [[0, 0.9, 0.1, 0.5],
         [0.9, 0, 0.2, 0.4],
         [0.1, 0.2, 0, 0.8],
         [0.5, 0.4, 0.8, 0]]
    )
    # 6 undirected edges; keep 50% -> 3 strongest (0.9, 0.8, 0.5)
    T = G.threshold_proportional(W, 0.5)
    assert np.count_nonzero(np.triu(T)) == 3
    kept = set(np.round(T[np.triu_indices(4, 1)], 3)) - {0.0}
    assert kept == {0.9, 0.8, 0.5}
    # symmetry preserved
    assert np.allclose(T, T.T)


def test_threshold_proportional_full_and_empty():
    W = complete(4)
    assert np.count_nonzero(G.threshold_proportional(W, 1.0)) == np.count_nonzero(W)
    assert np.count_nonzero(G.threshold_proportional(np.zeros((4, 4)), 0.5)) == 0


def test_community_louvain_two_cliques():
    # two disjoint triangles connected by a single bridge edge
    A = np.zeros((6, 6))
    for block in ([0, 1, 2], [3, 4, 5]):
        for i in block:
            for j in block:
                if i != j:
                    A[i, j] = 1
    A[2, 3] = A[3, 2] = 1  # bridge
    Ci, Q = G.community_louvain(A, seed=0)
    assert len(np.unique(Ci)) == 2
    assert Q > 0.3
    # nodes within a clique share a community
    assert Ci[0] == Ci[1] == Ci[2]
    assert Ci[3] == Ci[4] == Ci[5]
    assert Ci[0] != Ci[3]


def test_participation_coef_bounds():
    A = complete(6)
    Ci = np.array([1, 1, 1, 2, 2, 2])
    P = G.participation_coef(A, Ci)
    # every node connects equally to both modules -> P near 0.5
    assert np.all(P >= 0) and np.all(P <= 1)
    assert np.allclose(P, 0.5, atol=0.05)


def test_participation_coef_single_module_is_zero():
    A = complete(5)
    Ci = np.ones(5, dtype=int)
    P = G.participation_coef(A, Ci)
    assert np.allclose(P, 0.0)


def test_modular_span():
    bincoh = complete(4)
    Ci = np.array([1, 1, 2, 2])
    chandist = np.array(
        [[0, 0.5, 1, 1], [0.5, 0, 1, 1], [1, 1, 0, 0.5], [1, 1, 0.5, 0]]
    )
    span = G.modular_span(bincoh, Ci, chandist)
    # module 1: within-edge distance 0.5 / size 2 = 0.25; same for module 2
    assert span == pytest.approx(0.25)
