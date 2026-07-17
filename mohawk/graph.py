"""Graph-theoretic network metrics on binary undirected graphs.

Python ports of the Brain Connectivity Toolbox (BCT) routines used by the
original ``calcgraph.m`` / ``plotgraph3d.m``:

* ``threshold_proportional``  – keep the strongest proportion of edges
* ``degrees_und``             – node degree
* ``clustering_coef_bu``      – clustering coefficient
* ``distance_bin``            – shortest-path distance matrix
* ``charpath``                – characteristic path length
* ``efficiency_bin``          – global efficiency
* ``betweenness_bin``         – betweenness centrality
* ``community_louvain``       – Louvain community detection + modularity Q
* ``participation_coef``      – participation coefficient
* ``modular_span``            – modular span (Chennu et al., 2014)

The metrics operate on symmetric matrices; connectivity input is binarised
after proportional thresholding, exactly as in ``calcgraph.m``.

References
----------
Rubinov & Sporns (2010), NeuroImage 52:1059-1069 (BCT).
Chennu et al. (2014), PLoS Comput Biol 10:e1003887 (modular span).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import shortest_path


def threshold_proportional(W: np.ndarray, p: float) -> np.ndarray:
    """Keep the strongest proportion ``p`` of edges of ``W``.

    Faithful port of BCT ``threshold_proportional`` for symmetric matrices.
    The diagonal is cleared, edges are ranked by weight (descending), and only
    the top ``round((n^2 - n) * p / 2)`` undirected edges are retained.
    """
    W = np.array(W, dtype=float)
    n = W.shape[0]
    np.fill_diagonal(W, 0.0)

    symmetric = np.allclose(W, W.T)
    if symmetric:
        ud = 2
        Wt = np.triu(W)
    else:
        ud = 1
        Wt = W.copy()

    ind = np.flatnonzero(Wt)
    if ind.size == 0:
        return np.zeros_like(W)

    vals = Wt.flat[ind]
    order = ind[np.argsort(-vals, kind="stable")]
    en = int(round((n * n - n) * p / ud))
    en = max(0, min(en, order.size))

    out = np.zeros_like(W)
    out.flat[order[:en]] = Wt.flat[order[:en]]
    if ud == 2:
        out = out + out.T
    return out


def degrees_und(A: np.ndarray) -> np.ndarray:
    """Node degree of a binary undirected graph (BCT ``degrees_und``)."""
    B = (np.asarray(A) != 0).astype(float)
    np.fill_diagonal(B, 0.0)
    return B.sum(axis=0)


def clustering_coef_bu(A: np.ndarray) -> np.ndarray:
    """Clustering coefficient, binary undirected (BCT ``clustering_coef_bu``).

    Vectorised: the number of closed triangles through each node is
    ``diag(A @ A @ A)`` and the clustering coefficient is that count divided by
    ``k*(k-1)`` (the number of possible connected neighbour pairs).
    """
    G = (np.asarray(A) != 0).astype(float)
    np.fill_diagonal(G, 0.0)
    k = G.sum(axis=1)
    triangles_twice = np.diagonal(G @ G @ G)
    denom = k * (k - 1)
    return np.divide(
        triangles_twice, denom,
        out=np.zeros_like(k, dtype=float), where=denom > 0,
    )


def distance_bin(A: np.ndarray) -> np.ndarray:
    """Shortest-path length matrix of a binary graph (BCT ``distance_bin``).

    Disconnected pairs receive ``inf``; the diagonal is ``0``.
    """
    G = (np.asarray(A) != 0).astype(float)
    np.fill_diagonal(G, 0.0)
    D = shortest_path(G, method="D", unweighted=True, directed=False)
    return D


def charpath(D: np.ndarray) -> float:
    """Characteristic path length (BCT ``charpath(D, 0, 0)``).

    Mean of the finite, off-diagonal entries of the distance matrix ``D``
    (diagonal excluded, infinite distances excluded).
    """
    D = np.array(D, dtype=float)
    n = D.shape[0]
    mask = ~np.eye(n, dtype=bool) & np.isfinite(D)
    if not mask.any():
        return np.inf
    return float(D[mask].mean())


def efficiency_bin(A: np.ndarray) -> float:
    """Global efficiency of a binary graph (BCT ``efficiency_bin``).

    Average of the inverse shortest-path lengths over all ordered node pairs.
    """
    D = distance_bin(A)
    n = D.shape[0]
    with np.errstate(divide="ignore"):
        inv = 1.0 / D
    inv[~np.isfinite(inv)] = 0.0
    np.fill_diagonal(inv, 0.0)
    if n <= 1:
        return 0.0
    return float(inv.sum() / (n * (n - 1)))


def betweenness_bin(G: np.ndarray) -> np.ndarray:
    """Node betweenness centrality, binary graph (BCT ``betweenness_bin``).

    Returns raw (unnormalised) betweenness for each node, counting the number
    of shortest paths that pass through it.
    """
    G = (np.asarray(G) != 0).astype(float)
    n = G.shape[0]
    I = np.eye(n, dtype=bool)

    d = 1
    NPd = G.copy()
    NSPd = NPd.copy()
    NSP = NSPd.copy()
    NSP[I] = 1
    L = NSPd.copy()
    L[I] = 1

    while NSPd.any():
        d += 1
        NPd = NPd @ G
        NSPd = NPd * (L == 0)
        NSP = NSP + NSPd
        L = L + d * (NSPd != 0)

    L[L == 0] = np.inf
    L[I] = 0
    NSP[NSP == 0] = 1

    Gt = G.T
    DP = np.zeros((n, n))
    diam = d - 1
    for layer in range(diam - 1, 0, -1):
        contrib = ((L == layer + 1) * (1 + DP) / NSP) @ Gt
        DPd1 = contrib * ((L == layer) * NSP)
        DP = DP + DPd1
    return DP.sum(axis=0)


def community_louvain(
    A: np.ndarray, seed: int | None = None
) -> tuple[np.ndarray, float]:
    """Louvain community detection returning ``(Ci, Q)``.

    ``Ci`` is a 1-based community-affiliation vector (to match MATLAB
    conventions used downstream); ``Q`` is the modularity.
    """
    B = (np.asarray(A) != 0).astype(float)
    np.fill_diagonal(B, 0.0)
    n = B.shape[0]
    g = nx.from_numpy_array(B)

    isolated = [i for i in range(n) if B[i].sum() == 0]
    communities = nx.community.louvain_communities(g, seed=seed, weight="weight")

    Ci = np.zeros(n, dtype=int)
    label = 1
    for comm in communities:
        for node in comm:
            Ci[node] = label
        label += 1
    # Give every isolated node its own community (BCT-like behaviour).
    for node in isolated:
        Ci[node] = label
        label += 1

    Q = nx.community.modularity(g, communities, weight="weight") if g.number_of_edges() else 0.0
    return Ci, float(Q)


def participation_coef(W: np.ndarray, Ci: np.ndarray) -> np.ndarray:
    """Participation coefficient (BCT ``participation_coef``)."""
    W = np.asarray(W, dtype=float)
    Ci = np.asarray(Ci)
    np.fill_diagonal(W, 0.0)
    n = W.shape[0]
    Ko = W.sum(axis=1)
    Kc2 = np.zeros(n)
    for m in np.unique(Ci):
        Kc2 += (W * (Ci[np.newaxis, :] == m)).sum(axis=1) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        P = 1.0 - Kc2 / (Ko ** 2)
    P[Ko == 0] = 0.0
    return P


def modular_span(bincoh: np.ndarray, Ci: np.ndarray, chandist: np.ndarray) -> float:
    """Modular span of the largest-spanning module (Chennu et al., 2014).

    For each module ``m`` with more than one node, the span is the sum of the
    within-module edge distances divided by the module size; the returned
    value is the maximum over modules.
    """
    bincoh = np.asarray(bincoh, dtype=float)
    chandist = np.asarray(chandist, dtype=float)
    Ci = np.asarray(Ci)
    spans = []
    for m in np.unique(Ci):
        idx = np.flatnonzero(Ci == m)
        if idx.size > 1:
            sub = chandist[np.ix_(idx, idx)] * bincoh[np.ix_(idx, idx)]
            triu = sub[np.triu_indices(idx.size, k=1)]
            triu = triu[triu != 0]
            spans.append(triu.sum() / idx.size)
    spans = [s for s in spans if s > 0]
    return float(max(spans)) if spans else 0.0
