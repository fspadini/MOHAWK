"""Per-band / per-threshold graph metrics (port of ``calcgraph.m``).

Binarises each band's connectivity matrix at a range of proportional density
thresholds and computes micro-, meso- and macro-scale graph metrics, averaging
the heuristic (community-based) metrics over repeated Louvain runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import graph as G
from .config import GraphConfig


@dataclass
class GraphMetrics:
    tvals: np.ndarray
    clustering: np.ndarray            # (n_bands, n_thresh, n_ch)
    char_path: np.ndarray             # (n_bands, n_thresh)
    global_efficiency: np.ndarray     # (n_bands, n_thresh)
    modularity: np.ndarray            # (n_bands, n_thresh)
    modules: np.ndarray               # (n_bands, n_thresh, n_ch)
    centrality: np.ndarray            # (n_bands, n_thresh, n_ch)
    modular_span: np.ndarray          # (n_bands, n_thresh)
    participation: np.ndarray         # (n_bands, n_thresh, n_ch)
    degree: np.ndarray                # (n_bands, n_thresh, n_ch)
    band_names: tuple = ()


def channel_distance(pos: np.ndarray) -> np.ndarray:
    """Normalised Euclidean inter-channel distance matrix (ichandist)."""
    diff = pos[:, None, :] - pos[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=-1))
    m = d.max()
    return d / m if m > 0 else d


def calc_graph_metrics(
    matrix: np.ndarray,
    chandist: np.ndarray,
    cfg: GraphConfig | None = None,
    band_names: tuple = (),
    seed: int | None = None,
) -> GraphMetrics:
    """Compute graph metrics for every band and density threshold.

    Parameters
    ----------
    matrix : (n_bands, n_ch, n_ch) connectivity matrices.
    chandist : (n_ch, n_ch) normalised channel distances (for modular span).
    """
    cfg = cfg or GraphConfig()
    tvals = np.asarray(cfg.thresholds, dtype=float)
    n_bands, n_ch, _ = matrix.shape
    n_t = tvals.size

    clustering = np.zeros((n_bands, n_t, n_ch))
    char_path = np.zeros((n_bands, n_t))
    global_eff = np.zeros((n_bands, n_t))
    modularity = np.zeros((n_bands, n_t))
    modules = np.zeros((n_bands, n_t, n_ch))
    centrality = np.zeros((n_bands, n_t, n_ch))
    mod_span = np.zeros((n_bands, n_t))
    participation = np.zeros((n_bands, n_t, n_ch))
    degree = np.zeros((n_bands, n_t, n_ch))

    rng = np.random.default_rng(seed)

    for f in range(n_bands):
        cohmat = np.abs(np.nan_to_num(matrix[f]))
        for ti, t in enumerate(tvals):
            bincoh = (G.threshold_proportional(cohmat, t) != 0).astype(float)

            clustering[f, ti] = G.clustering_coef_bu(bincoh)
            char_path[f, ti] = G.charpath(G.distance_bin(bincoh))
            global_eff[f, ti] = G.efficiency_bin(bincoh)
            centrality[f, ti] = G.betweenness_bin(bincoh)
            degree[f, ti] = G.degrees_und(bincoh)

            Qs = np.empty(cfg.heuristic)
            spans = np.empty(cfg.heuristic)
            pcs = np.empty((cfg.heuristic, n_ch))
            first_ci = None
            for h in range(cfg.heuristic):
                s = int(rng.integers(0, 2**31 - 1))
                Ci, Q = G.community_louvain(bincoh, seed=s)
                if first_ci is None:
                    first_ci = Ci
                Qs[h] = Q
                spans[h] = G.modular_span(bincoh, Ci, chandist)
                pcs[h] = G.participation_coef(bincoh, Ci)

            modularity[f, ti] = Qs.mean()
            mod_span[f, ti] = spans.mean()
            participation[f, ti] = pcs.mean(axis=0)
            modules[f, ti] = first_ci

    return GraphMetrics(
        tvals=tvals,
        clustering=clustering,
        char_path=char_path,
        global_efficiency=global_eff,
        modularity=modularity,
        modules=modules,
        centrality=centrality,
        modular_span=mod_span,
        participation=participation,
        degree=degree,
        band_names=band_names,
    )
