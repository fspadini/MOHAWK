"""Unit tests for per-band/per-threshold graph metrics (mohawk.metrics)."""

import numpy as np

from mohawk.config import GraphConfig
from mohawk.metrics import calc_graph_metrics, channel_distance


def _two_module_matrix(n_ch=8):
    """Connectivity with two strongly-connected blocks (clear modular structure)."""
    mat = np.zeros((1, n_ch, n_ch))
    half = n_ch // 2
    block = np.ones((half, half)) - np.eye(half)
    mat[0, :half, :half] = block * 0.9
    mat[0, half:, half:] = block * 0.9
    # weak inter-module links
    mat[0, :half, half:] = 0.1
    mat[0, half:, :half] = 0.1
    np.fill_diagonal(mat[0], 0)
    return mat


def test_channel_distance_normalised():
    pos = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0]], float)
    d = channel_distance(pos)
    assert d.max() == 1.0
    assert np.allclose(np.diag(d), 0)
    assert np.allclose(d, d.T)


def test_graph_metrics_shapes_and_modularity():
    mat = _two_module_matrix(8)
    pos = np.random.default_rng(0).random((8, 3))
    chandist = channel_distance(pos)
    cfg = GraphConfig(thresholds=np.array([1.0, 0.5]), heuristic=3)
    gm = calc_graph_metrics(mat, chandist, cfg, band_names=("delta",), seed=1)

    n_t = 2
    assert gm.clustering.shape == (1, n_t, 8)
    assert gm.modularity.shape == (1, n_t)
    assert gm.participation.shape == (1, n_t, 8)
    assert gm.degree.shape == (1, n_t, 8)
    # at full density every edge is kept (one component); the modular split
    # emerges once the weak inter-module links are thresholded away (t=0.5)
    assert len(np.unique(gm.modules[0, 0])) == 1
    assert len(np.unique(gm.modules[0, 1])) == 2
    # modularity is a valid value in (-0.5, 1]
    assert np.all(gm.modularity <= 1.0)
    assert np.all(gm.modularity[0] >= 0.0)


def test_graph_metrics_degree_decreases_with_threshold():
    mat = _two_module_matrix(10)
    chandist = channel_distance(np.random.default_rng(1).random((10, 3)))
    cfg = GraphConfig(thresholds=np.array([1.0, 0.3]), heuristic=2)
    gm = calc_graph_metrics(mat, chandist, cfg, seed=2)
    # sparser threshold -> fewer edges -> lower mean degree
    assert gm.degree[0, 1].mean() < gm.degree[0, 0].mean()
