"""Tests for the group-level stack: features, stats, classification."""

import numpy as np
import pytest

from mohawk.classification import FeatureSpec, build_feature_matrix, svm_cross_validation
from mohawk.features import SubjectFeatures
from mohawk.stats import absolute_auc, fdr_bh, jonckheere_terpstra, screen_pair


def _subject(shift, seed):
    rng = np.random.default_rng(seed)
    nb, nt, nc = 3, 4, 8
    return SubjectFeatures(
        ch_names=tuple(f"c{i}" for i in range(nc)),
        band_names=("delta", "theta", "alpha"),
        densities=np.array([0.9, 0.5, 0.3, 0.1]),
        relative_power=rng.normal(shift, 1, (nb, nc)),
        connectivity=np.abs(rng.normal(shift * 0.1, 0.1, (nb, nc, nc))),
        clustering=rng.normal(shift, 1, (nb, nt, nc)),
        path_length=rng.normal(shift, 1, (nb, nt)),
        modularity=rng.normal(shift, 1, (nb, nt)),
        participation=rng.normal(shift, 1, (nb, nt, nc)),
        modular_span=rng.normal(shift, 1, (nb, nt)),
    )


def test_absolute_auc_direction_agnostic():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert absolute_auc(y, np.array([1, 2, 3, 4, 5, 6.0])) == 1.0
    assert absolute_auc(y, np.array([6, 5, 4, 3, 2, 1.0])) == 1.0  # folded
    assert absolute_auc(y, np.array([1, 6, 2, 5, 3, 4.0])) == pytest.approx(0.5, abs=0.2)


def test_fdr_bh_monotone_and_bounded():
    q = fdr_bh([0.01, 0.02, 0.5])
    assert np.all((q >= 0) & (q <= 1))
    assert q[2] >= q[0]


def test_features_save_load_roundtrip(tmp_path):
    sf = _subject(0.0, 1)
    path = sf.save(tmp_path / "s.npz")
    loaded = SubjectFeatures.load(path)
    assert loaded.band_names == sf.band_names
    assert np.allclose(loaded.participation, sf.participation)
    assert len(loaded.node_feature("participation", "alpha", 0.3)) == 8
    assert np.isfinite(loaded.scalar_metric("relative_power", "delta"))
    assert np.isfinite(loaded.scalar_metric("median_connectivity", "alpha"))


def test_screen_pair_finds_separating_metric():
    subs = [_subject(0.0, s) for s in range(8)] + [_subject(4.0, 100 + s) for s in range(8)]
    labels = ["UWS"] * 8 + ["MCS"] * 8
    rows = screen_pair(subs, labels, "UWS", "MCS")
    assert rows[0].auc > 0.9          # clearly separable
    assert rows[0].auc >= rows[-1].auc  # sorted descending
    assert np.isfinite(rows[0].p_fdr)


def test_jonckheere_terpstra_detects_trend():
    groups = [np.array([1, 2, 3.0]), np.array([3, 4, 5.0]), np.array([6, 7, 8.0])]
    obs, p = jonckheere_terpstra(groups, n_permutations=500, random_state=0)
    assert p < 0.05
    flat = [np.array([1, 2, 3.0]), np.array([1, 2, 3.0]), np.array([1, 2, 3.0])]
    _, p_flat = jonckheere_terpstra(flat, n_permutations=500, random_state=0)
    assert p_flat > 0.05


def test_svm_cross_validation_separable():
    # two clearly separated clusters in feature space
    rng = np.random.default_rng(0)
    subs = [_subject(0.0, s) for s in range(10)] + [_subject(6.0, 100 + s) for s in range(10)]
    labels = ["UWS"] * 10 + ["MCS"] * 10
    x = build_feature_matrix(subs, [FeatureSpec("relative_power", "delta")])
    result = svm_cross_validation(x, labels, compact_grid=True)
    assert result.accuracy > 0.8
    assert result.confusion.shape == (2, 2)
    assert result.youden_threshold is not None
