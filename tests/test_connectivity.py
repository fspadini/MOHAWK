"""Unit tests for dwPLI connectivity (mohawk.connectivity)."""

import numpy as np

from mohawk.config import FREQ_BANDS
from mohawk.connectivity import _peak_band_matrix, calc_connectivity


def test_peak_band_matrix_symmetric_and_shaped():
    rng = np.random.default_rng(0)
    n_ch, n_freq = 6, 200
    freqs = np.linspace(0.5, 45, n_freq)
    wpli = rng.random((n_ch, n_ch, n_freq))
    wpli = (wpli + wpli.transpose(1, 0, 2)) / 2  # symmetrise
    mat = _peak_band_matrix(wpli, freqs, FREQ_BANDS)
    assert mat.shape == (5, n_ch, n_ch)
    for f in range(5):
        assert np.allclose(mat[f], mat[f].T)


def test_connectivity_high_for_phase_coupled_channels():
    from mohawk.datasets import synthetic_raw
    from mohawk.epoching import make_epochs
    from mohawk.preprocess import preprocess_raw

    raw = synthetic_raw(duration=60.0, n_channels=16, seed=5)
    preprocess_raw(raw)
    epochs = make_epochs(raw)
    conn = calc_connectivity(epochs)
    alpha = conn.matrix[2]  # alpha band
    off = alpha[np.triu_indices_from(alpha, k=1)]
    assert conn.matrix.shape[0] == 5
    assert np.all(off >= 0) and np.all(off <= 1.001)
    # phase-coupled alpha -> strong mean connectivity
    assert off.mean() > 0.5


def test_connectivity_matrix_zero_diagonal():
    from mohawk.datasets import synthetic_raw
    from mohawk.epoching import make_epochs
    from mohawk.preprocess import preprocess_raw

    raw = synthetic_raw(duration=40.0, n_channels=12, seed=6)
    preprocess_raw(raw)
    epochs = make_epochs(raw)
    conn = calc_connectivity(epochs)
    for f in range(conn.matrix.shape[0]):
        assert np.allclose(np.diag(conn.matrix[f]), 0.0)
