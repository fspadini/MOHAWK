"""Unit tests for the multitaper spectrum / band power (mohawk.spectrum)."""

import numpy as np

from mohawk.config import FREQ_BANDS
from mohawk.spectrum import _peak_band_power


def test_band_power_normalised_per_channel():
    rng = np.random.default_rng(0)
    freqs = np.linspace(0.5, 45, 400)
    spectra = rng.random((10, freqs.size)) + 0.1
    bp = _peak_band_power(spectra, freqs, FREQ_BANDS)
    assert bp.shape == (5, 10)
    # each channel's band powers sum to 1
    assert np.allclose(bp.sum(axis=0), 1.0)


def test_band_power_peak_picks_max_in_band():
    freqs = np.linspace(0.5, 45, 900)
    spectra = np.ones((3, freqs.size)) * 1e-3
    # inject a strong 10 Hz (alpha) peak on all channels
    alpha_bin = int(np.argmin(np.abs(freqs - 10.0)))
    spectra[:, alpha_bin] += 5.0
    bp = _peak_band_power(spectra, freqs, FREQ_BANDS)
    # alpha (index 2) should dominate every channel's normalised power
    assert np.all(bp[2] > bp[0])
    assert np.all(bp[2] > bp[1])
    assert np.all(bp[2] > bp[3])


def test_calc_spectrum_detects_alpha_peak():
    from mohawk.datasets import synthetic_raw
    from mohawk.epoching import make_epochs
    from mohawk.preprocess import preprocess_raw
    from mohawk.spectrum import calc_spectrum

    raw = synthetic_raw(duration=40.0, n_channels=16, seed=3)
    preprocess_raw(raw)
    epochs = make_epochs(raw)
    res = calc_spectrum(epochs)
    mp = res.spectra.mean(axis=0)
    f = res.freqs
    alpha = mp[(f >= 8) & (f < 13)].max()
    theta = mp[(f >= 4) & (f < 8)].max()
    assert alpha > 10 * theta  # clear alpha peak
