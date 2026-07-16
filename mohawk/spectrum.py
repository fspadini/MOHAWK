"""Multitaper power spectrum and band power (port of ``calcftspec.m``).

Estimates the channel-wise power spectrum with the multitaper method and
derives normalised band power, using the same peak-in-band logic as the
original FieldTrip-based code.
"""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np
from mne.time_frequency import psd_array_multitaper

from .config import FREQ_BANDS, SpectrumConfig


@dataclass
class SpectrumResult:
    freqs: np.ndarray                 # (n_freqs,)
    spectra: np.ndarray               # (n_channels, n_freqs) mean power
    band_power: np.ndarray            # (n_bands, n_channels) normalised
    ch_names: list
    bands: np.ndarray                 # (n_bands, 2)


def calc_spectrum(
    epochs: mne.Epochs,
    cfg: SpectrumConfig | None = None,
    bands: np.ndarray | None = None,
) -> SpectrumResult:
    """Compute the multitaper spectrum and normalised band power."""
    cfg = cfg or SpectrumConfig()
    bands = FREQ_BANDS if bands is None else np.asarray(bands, dtype=float)

    data = epochs.get_data(copy=False)  # (n_epochs, n_ch, n_times)
    sfreq = epochs.info["sfreq"]

    psds, freqs = psd_array_multitaper(
        data, sfreq, fmin=cfg.fmin, fmax=cfg.fmax,
        bandwidth=cfg.bandwidth, adaptive=False, normalization="full",
        verbose="ERROR",
    )
    spectra = psds.mean(axis=0)  # average over epochs -> (n_ch, n_freqs)

    band_power = _peak_band_power(spectra, freqs, bands)
    return SpectrumResult(
        freqs=freqs,
        spectra=spectra,
        band_power=band_power,
        ch_names=list(epochs.ch_names),
        bands=bands,
    )


def _peak_band_power(
    spectra: np.ndarray, freqs: np.ndarray, bands: np.ndarray
) -> np.ndarray:
    """Peak-in-band power, normalised per channel (calcftspec.m).

    For each band the frequency bin where the channel-averaged power peaks is
    located, and the power of every channel at that bin is taken as the band
    power.  Each channel's band-power vector is then normalised to sum to 1.
    """
    n_bands = bands.shape[0]
    n_ch = spectra.shape[0]
    bpower = np.zeros((n_bands, n_ch))
    for f in range(n_bands):
        bstart = int(np.argmin(np.abs(freqs - bands[f, 0])))
        bstop = int(np.argmin(np.abs(freqs - bands[f, 1])))
        lo, hi = min(bstart, bstop), max(bstart, bstop)
        seg = spectra[:, lo:hi + 1]
        peak = int(np.argmax(seg.mean(axis=0)))
        bpower[f, :] = spectra[:, lo + peak]
    # normalise each channel across bands
    col_sum = bpower.sum(axis=0, keepdims=True)
    col_sum[col_sum == 0] = 1.0
    return bpower / col_sum
