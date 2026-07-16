"""dwPLI connectivity (port of ``ftcoherence.m``).

Estimates debiased weighted phase-lag index (dwPLI) between channel pairs in
each canonical frequency band, using MNE-Connectivity's ``wpli2_debiased``
(the debiased WPLI of Vinck et al., 2011 — the ``ft_connectivity_wpli`` used
with ``debias=true`` in the original code).
"""

from __future__ import annotations

from dataclasses import dataclass

import mne
import numpy as np
from mne_connectivity import spectral_connectivity_epochs

from .config import FREQ_BANDS, ConnectivityConfig


@dataclass
class ConnectivityResult:
    matrix: np.ndarray                # (n_bands, n_ch, n_ch) symmetric
    freqs: np.ndarray                 # (n_freqs,)
    wpli: np.ndarray                  # (n_ch, n_ch, n_freqs) full spectrum
    chan_wpli: np.ndarray             # (n_ch, n_freqs) mean per channel
    ch_names: list
    bands: np.ndarray


def calc_connectivity(
    epochs: mne.Epochs,
    cfg: ConnectivityConfig | None = None,
    bands: np.ndarray | None = None,
) -> ConnectivityResult:
    """Compute per-band dwPLI connectivity matrices with peak-in-band logic."""
    cfg = cfg or ConnectivityConfig()
    bands = FREQ_BANDS if bands is None else np.asarray(bands, dtype=float)

    data = epochs.get_data(copy=False)
    sfreq = epochs.info["sfreq"]
    n_ch = data.shape[1]

    con = spectral_connectivity_epochs(
        data,
        method="wpli2_debiased",
        mode="multitaper",
        sfreq=sfreq,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        mt_bandwidth=cfg.bandwidth,
        mt_adaptive=False,
        faverage=False,
        verbose="ERROR",
    )
    freqs = np.asarray(con.freqs)
    dense = con.get_data(output="dense")  # (n_ch, n_ch, n_freqs), lower-tri
    # symmetrise (MNE fills the lower triangle only)
    wpli = np.abs(dense) + np.abs(dense).transpose(1, 0, 2)

    matrix = _peak_band_matrix(wpli, freqs, bands)

    # mean connectivity per channel across the full spectrum
    chan_wpli = wpli.sum(axis=1) / max(n_ch - 1, 1)

    return ConnectivityResult(
        matrix=matrix,
        freqs=freqs,
        wpli=wpli,
        chan_wpli=chan_wpli,
        ch_names=list(epochs.ch_names),
        bands=bands,
    )


def _peak_band_matrix(
    wpli: np.ndarray, freqs: np.ndarray, bands: np.ndarray
) -> np.ndarray:
    """Peak-in-band connectivity matrix (ftcoherence.m).

    For each band, the frequency where the mean connectivity over all channel
    pairs peaks is located, and every pair's connectivity at that frequency is
    stored.
    """
    n_ch = wpli.shape[0]
    n_bands = bands.shape[0]
    matrix = np.zeros((n_bands, n_ch, n_ch))
    triu = np.triu_indices(n_ch, k=1)
    pair_spectrum = wpli[triu[0], triu[1], :]  # (n_pairs, n_freqs)

    for f in range(n_bands):
        bstart = int(np.argmin(np.abs(freqs - bands[f, 0])))
        bend = int(np.argmin(np.abs(freqs - bands[f, 1])))
        lo, hi = min(bstart, bend), max(bstart, bend)
        band_mean = pair_spectrum[:, lo:hi + 1].mean(axis=0)
        peak = lo + int(np.argmax(band_mean))
        matrix[f] = wpli[:, :, peak]
    return matrix
