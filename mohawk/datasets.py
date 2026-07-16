"""Synthetic hdEEG generator for tests and demos.

The repository ships no patient data, so this module fabricates resting-state
EEG with a realistic 1/f background plus band-limited oscillations that are
phase-coupled within spatial clusters.  The coupling produces non-trivial dwPLI
connectivity and modular network structure, which the pipeline and tests can
verify.
"""

from __future__ import annotations

import mne
import numpy as np


def synthetic_raw(
    montage: str = "GSN-HydroCel-64_1.0",
    sfreq: float = 500.0,
    duration: float = 120.0,
    n_channels: int | None = 32,
    alpha_freq: float = 10.0,
    seed: int = 0,
) -> mne.io.RawArray:
    """Generate a synthetic resting-state EEG recording.

    Two spatial clusters share a common phase-coupled alpha oscillation (one
    per cluster), on top of independent 1/f noise, so that connectivity within
    clusters is high and modular structure emerges.
    """
    rng = np.random.default_rng(seed)
    dig = mne.channels.make_standard_montage(montage)
    ch_pos = dig.get_positions()["ch_pos"]
    all_names = list(ch_pos.keys())
    if n_channels is not None and n_channels < len(all_names):
        # evenly spaced subset for good spatial coverage (avoid a biased block)
        idx = np.linspace(0, len(all_names) - 1, n_channels).round().astype(int)
        ch_names = [all_names[i] for i in idx]
    else:
        ch_names = all_names

    n_ch = len(ch_names)
    n_times = int(duration * sfreq)
    t = np.arange(n_times) / sfreq

    # 1/f background
    data = np.zeros((n_ch, n_times))
    for c in range(n_ch):
        white = rng.standard_normal(n_times)
        spec = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n_times, 1 / sfreq)
        freqs[0] = freqs[1]
        spec = spec / freqs  # 1/f amplitude
        data[c] = np.fft.irfft(spec, n=n_times)
    data /= data.std(axis=1, keepdims=True)

    # two phase-coupled alpha clusters (left vs right hemisphere) -> two modules
    pos = np.array([ch_pos[ch] for ch in ch_names])
    left = pos[:, 0] < np.median(pos[:, 0])
    clusters = [left, ~left]
    for cl in clusters:
        phase = rng.uniform(0, 2 * np.pi)
        jitter = rng.uniform(-0.15, 0.15, size=n_ch)
        for c in np.flatnonzero(cl):
            data[c] += 3.0 * np.sin(2 * np.pi * alpha_freq * t + phase + jitter[c])

    data *= 20e-6  # scale to ~20 microvolts
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_montage(dig, verbose="ERROR")
    return raw
