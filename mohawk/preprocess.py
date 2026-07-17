"""Signal preprocessing (port of ``dataimport.m`` / ``epochdata.m``).

Downsampling to 250 Hz, band-pass filtering (0.5-45 Hz), line-noise removal,
average referencing, and fixed-length epoching.
"""

from __future__ import annotations

import mne
import numpy as np

from .config import PreprocConfig


def resample(raw: mne.io.BaseRaw, cfg: PreprocConfig) -> mne.io.BaseRaw:
    """Downsample to ``cfg.resample_sfreq`` (dataimport.m: pop_resample 250)."""
    if raw.info["sfreq"] > cfg.resample_sfreq:
        raw.resample(cfg.resample_sfreq, verbose="ERROR")
    elif raw.info["sfreq"] < cfg.resample_sfreq:
        raise ValueError(
            f"Sampling rate too low! ({raw.info['sfreq']} < {cfg.resample_sfreq})"
        )
    return raw


def filter_band(raw: mne.io.BaseRaw, cfg: PreprocConfig) -> mne.io.BaseRaw:
    """Band-pass filter 0.5-45 Hz (dataimport.m: pop_eegfiltnew)."""
    raw.filter(
        l_freq=cfg.hp_freq, h_freq=cfg.lp_freq,
        fir_design="firwin", verbose="ERROR",
    )
    return raw


def detect_line_frequency(
    raw: mne.io.BaseRaw, candidates=(50.0, 60.0), min_prominence=6.0
) -> float:
    """Detect the mains frequency (50 vs 60 Hz) from the raw power spectrum.

    Returns whichever candidate has the most prominent narrowband peak above its
    local baseline; falls back to the first candidate if neither stands out
    (e.g. already line-filtered data).
    """
    from mne.time_frequency import psd_array_welch

    sfreq = raw.info["sfreq"]
    nyq = sfreq / 2.0
    cands = [c for c in candidates if c < nyq - 2]
    if not cands:
        return candidates[0]
    data = raw.get_data(picks="eeg")
    n_fft = int(min(data.shape[-1], sfreq * 4))
    psds, freqs = psd_array_welch(
        data, sfreq, fmin=1.0, fmax=min(nyq - 1, max(cands) + 10),
        n_fft=n_fft, verbose="ERROR",
    )
    m = 10 * np.log10(psds.mean(axis=0) + 1e-30)

    def prominence(f0: float) -> float:
        peak = m[(freqs >= f0 - 2) & (freqs <= f0 + 2)].max()
        base = np.median(m[(freqs >= f0 - 8) & (freqs <= f0 + 8)])
        return float(peak - base)

    proms = {c: prominence(c) for c in cands}
    best = max(proms, key=proms.get)
    return best if proms[best] >= min_prominence else candidates[0]


def remove_line_noise(raw: mne.io.BaseRaw, cfg: PreprocConfig) -> mne.io.BaseRaw:
    """Remove line noise at the mains frequency and its harmonics (rmlinenoisemt).

    With ``cfg.line_freq == "auto"`` the mains frequency is detected from the
    data (50 vs 60 Hz), so US and European recordings are both handled.
    """
    nyq = raw.info["sfreq"] / 2.0
    line = cfg.line_freq
    if isinstance(line, str) or line is None:
        line = detect_line_frequency(raw)
    freqs = np.arange(line, nyq, line)
    if freqs.size:
        raw.notch_filter(freqs=freqs, verbose="ERROR")
    return raw


def average_reference(inst):
    """Re-reference to common average (rereference.m / martin_pipeline.m)."""
    inst.set_eeg_reference("average", projection=False, verbose="ERROR")
    return inst


def preprocess_raw(
    raw: mne.io.BaseRaw, cfg: PreprocConfig | None = None
) -> mne.io.BaseRaw:
    """Run the full continuous-data preprocessing chain."""
    cfg = cfg or PreprocConfig()
    resample(raw, cfg)
    # Notch before the band-pass so the mains frequency can be detected (the
    # 0.5-45 Hz low-pass would otherwise remove the 50/60 Hz peak first).
    remove_line_noise(raw, cfg)
    filter_band(raw, cfg)
    return raw
