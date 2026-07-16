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


def remove_line_noise(raw: mne.io.BaseRaw, cfg: PreprocConfig) -> mne.io.BaseRaw:
    """Remove line noise at ``cfg.line_freq`` and harmonics (rmlinenoisemt)."""
    nyq = raw.info["sfreq"] / 2.0
    freqs = np.arange(cfg.line_freq, nyq, cfg.line_freq)
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
    filter_band(raw, cfg)
    remove_line_noise(raw, cfg)
    return raw
