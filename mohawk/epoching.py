"""Fixed-length epoching (port of ``epochdata.m`` and ``checktrials.m``)."""

from __future__ import annotations

import mne
import numpy as np

from .config import PreprocConfig


def make_epochs(
    raw: mne.io.BaseRaw, cfg: PreprocConfig | None = None
) -> mne.Epochs:
    """Segment continuous data into non-overlapping fixed-length epochs.

    Mirrors ``epochdata.m``: 10-second epochs with per-epoch baseline (mean)
    removal.
    """
    cfg = cfg or PreprocConfig()
    epochs = mne.make_fixed_length_epochs(
        raw, duration=cfg.epoch_length, preload=True,
        reject_by_annotation=True, verbose="ERROR",
    )
    epochs.apply_baseline((None, None), verbose="ERROR")
    return epochs


def check_trials(epochs: mne.Epochs, set_trials: int) -> mne.Epochs:
    """Retain a fixed number of the least-variable epochs (checktrials.m).

    If there are more than ``set_trials`` epochs, drops the highest-variance
    ones so that exactly ``set_trials`` remain.  If there are fewer, returns
    the epochs unchanged (with a warning-style no-op, as in the original).
    """
    n = len(epochs)
    if n <= set_trials:
        return epochs
    data = epochs.get_data(copy=False)  # (n_epochs, n_ch, n_times)
    flat = data.transpose(0, 1, 2).reshape(n, -1)
    trial_var = flat.var(axis=1, ddof=1)
    keep = np.argsort(trial_var)[:set_trials]
    keep = np.sort(keep)
    return epochs[keep]
