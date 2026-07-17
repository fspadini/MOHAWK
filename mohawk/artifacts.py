"""Automatic artefact rejection (replaces interactive rejartifacts / rejectic).

The original MOHAWK pipeline required two manual passes of variance-based bad
channel / epoch selection (``rejartifacts``) and manual ICA component review
(``rejectic``).  To run fully automatically we reproduce the variance
z-score logic and add automatic ICA artefact-component detection.
"""

from __future__ import annotations

import mne
import numpy as np
from mne.preprocessing import ICA

from .config import ArtifactConfig


def _robust_z(x: np.ndarray) -> np.ndarray:
    """Median/MAD-based robust z-score."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        std = x.std()
        return (x - med) / std if std > 0 else np.zeros_like(x)
    return (x - med) / (1.4826 * mad)


def suggest_variance_outliers(
    epochs: mne.Epochs, cfg: ArtifactConfig | None = None
) -> tuple[list[str], list[int]]:
    """Suggest noisy channels and epochs *without mutating the data*.

    The original paper required visual confirmation of every rejected channel,
    epoch and ICA component. This mirrors that workflow: it returns a review
    list (bad channels, bad epochs) that a human can inspect and pass back to
    ``run_pipeline`` in faithful mode, rather than rejecting anything itself.
    """
    cfg = cfg or ArtifactConfig()
    bad_ch = detect_bad_channels(epochs, cfg)
    for nb in detect_narrowband_channels(epochs, cfg):
        if nb not in bad_ch:
            bad_ch.append(nb)
    bad_ep = detect_bad_epochs(epochs, cfg)
    return bad_ch, bad_ep


def drop_reference_channels(inst, cfg: ArtifactConfig | None = None) -> list[str]:
    """Drop flat / reference channels (e.g. an all-zero ``"REF CZ"``).

    Modern EGI nets carry the online reference as a flat channel; it must be
    removed before it corrupts the variance statistics, the average reference,
    and connectivity. Returns the dropped channel names.
    """
    cfg = cfg or ArtifactConfig()
    if not cfg.drop_reference:
        return []
    data = inst.get_data()
    std = data.std(axis=-1)
    if data.ndim == 3:  # epochs: (n_epochs, n_ch, n_times)
        std = data.std(axis=(0, 2))
    flat = std < 1e-15
    named_ref = np.array([
        ("REF" in ch.upper()) or ("VREF" in ch.upper()) for ch in inst.ch_names
    ])
    drop = [inst.ch_names[i] for i in np.flatnonzero(flat | named_ref)]
    if drop:
        inst.drop_channels(drop)
    return drop


def detect_bad_channels(
    epochs: mne.Epochs, cfg: ArtifactConfig | None = None
) -> list[str]:
    """Detect noisy channels by variance z-scoring (rejartifacts).

    A channel is flagged if the robust z-score of its log-variance is an
    outlier beyond ``cfg.var_zthresh`` (either tail), or beyond the stricter
    one-sided ``cfg.hi_var_zthresh`` on the *high* side (noisy channels are more
    common than pathologically quiet ones), or if it is effectively flat. The
    fraction removed is capped at ``cfg.max_bad_channel_frac``.
    """
    cfg = cfg or ArtifactConfig()
    data = epochs.get_data(copy=False)  # (n_epochs, n_ch, n_times)
    # variance per channel over concatenated epochs
    var = data.transpose(1, 0, 2).reshape(data.shape[1], -1).var(axis=1)
    logvar = np.log(var + np.finfo(float).eps)
    z = _robust_z(logvar)

    flat = var < (np.median(var) * 1e-3)
    bad_mask = (np.abs(z) > cfg.var_zthresh) | (z > cfg.hi_var_zthresh) | flat

    # Cap how many channels we are willing to drop.
    max_bad = int(np.floor(len(epochs.ch_names) * cfg.max_bad_channel_frac))
    if bad_mask.sum() > max_bad:
        # keep the most extreme outliers only
        order = np.argsort(-np.abs(z))
        keep_bad = order[:max_bad]
        new_mask = np.zeros_like(bad_mask)
        new_mask[keep_bad] = True
        bad_mask = new_mask
    return [epochs.ch_names[i] for i in np.flatnonzero(bad_mask)]


def detect_narrowband_channels(
    epochs: mne.Epochs, cfg: ArtifactConfig | None = None
) -> list[str]:
    """Detect channels with a narrowband spectral peak the others do not share.

    Single-channel line-like artefacts (e.g. a ~25 Hz electronic spike) barely
    raise a channel's broadband variance, so ``detect_bad_channels`` misses
    them.  Here the per-channel power spectrum is compared to the across-channel
    consensus (median at each frequency): a channel is flagged if its power
    exceeds the consensus by more than ``cfg.narrowband_db`` at some frequency
    *and* that excess is a robust-z outlier across channels.  Because genuine
    rhythms such as posterior alpha are present on (most) channels, they sit
    near the consensus and are not flagged.
    """
    cfg = cfg or ArtifactConfig()
    if not cfg.detect_narrowband:
        return []
    from mne.time_frequency import psd_array_welch
    from scipy.ndimage import median_filter

    data = epochs.get_data(copy=False)  # (n_epochs, n_ch, n_times)
    sfreq = epochs.info["sfreq"]
    if data.shape[1] < 4:
        return []  # need enough channels to form a consensus

    n_per_seg = int(min(data.shape[-1], sfreq * 2))  # ~0.5 Hz resolution
    psds, freqs = psd_array_welch(
        data, sfreq, fmin=1.0, fmax=min(45.0, sfreq / 2 - 1),
        n_fft=n_per_seg, n_per_seg=n_per_seg, verbose="ERROR",
    )
    psd = psds.mean(axis=0)  # (n_ch, n_freq)
    logpsd = 10 * np.log10(psd + 1e-30)

    # Isolate NARROW peaks: subtract each channel's own frequency-smoothed
    # baseline. This removes the smooth 1/f slope and broad rhythms (e.g. the
    # several-Hz-wide alpha peak), leaving only narrow, line-like excesses.
    df = float(np.median(np.diff(freqs)))
    win = max(3, int(round(4.0 / df)) | 1)  # ~4 Hz window, odd length
    baseline = median_filter(logpsd, size=(1, win), mode="nearest")
    resid = logpsd - baseline
    resid[resid < 0] = 0.0
    peak_resid = resid.max(axis=1)  # strongest narrowband peak per channel (dB)

    z = _robust_z(peak_resid)
    bad_mask = (peak_resid > cfg.narrowband_db) & (z > cfg.narrowband_zthresh)

    max_bad = int(np.floor(len(epochs.ch_names) * cfg.max_bad_channel_frac))
    if bad_mask.sum() > max_bad:
        order = np.argsort(-peak_resid)
        keep = order[:max_bad]
        new_mask = np.zeros_like(bad_mask)
        new_mask[keep] = True
        bad_mask = new_mask
    return [epochs.ch_names[i] for i in np.flatnonzero(bad_mask)]


def detect_bad_epochs(
    epochs: mne.Epochs, cfg: ArtifactConfig | None = None
) -> list[int]:
    """Detect noisy epochs by variance z-scoring (rejartifacts, threshold 4)."""
    cfg = cfg or ArtifactConfig()
    data = epochs.get_data(copy=False)
    ep_var = data.reshape(data.shape[0], -1).var(axis=1)
    z = _robust_z(np.log(ep_var + np.finfo(float).eps))
    return list(np.flatnonzero(z > cfg.epoch_zthresh))


def reject_and_interpolate(
    epochs: mne.Epochs, cfg: ArtifactConfig | None = None,
    protect_frontal: bool = True,
    auto: bool = True,
    manual_bads: "list[str] | tuple" = (),
) -> mne.Epochs:
    """Mark bad channels, interpolate them, and drop bad epochs.

    Combines both ``rejartifacts`` passes: bad channels are interpolated
    (requires a montage) and high-variance epochs are dropped.

    Frontal channels are legitimately high-variance because of eye blinks, so
    when ``protect_frontal`` is set the EOG-proxy channel is kept out of the
    *auto-detected* bad list — otherwise interpolation would erase the very
    blink signal that the ICA step relies on to detect ocular components.

    ``auto`` toggles automatic detection; ``manual_bads`` are channels the
    caller has selected by hand (always interpolated, and never protected).
    In faithful mode set ``auto=False`` and pass reviewed ``manual_bads``.
    """
    cfg = cfg or ArtifactConfig()
    epochs = epochs.copy()

    bad_ch: list[str] = []
    if auto:
        bad_ch = detect_bad_channels(epochs, cfg)
        for nb in detect_narrowband_channels(epochs, cfg):
            if nb not in bad_ch:
                bad_ch.append(nb)
        if protect_frontal:
            proxy = _frontal_proxy(epochs)
            bad_ch = [b for b in bad_ch if b != proxy]
    for mb in manual_bads:
        if mb in epochs.ch_names and mb not in bad_ch:
            bad_ch.append(mb)
    if bad_ch:
        epochs.info["bads"] = bad_ch
        has_pos = epochs.get_montage() is not None
        if has_pos:
            epochs.interpolate_bads(reset_bads=True, verbose="ERROR")
        else:
            epochs.drop_channels(bad_ch)

    if auto:
        bad_ep = detect_bad_epochs(epochs, cfg)
        if bad_ep:
            epochs.drop(bad_ep, reason="VARIANCE", verbose="ERROR")

    return epochs


def run_ica(
    epochs: mne.Epochs, cfg: ArtifactConfig | None = None,
    auto: bool = True, exclude: "list[int] | None" = None,
) -> tuple[mne.Epochs, ICA, list[int]]:
    """Fit ICA and remove artefact components (computeic/rejectic).

    In automatic mode (``auto=True`` and no explicit ``exclude``), components
    are detected with MNE's muscle-artefact heuristic plus a frontal EOG proxy.
    In faithful mode pass reviewed component indices via ``exclude`` (and
    optionally ``auto=False``); ICA is fit and exactly those are removed.
    Returns the cleaned epochs, the fitted ICA, and the excluded indices.
    """
    cfg = cfg or ArtifactConfig()
    n_ch = len(mne.pick_types(epochs.info, eeg=True))
    # keep n_components below rank to stay stable
    n_components = min(cfg.n_ica_components, n_ch - 1) if isinstance(
        cfg.n_ica_components, int
    ) else cfg.n_ica_components

    fit_params = {"extended": cfg.ica_extended} if cfg.ica_method == "infomax" else None
    ica = ICA(
        n_components=n_components,
        method=cfg.ica_method,
        fit_params=fit_params,
        random_state=cfg.ica_random_state,
        max_iter="auto",
        verbose="ERROR",
    )
    ica.fit(epochs, verbose="ERROR")

    chosen: set[int] = set()
    if exclude is not None:
        # faithful mode: use exactly the reviewed component indices
        chosen.update(int(i) for i in exclude if 0 <= int(i) < ica.n_components_)
    elif auto:
        # Muscle artefacts (spectral slope heuristic).
        try:
            muscle_idx, _ = ica.find_bads_muscle(epochs, verbose="ERROR")
            chosen.update(muscle_idx)
        except Exception:
            pass
        # Eye-movement artefacts via a frontal channel as EOG proxy.
        frontal = _frontal_proxy(epochs)
        if frontal is not None:
            try:
                eog_idx, _ = ica.find_bads_eog(
                    epochs, ch_name=frontal, threshold=cfg.eog_threshold,
                    verbose="ERROR",
                )
                chosen.update(eog_idx)
            except Exception:
                pass

    ica.exclude = sorted(chosen)
    cleaned = ica.apply(epochs.copy(), verbose="ERROR")
    return cleaned, ica, ica.exclude


def _frontal_proxy(epochs: mne.Epochs) -> str | None:
    """Pick a frontal channel to serve as an EOG proxy, if available.

    Handles arbitrary naming (``Fp1``, ``FP1``, ``129 FP1``, EGI ``E22`` …) by
    substring match, then falls back to the most anterior channel in the
    montage so blink detection still runs on nets without named frontal sites.
    """
    tokens = ("FP1", "FPZ", "FP2", "AF7", "AF8", "AFZ", "FZ")
    compact = {ch: ch.upper().replace(" ", "") for ch in epochs.ch_names}
    for token in tokens:
        for ch, name in compact.items():
            if token in name:
                return ch
    for ch in ("E22", "E9", "E14", "E15"):  # EGI frontopolar sites
        if ch in epochs.ch_names:
            return ch
    # fallback: most anterior (max +y) channel from the montage
    montage = epochs.get_montage()
    if montage is not None:
        pos = montage.get_positions()["ch_pos"]
        best, best_y = None, -np.inf
        for ch in epochs.ch_names:
            p = pos.get(ch)
            if p is not None and np.isfinite(p).all() and p[1] > best_y:
                best_y, best = p[1], ch
        return best
    return None
