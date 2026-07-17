"""Unit tests for automatic artefact rejection (mohawk.artifacts)."""

import numpy as np
import pytest

import mne
from mohawk import artifacts
from mohawk.config import ArtifactConfig
from mohawk.datasets import synthetic_raw
from mohawk.epoching import make_epochs
from mohawk.preprocess import preprocess_raw


def _epochs_with(n_channels=16, seed=0):
    raw = synthetic_raw(duration=60.0, n_channels=n_channels, seed=seed)
    preprocess_raw(raw)
    return make_epochs(raw)


def test_drop_reference_channels_flat_and_named():
    info = mne.create_info(["1", "2", "3", "REF CZ"], 250.0, "eeg")
    data = np.random.default_rng(0).standard_normal((4, 500)) * 1e-6
    data[3] = 0.0  # flat reference
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    dropped = artifacts.drop_reference_channels(raw, ArtifactConfig())
    assert "REF CZ" in dropped
    assert "REF CZ" not in raw.ch_names
    assert len(raw.ch_names) == 3


def test_drop_reference_can_be_disabled():
    info = mne.create_info(["1", "REF CZ"], 250.0, "eeg")
    data = np.ones((2, 200)) * 1e-6
    data[1] = 0.0
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    cfg = ArtifactConfig(drop_reference=False)
    assert artifacts.drop_reference_channels(raw, cfg) == []
    assert len(raw.ch_names) == 2


def test_frontal_proxy_matches_prefixed_name():
    info = mne.create_info(["1", "2", "129 FP1", "REF CZ"], 250.0, "eeg")
    raw = mne.io.RawArray(np.zeros((4, 100)), info, verbose="ERROR")
    ep = mne.make_fixed_length_epochs(raw, duration=0.2, verbose="ERROR")
    assert artifacts._frontal_proxy(ep) == "129 FP1"


def test_frontal_proxy_fallback_to_anterior():
    # generic names with no known frontal token -> most anterior (max +y)
    names = [f"c{i}" for i in range(6)]
    info = mne.create_info(names, 250.0, "eeg")
    raw = mne.io.RawArray(np.zeros((6, 100)), info, verbose="ERROR")
    ch_pos = {n: np.array([0.0, y, 0.0]) for n, y in zip(names, [0, 1, 2, 3, 4, 5])}
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    raw.set_montage(montage, verbose="ERROR")
    ep = mne.make_fixed_length_epochs(raw, duration=0.2, verbose="ERROR")
    assert artifacts._frontal_proxy(ep) == "c5"  # largest +y


def test_detect_bad_channels_catches_high_variance():
    # uniform white-noise channels + one loud channel -> clear outlier
    rng = np.random.default_rng(0)
    n_ep, n_ch, n_t = 20, 16, 250
    data = rng.standard_normal((n_ep, n_ch, n_t)) * 1e-6
    data[:, 5, :] *= 20.0  # inject a very noisy channel
    info = mne.create_info([f"c{i}" for i in range(n_ch)], 250.0, "eeg")
    ep = mne.EpochsArray(data, info, verbose="ERROR")
    bad = artifacts.detect_bad_channels(ep, ArtifactConfig())
    assert "c5" in bad


def test_reject_and_interpolate_protects_frontal(monkeypatch):
    ep = _epochs_with(16, seed=3)
    proxy = artifacts._frontal_proxy(ep)
    # force the proxy to look "bad" and confirm it is protected from the drop
    monkeypatch.setattr(artifacts, "detect_bad_channels", lambda e, c=None: [proxy])
    out = artifacts.reject_and_interpolate(ep, ArtifactConfig(), protect_frontal=True)
    assert proxy in out.ch_names
    assert out.info["bads"] == []


def test_reject_and_interpolate_runs():
    ep = _epochs_with(16, seed=4)
    out = artifacts.reject_and_interpolate(ep, ArtifactConfig())
    assert len(out.ch_names) == len(ep.ch_names)  # bads interpolated, not dropped
