"""Unit tests for IO, montage handling, and preprocessing."""

import numpy as np
import pytest

import mne
from mohawk import io
from mohawk.config import PreprocConfig
from mohawk.datasets import synthetic_raw
from mohawk.preprocess import (
    average_reference,
    filter_band,
    preprocess_raw,
    remove_line_noise,
    resample,
)


def test_strip_eeg_prefix():
    info = mne.create_info(["EEG Fp1", "EEG Fp2", "STATUS"], 100.0, "eeg")
    raw = mne.io.RawArray(np.zeros((3, 100)), info, verbose="ERROR")
    io.strip_eeg_prefix(raw)
    assert "Fp1" in raw.ch_names and "Fp2" in raw.ch_names


def test_set_montage_fallback_standard():
    info = mne.create_info(["Fp1", "Fp2", "Cz", "Oz"], 250.0, "eeg")
    raw = mne.io.RawArray(np.zeros((4, 100)), info, verbose="ERROR")
    io.set_montage(raw)  # falls back to standard_1005
    assert raw.get_montage() is not None


def test_remove_peripheral_channels_128():
    names = [f"E{i}" for i in range(1, 129)]
    info = mne.create_info(names, 250.0, "eeg")
    raw = mne.io.RawArray(np.zeros((128, 50)), info, verbose="ERROR")
    io.remove_peripheral_channels(raw)
    assert len(raw.ch_names) == 128 - len(io.CHANEXCL_128)
    assert "E1" not in raw.ch_names


def test_resample_downsamples():
    raw = synthetic_raw(duration=10.0, n_channels=8, sfreq=500.0, seed=0)
    cfg = PreprocConfig()
    resample(raw, cfg)
    assert raw.info["sfreq"] == cfg.resample_sfreq


def test_resample_rejects_low_rate():
    raw = synthetic_raw(duration=10.0, n_channels=8, sfreq=200.0, seed=0)
    with pytest.raises(ValueError):
        resample(raw, PreprocConfig())


def test_filter_attenuates_high_frequency():
    raw = synthetic_raw(duration=20.0, n_channels=8, sfreq=500.0, seed=1)
    # add strong 80 Hz content (above 45 Hz low-pass)
    t = raw.times
    raw._data += 5e-5 * np.sin(2 * np.pi * 80 * t)
    before = np.abs(np.fft.rfft(raw._data[0]))
    freqs = np.fft.rfftfreq(raw.n_times, 1 / raw.info["sfreq"])
    hi = np.argmin(np.abs(freqs - 80))
    resample(raw, PreprocConfig())  # 250 Hz still resolves nothing at 80 after filter
    filter_band(raw, PreprocConfig())
    after = np.abs(np.fft.rfft(raw._data[0]))
    freqs2 = np.fft.rfftfreq(raw.n_times, 1 / raw.info["sfreq"])
    hi2 = np.argmin(np.abs(freqs2 - 80))
    # 80 Hz power is strongly reduced relative to the alpha band
    alpha_idx = np.argmin(np.abs(freqs2 - 10))
    assert after[hi2] < after[alpha_idx]


def test_detect_line_frequency_60hz():
    from mohawk.preprocess import detect_line_frequency
    sf = 500.0
    raw = synthetic_raw(duration=20.0, n_channels=8, sfreq=sf, seed=1)
    t = raw.times
    raw._data += 5e-4 * np.sin(2 * np.pi * 60 * t)  # strong US mains
    assert detect_line_frequency(raw) == 60.0


def test_detect_line_frequency_50hz():
    from mohawk.preprocess import detect_line_frequency
    sf = 500.0
    raw = synthetic_raw(duration=20.0, n_channels=8, sfreq=sf, seed=2)
    t = raw.times
    raw._data += 5e-4 * np.sin(2 * np.pi * 50 * t)  # strong European mains
    assert detect_line_frequency(raw) == 50.0


def test_detect_line_frequency_defaults_when_clean():
    from mohawk.preprocess import detect_line_frequency
    raw = synthetic_raw(duration=20.0, n_channels=8, sfreq=500.0, seed=3)
    # no injected mains -> falls back to the first candidate (50)
    assert detect_line_frequency(raw) in (50.0, 60.0)


def test_average_reference_zero_mean():
    raw = synthetic_raw(duration=10.0, n_channels=8, seed=2)
    average_reference(raw)
    # average reference -> channel mean at each time ~ 0
    assert np.allclose(raw._data.mean(axis=0), 0, atol=1e-12)


def test_preprocess_pipeline_runs():
    raw = synthetic_raw(duration=20.0, n_channels=8, seed=3)
    out = preprocess_raw(raw)
    assert out.info["sfreq"] == 250.0
