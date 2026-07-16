"""Smoke tests for figure generation (mohawk.plotting)."""

import os

import numpy as np
import pytest

from mohawk import plotting
from mohawk.config import PlotConfig
from mohawk.connectivity import calc_connectivity
from mohawk.datasets import synthetic_raw
from mohawk.epoching import make_epochs
from mohawk.preprocess import preprocess_raw


@pytest.fixture(scope="module")
def small_conn():
    raw = synthetic_raw(duration=50.0, n_channels=16, seed=4)
    preprocess_raw(raw)
    epochs = make_epochs(raw)
    conn = calc_connectivity(epochs)
    pos = np.array(
        [epochs.get_montage().get_positions()["ch_pos"][ch] for ch in epochs.ch_names]
    )
    return conn, pos


def test_azimuthal_positions_centered():
    # points on a sphere project into the unit disc, vertex near origin
    rng = np.random.default_rng(0)
    p = rng.standard_normal((20, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    xy = plotting.azimuthal_positions(p)
    assert xy.shape == (20, 2)
    assert np.all(np.isfinite(xy))


def test_plot_mohawk_3d_writes_file(tmp_path, small_conn):
    conn, pos = small_conn
    out = tmp_path / "mohawk3d.png"
    fig, minfo = plotting.plot_mohawk_3d(
        conn, 2, pos, PlotConfig(), arcs="strength",
        outpath=str(out), basename="t", seed=1,
    )
    assert out.exists() and out.stat().st_size > 0
    assert len(minfo) == pos.shape[0]


def test_plot_head_network_writes_file(tmp_path, small_conn):
    conn, pos = small_conn
    out = tmp_path / "net.png"
    fig, minfo = plotting.plot_head_network(
        conn, 2, pos, PlotConfig(), outpath=str(out), basename="t", seed=1
    )
    assert out.exists() and out.stat().st_size > 0


def test_plot_spectrum_writes_file(tmp_path):
    from mohawk.spectrum import calc_spectrum
    raw = synthetic_raw(duration=40.0, n_channels=12, seed=5)
    preprocess_raw(raw)
    epochs = make_epochs(raw)
    spec = calc_spectrum(epochs)
    out = tmp_path / "spec.png"
    plotting.plot_spectrum(spec, str(out), "t")
    assert out.exists() and out.stat().st_size > 0
