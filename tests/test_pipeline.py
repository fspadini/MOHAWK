"""End-to-end pipeline integration tests (mohawk.pipeline)."""

import os

import numpy as np

from mohawk import run_pipeline, save_results
from mohawk.config import GraphConfig, MohawkConfig
from mohawk.datasets import synthetic_raw
from mohawk.epoching import check_trials, make_epochs
from mohawk.preprocess import preprocess_raw


def _fast_config():
    cfg = MohawkConfig()
    cfg.graph = GraphConfig(thresholds=np.array([1.0, 0.5]), heuristic=2)
    return cfg


def test_check_trials_retains_least_variable():
    raw = synthetic_raw(duration=80.0, n_channels=8, seed=0)
    preprocess_raw(raw)
    epochs = make_epochs(raw)
    n = len(epochs)
    assert n >= 5
    kept = check_trials(epochs, 4)
    assert len(kept) == 4


def test_pipeline_end_to_end_on_raw(tmp_path):
    raw = synthetic_raw(duration=70.0, n_channels=16, seed=1)
    res = run_pipeline(
        raw, basename="t", outdir=str(tmp_path), config=_fast_config(),
        set_trials=4, plot_bands=(2,), seed=1,
    )
    # spectrum / connectivity / graph shapes
    assert res.spectrum.band_power.shape[0] == 5
    assert res.connectivity.matrix.shape == (5, len(res.epochs.ch_names),
                                             len(res.epochs.ch_names))
    assert res.graph is not None
    # figures were produced on disk
    for path in res.figures.values():
        assert os.path.exists(path)
    # results saved and reloadable
    npz = save_results(res, str(tmp_path))
    data = np.load(npz, allow_pickle=True)
    assert "matrix" in data and "band_power" in data
    assert data["matrix"].shape == (5, len(res.epochs.ch_names),
                                    len(res.epochs.ch_names))


def test_pipeline_from_fif_file(tmp_path):
    """Round-trip through a real file to exercise the IO path."""
    raw = synthetic_raw(duration=70.0, n_channels=16, seed=2)
    fif = tmp_path / "sub_raw.fif"
    raw.save(fif, overwrite=True, verbose="ERROR")
    res = run_pipeline(
        str(fif), basename="sub", outdir=str(tmp_path), config=_fast_config(),
        set_trials=4, compute_graph=False, plot_bands=(2,), seed=2,
    )
    assert res.connectivity.matrix.shape[0] == 5
    assert os.path.exists(res.figures["spectrum"])


def test_pipeline_no_figures_no_graph(tmp_path):
    raw = synthetic_raw(duration=50.0, n_channels=12, seed=3)
    res = run_pipeline(
        raw, basename="q", outdir=str(tmp_path), config=_fast_config(),
        set_trials=3, compute_graph=False, make_figures=False, seed=3,
    )
    assert res.graph is None
    assert res.figures == {}
