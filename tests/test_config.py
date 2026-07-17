"""Tests for YAML config loading and derived options."""

import numpy as np

from mohawk.config import config_from_dict, load_config


def test_config_from_dict_overrides():
    cfg = config_from_dict({
        "preproc": {"line_freq": 60.0},
        "spectrum": {"relative_power_nbands": 3},
        "artifact": {"ica_method": "infomax"},
        "densities": [0.9, 0.5, 0.1],
        "bands": [[0, 4], [4, 8], [8, 13]],
        "band_names": ["delta", "theta", "alpha"],
    })
    assert cfg.preproc.line_freq == 60.0
    assert cfg.spectrum.relative_power_nbands == 3
    assert cfg.artifact.ica_method == "infomax"
    assert len(cfg.graph.thresholds) == 3
    assert cfg.bands.shape == (3, 2)
    assert cfg.band_names == ("delta", "theta", "alpha")


def test_config_from_dict_ignores_unknown_keys():
    cfg = config_from_dict({"preproc": {"line_freq": 50.0, "nonsense": 1}})
    assert cfg.preproc.line_freq == 50.0


def test_load_paper_yaml():
    # the shipped paper config parses and reflects the paper's choices
    cfg = load_config("config/paper.yml")
    assert cfg.bands.shape[0] == 3
    assert len(cfg.graph.thresholds) == 33
    assert cfg.spectrum.relative_power_nbands == 3
    assert cfg.artifact.ica_method == "infomax"
    assert np.isclose(cfg.graph.thresholds[0], 0.9)
