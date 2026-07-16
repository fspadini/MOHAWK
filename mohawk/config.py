"""Configuration constants for the MOHAWK pipeline.

These mirror the hard-coded parameters scattered across the original MATLAB
sources (``dataimport.m``, ``epochdata.m``, ``calcftspec.m``,
``ftcoherence.m``, ``calcgraph.m``, ``freqlist.mat``) so that the Python
port reproduces the same analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Canonical frequency bands, from freqlist.mat.  Order matters: band index 1
# is delta, 2 theta, 3 alpha, 4 beta, 5 gamma (see plothead.m / plotftspec.m).
BAND_NAMES = ("delta", "theta", "alpha", "beta", "gamma")
FREQ_BANDS = np.array(
    [
        [0.0, 4.0],   # delta
        [4.0, 8.0],   # theta
        [8.0, 13.0],  # alpha
        [13.0, 30.0], # beta
        [30.0, 45.0], # gamma
    ]
)


@dataclass
class PreprocConfig:
    """Preprocessing parameters (dataimport.m / epochdata.m)."""

    resample_sfreq: float = 250.0      # pop_resample(EEG, 250)
    hp_freq: float = 0.5               # high-pass, hpfreq in dataimport.m
    lp_freq: float = 45.0              # low-pass, lpfreq in dataimport.m
    line_freq: float = 50.0            # rmlinenoisemt(EEG, 50)
    epoch_length: float = 10.0         # epochlength in epochdata.m (seconds)


@dataclass
class SpectrumConfig:
    """Multitaper spectrum parameters (calcftspec.m)."""

    fmin: float = 0.5                  # cfg.foilim(1)
    fmax: float = 45.0                 # cfg.foilim(2)
    bandwidth: float = 0.6             # 2 * cfg.tapsmofrq (0.3) -> full bandwidth


@dataclass
class ConnectivityConfig:
    """dwPLI connectivity parameters (ftcoherence.m)."""

    fmin: float = 0.5
    fmax: float = 45.0
    bandwidth: float = 0.6             # matches SpectrumConfig


@dataclass
class GraphConfig:
    """Graph-theoretic metric parameters (calcgraph.m)."""

    # Proportional density thresholds: 1 -> keep all edges, step -0.025 to 0.1.
    thresholds: np.ndarray = field(
        default_factory=lambda: np.round(np.arange(1.0, 0.1 - 1e-9, -0.025), 5)
    )
    heuristic: int = 50                # community_louvain repetitions


@dataclass
class ArtifactConfig:
    """Automatic artefact rejection parameters.

    The original pipeline used two interactive ``rejartifacts`` passes (z-score
    variance thresholding, threshold 4) plus manual ICA component selection.
    For a fully automated pipeline we replicate the variance thresholding and
    add automatic ICA artefact detection.
    """

    var_zthresh: float = 4.0           # rejartifacts z threshold
    max_bad_channel_frac: float = 0.25 # safety cap on channels removed
    n_ica_components: float = 0.99     # explained-variance for ICA (computeic)
    ica_random_state: int = 42


@dataclass
class PlotConfig:
    """3D topograph / spectrum plotting parameters (plothead.m/plotgraph3d.m)."""

    plot_quantile: float = 0.3         # plotqt: proportion of strongest edges
    erange: tuple = (0.0, 1.25)        # edge weight rescale range
    vrange: tuple = (0.0, 1.25)        # vertex weight rescale range
    num_colors: int = 4                # plothead numcolors


@dataclass
class MohawkConfig:
    """Top-level configuration bundle for the whole pipeline."""

    preproc: PreprocConfig = field(default_factory=PreprocConfig)
    spectrum: SpectrumConfig = field(default_factory=SpectrumConfig)
    connectivity: ConnectivityConfig = field(default_factory=ConnectivityConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    artifact: ArtifactConfig = field(default_factory=ArtifactConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    bands: np.ndarray = field(default_factory=lambda: FREQ_BANDS.copy())
    band_names: tuple = BAND_NAMES
