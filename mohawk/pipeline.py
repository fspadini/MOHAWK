"""End-to-end automated MOHAWK pipeline (port of ``mohawk.m``).

Runs the full single-subject analysis without any manual intervention:

    import -> montage -> preprocess -> epoch -> auto artefact reject ->
    auto ICA -> average reference -> retain N epochs -> multitaper spectrum ->
    dwPLI connectivity -> graph metrics -> figures.

The interactive bad-channel/ICA steps of the original MATLAB app are replaced
by automatic equivalents (see ``artifacts.py``), so the whole thing runs to
completion on a file path or an in-memory ``mne.io.Raw``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import mne
import numpy as np

from . import artifacts, connectivity, epoching, io, metrics, plotting, preprocess, spectrum
from .config import MohawkConfig


@dataclass
class PipelineResult:
    basename: str
    epochs: mne.Epochs
    spectrum: spectrum.SpectrumResult
    connectivity: connectivity.ConnectivityResult
    graph: metrics.GraphMetrics | None
    ica_excluded: list
    figures: dict = field(default_factory=dict)


def _positions(epochs: mne.Epochs) -> np.ndarray:
    """3D channel positions in the epochs' current channel order."""
    montage = epochs.get_montage()
    pos = montage.get_positions()["ch_pos"]
    return np.array([pos[ch] for ch in epochs.ch_names])


def run_pipeline(
    source,
    basename: str = "mohawk",
    outdir: str | None = None,
    montage: str | None = None,
    config: MohawkConfig | None = None,
    set_trials: int | None = 60,
    compute_graph: bool = True,
    make_figures: bool = True,
    plot_bands=(0, 1, 2),
    seed: int | None = 42,
) -> PipelineResult:
    """Run the complete automated pipeline.

    Parameters
    ----------
    source : str | mne.io.BaseRaw
        Path to an EEG file or an already-loaded Raw object.
    basename : str
        Base name for saved outputs.
    outdir : str | None
        Directory for figures/results; defaults to the current directory.
    set_trials : int | None
        Retain this many least-variable epochs (checktrials.m); None keeps all.
    plot_bands : sequence of int
        Band indices to render as head-network plots (default delta/theta/alpha).
    """
    config = config or MohawkConfig()
    outdir = outdir or os.getcwd()
    figdir = os.path.join(outdir, "figures")
    if make_figures:
        os.makedirs(figdir, exist_ok=True)

    # 1. Import + montage + peripheral channel removal
    if isinstance(source, mne.io.BaseRaw):
        raw = source.copy()
    else:
        raw = io.read_raw(source)
        io.strip_eeg_prefix(raw)
    if raw.get_montage() is None or montage is not None:
        io.set_montage(raw, montage)
    io.remove_peripheral_channels(raw)
    # drop flat / online-reference channels (e.g. EGI "REF CZ") before they
    # corrupt the variance statistics and the average reference
    dropped_ref = artifacts.drop_reference_channels(raw, config.artifact)

    # 2. Preprocess (resample, band-pass, line noise)
    preprocess.preprocess_raw(raw, config.preproc)

    # 3. Epoch
    epochs = epoching.make_epochs(raw, config.preproc)

    # 4. Automatic artefact rejection (bad channels/epochs) + ICA
    epochs = artifacts.reject_and_interpolate(epochs, config.artifact)
    epochs, _ica, ica_excluded = artifacts.run_ica(epochs, config.artifact)

    # 5. Average reference
    preprocess.average_reference(epochs)

    # 6. Retain a fixed number of epochs
    if set_trials is not None:
        epochs = epoching.check_trials(epochs, set_trials)

    # 7. Spectrum + band power
    spec = spectrum.calc_spectrum(epochs, config.spectrum, config.bands)

    # 8. Connectivity
    conn = connectivity.calc_connectivity(epochs, config.connectivity, config.bands)

    # 9. Graph metrics
    graph_metrics = None
    if compute_graph:
        pos = _positions(epochs)
        chandist = metrics.channel_distance(pos)
        graph_metrics = metrics.calc_graph_metrics(
            conn.matrix, chandist, config.graph,
            band_names=config.band_names, seed=seed,
        )

    figures: dict = {}
    if make_figures:
        pos = _positions(epochs)
        spec_path = os.path.join(figdir, f"{basename}_spec.png")
        plotting.plot_spectrum(spec, spec_path, basename)
        figures["spectrum"] = spec_path
        for b in plot_bands:
            name = config.band_names[b]
            # signature 3D "mohawk" topograph (plothead.m)
            mohawk_path = os.path.join(figdir, f"{basename}_{name}_mohawk.png")
            plotting.plot_mohawk_3d(
                conn, b, pos, config.plot, arcs="strength",
                outpath=mohawk_path, basename=basename, seed=seed,
            )
            figures[f"mohawk_{name}"] = mohawk_path
            # supplementary flat topographic network
            net_path = os.path.join(figdir, f"{basename}_{name}_network.png")
            plotting.plot_head_network(
                conn, b, pos, config.plot, arcs="strength",
                outpath=net_path, basename=basename, seed=seed,
            )
            figures[f"network_{name}"] = net_path
            mat_path = os.path.join(figdir, f"{basename}_{name}_conn.png")
            plotting.plot_connectivity_matrix(conn, b, mat_path, basename)
            figures[f"matrix_{name}"] = mat_path

    return PipelineResult(
        basename=basename,
        epochs=epochs,
        spectrum=spec,
        connectivity=conn,
        graph=graph_metrics,
        ica_excluded=ica_excluded,
        figures=figures,
    )


def save_results(result: PipelineResult, outdir: str | None = None) -> str:
    """Save numeric results to a compressed ``.npz`` file (analogous to *_mohawk.mat)."""
    outdir = outdir or os.getcwd()
    path = os.path.join(outdir, f"{result.basename}_mohawk.npz")
    payload = dict(
        freqs=result.spectrum.freqs,
        spectra=result.spectrum.spectra,
        band_power=result.spectrum.band_power,
        matrix=result.connectivity.matrix,
        chan_wpli=result.connectivity.chan_wpli,
        ch_names=np.array(result.spectrum.ch_names),
        bands=result.spectrum.bands,
    )
    if result.graph is not None:
        payload.update(
            tvals=result.graph.tvals,
            clustering=result.graph.clustering,
            char_path=result.graph.char_path,
            global_efficiency=result.graph.global_efficiency,
            modularity=result.graph.modularity,
            modules=result.graph.modules,
            participation=result.graph.participation,
            degree=result.graph.degree,
        )
    np.savez_compressed(path, **payload)
    return path
