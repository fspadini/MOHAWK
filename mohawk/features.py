"""Serializable subject-level features for group analysis.

Bundles the per-subject spectral and network metrics that the paper's
group-level ROC screening and classification operate on, with ``save``/``load``
to a compressed ``.npz`` and accessors that select a metric by band and
network density.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class SubjectFeatures:
    ch_names: tuple
    band_names: tuple
    densities: np.ndarray             # (n_thresh,)
    relative_power: np.ndarray        # (n_bands, n_ch)
    connectivity: np.ndarray          # (n_bands, n_ch, n_ch)
    clustering: np.ndarray            # (n_bands, n_thresh, n_ch)
    path_length: np.ndarray           # (n_bands, n_thresh)
    modularity: np.ndarray            # (n_bands, n_thresh)
    participation: np.ndarray         # (n_bands, n_thresh, n_ch)
    modular_span: np.ndarray          # (n_bands, n_thresh)

    # ---- construction ----------------------------------------------------
    @classmethod
    def from_result(cls, result) -> "SubjectFeatures":
        """Build from a :class:`mohawk.pipeline.PipelineResult` (graph required)."""
        if result.graph is None:
            raise ValueError("PipelineResult has no graph metrics; run with compute_graph=True")
        g = result.graph
        n_bands = result.spectrum.bands.shape[0]
        band_names = tuple(g.band_names) if g.band_names else tuple(
            f"band{i}" for i in range(n_bands)
        )
        return cls(
            ch_names=tuple(result.spectrum.ch_names),
            band_names=band_names,
            densities=np.asarray(g.tvals),
            relative_power=result.spectrum.relative_power,
            connectivity=result.connectivity.matrix,
            clustering=g.clustering,
            path_length=g.char_path,
            modularity=g.modularity,
            participation=g.participation,
            modular_span=g.modular_span,
        )

    # ---- (de)serialization ----------------------------------------------
    def save(self, path: str | Path) -> str:
        np.savez_compressed(
            path,
            ch_names=np.asarray(self.ch_names),
            band_names=np.asarray(self.band_names),
            densities=self.densities,
            relative_power=self.relative_power,
            connectivity=self.connectivity,
            clustering=self.clustering,
            path_length=self.path_length,
            modularity=self.modularity,
            participation=self.participation,
            modular_span=self.modular_span,
        )
        return str(path)

    @classmethod
    def load(cls, path: str | Path) -> "SubjectFeatures":
        with np.load(path, allow_pickle=False) as d:
            return cls(
                ch_names=tuple(d["ch_names"].astype(str)),
                band_names=tuple(d["band_names"].astype(str)),
                densities=d["densities"],
                relative_power=d["relative_power"],
                connectivity=d["connectivity"],
                clustering=d["clustering"],
                path_length=d["path_length"],
                modularity=d["modularity"],
                participation=d["participation"],
                modular_span=d["modular_span"],
            )

    # ---- accessors -------------------------------------------------------
    def _band(self, band: str) -> int:
        return self.band_names.index(band)

    def _density_index(self, density: float) -> int:
        return int(np.argmin(np.abs(self.densities - density)))

    def scalar_metric(self, metric: str, band: str, density: float | None = None) -> float:
        """Scalar summary used for group ROC screening."""
        b = self._band(band)
        if metric == "relative_power":
            return float(self.relative_power[b].mean())
        if metric == "median_connectivity":
            iu = np.triu_indices(len(self.ch_names), 1)
            return float(np.median(self.connectivity[b][iu]))
        if density is None:
            raise ValueError(f"{metric} requires a density")
        t = self._density_index(density)
        if metric == "clustering":
            return float(self.clustering[b, t].mean())
        if metric == "path_length":
            return float(self.path_length[b, t])
        if metric == "modularity":
            return float(self.modularity[b, t])
        if metric == "participation":
            return float(self.participation[b, t].std())
        if metric == "modular_span":
            return float(self.modular_span[b, t])
        raise KeyError(metric)

    def node_feature(self, metric: str, band: str, density: float | None = None) -> np.ndarray:
        """Channel-wise feature vector used as classifier input."""
        b = self._band(band)
        if metric == "relative_power":
            return self.relative_power[b].copy()
        if density is None:
            raise ValueError(f"{metric} requires a density")
        t = self._density_index(density)
        if metric == "clustering":
            return self.clustering[b, t].copy()
        if metric == "participation":
            return self.participation[b, t].copy()
        return np.asarray([self.scalar_metric(metric, band, density)])
