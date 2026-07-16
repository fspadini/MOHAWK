"""MOHAWK: Python/MNE reimplementation of the hdEEG brain-network pipeline.

Reference: Chennu et al. (2017), *Brain* 140:2120-2132.

Typical use::

    from mohawk import run_pipeline
    result = run_pipeline("subject.edf", basename="subject", outdir="out")

or on the built-in synthetic data::

    from mohawk.datasets import synthetic_raw
    from mohawk import run_pipeline
    result = run_pipeline(synthetic_raw(), basename="demo")
"""

from __future__ import annotations

from .config import (
    BAND_NAMES,
    FREQ_BANDS,
    MohawkConfig,
)
from .pipeline import PipelineResult, run_pipeline, save_results

__version__ = "0.1.0"

__all__ = [
    "run_pipeline",
    "save_results",
    "PipelineResult",
    "MohawkConfig",
    "FREQ_BANDS",
    "BAND_NAMES",
    "__version__",
]
