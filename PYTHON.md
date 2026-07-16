# MOHAWK — Python / MNE port

A Python reimplementation of the MOHAWK hdEEG brain-network pipeline
(Chennu et al., 2017, *Brain*), built on [MNE-Python](https://mne.tools) and
[MNE-Connectivity](https://mne.tools/mne-connectivity). It reproduces the core
single-subject analysis of the original MATLAB app — from raw import to the
signature 3D-style connectivity "mohawk" topograph — and runs **fully
automatically**, with no interactive bad-channel or ICA-component selection.

The original MATLAB sources remain in this repository unchanged; this port
lives entirely in the `mohawk/` Python package.

## What the pipeline does

```
import → montage → preprocess → epoch → auto artefact reject →
auto ICA → average reference → retain N epochs →
multitaper spectrum → dwPLI connectivity → graph metrics → figures
```

1. **Import** EEG in EGI RAW/MFF, EDF, BrainVision, EEGLAB `.set`, or FIF.
2. **Montage** — GSN-HydroCel by channel count for EGI nets, else the 10-5
   `standard_1005` montage; MOHAWK's peripheral electrodes are dropped for
   128/256-channel nets.
3. **Preprocess** — resample to 250 Hz, band-pass 0.5–45 Hz, remove 50 Hz line
   noise (and harmonics).
4. **Epoch** into 10-second segments with per-epoch baseline removal.
5. **Automatic artefact rejection** — variance z-score bad-channel/epoch
   detection (interpolating bad channels), replacing the interactive
   `rejartifacts` passes.
6. **Automatic ICA** — FastICA with automatic muscle- and eye-artefact
   component detection, replacing the manual `rejectic` step.
7. **Average reference.**
8. **Retain** the N least-variable epochs (default 60 ≈ 10 min), as in
   `checktrials.m`.
9. **Multitaper spectrum** + normalised peak-in-band power (`calcftspec.m`).
10. **dwPLI connectivity** per canonical band via `wpli2_debiased`
    (`ftcoherence.m`).
11. **Graph-theory metrics** across proportional density thresholds
    (`calcgraph.m`): clustering, characteristic path length, global efficiency,
    modularity, modules, betweenness, participation coefficient, degree,
    modular span.
12. **Figures**: log power spectrum, per-band connectivity matrix, and the
    topographic brain-network plot (`plothead.m` / `plotgraph3d.m`).

## Setup (uv)

```bash
uv sync --extra test          # create the environment and install everything
```

## Usage

Command line — run on a file, or on built-in synthetic data with `--demo`:

```bash
uv run mohawk subject.edf -n subject -o out        # process a real recording
uv run mohawk --demo -o out_demo                   # synthetic demonstration
uv run mohawk --demo --heuristic 5 --trials 6 -o q # faster settings
```

Python API:

```python
from mohawk import run_pipeline, save_results

result = run_pipeline("subject.edf", basename="subject", outdir="out")
save_results(result, "out")          # writes subject_mohawk.npz
print(result.figures)                # dict of generated figure paths
```

On the synthetic generator (no data file needed):

```python
from mohawk.datasets import synthetic_raw
from mohawk import run_pipeline
result = run_pipeline(synthetic_raw(), basename="demo", outdir="out")
```

## MATLAB → Python mapping

| MATLAB source | Python location | Notes |
|---|---|---|
| `mohawk.m` (top-level) | `mohawk/pipeline.py` `run_pipeline` | fully automated |
| `dataimport.m` | `mohawk/io.py`, `mohawk/preprocess.py` | import, montage, peripheral removal, resample, filter, line noise |
| `preplocs.m` / `ichandist` | `mohawk/metrics.py` `channel_distance` | inter-channel distances from montage |
| `epochdata.m` | `mohawk/epoching.py` `make_epochs` | 10 s fixed-length epochs |
| `rejartifacts` (external) | `mohawk/artifacts.py` `reject_and_interpolate` | variance z-score, auto |
| `computeic` / `rejectic` (external) | `mohawk/artifacts.py` `run_ica` | FastICA + auto artefact detection |
| `rereference.m` | `mohawk/preprocess.py` `average_reference` | common average |
| `checktrials.m` | `mohawk/epoching.py` `check_trials` | keep least-variable epochs |
| `calcftspec.m` | `mohawk/spectrum.py` | multitaper PSD + peak band power |
| `plotftspec.m` | `mohawk/plotting.py` `plot_spectrum` | log spectrum + band markers |
| `ftcoherence.m` | `mohawk/connectivity.py` | dwPLI = `wpli2_debiased` |
| `calcgraph.m` | `mohawk/metrics.py` `calc_graph_metrics` | per-band/threshold metrics |
| BCT `threshold_proportional`, `clustering_coef_bu`, `charpath`, `distance_bin`, `efficiency_bin`, `betweenness_bin`, `degrees_und`, `community_louvain`, `participation_coef` | `mohawk/graph.py` | direct ports |
| `plothead.m` / `plotgraph3d.m` / `plotarc3d.m` | `mohawk/plotting.py` `plot_head_network` | topographic network plot |
| `freqlist.mat` | `mohawk/config.py` `FREQ_BANDS` | delta/theta/alpha/beta/gamma |

Parameters (filter cut-offs, taper bandwidth, thresholds, band definitions,
epoch length) are collected in `mohawk/config.py` and mirror the constants
hard-coded across the MATLAB sources.

## Automation: what replaced the manual steps

The original app paused for two interactive tasks. To run end-to-end these are
now automatic:

* **Bad channels / epochs** — flagged by robust (median/MAD) variance
  z-scoring at the original threshold of 4; bad channels are interpolated from
  the montage, high-variance epochs dropped. (`mohawk/artifacts.py`)
* **ICA components** — FastICA components are scored by MNE's muscle-artefact
  heuristic and by correlation with a frontal channel used as an EOG proxy;
  flagged components are removed automatically.

Both are configurable via `ArtifactConfig` in `mohawk/config.py`.

## Tests

```bash
uv run pytest
```

33 tests cover the graph metrics (validated against analytically known graphs
— rings, complete graphs, stars, two-clique modularity), spectrum band-power
normalisation and alpha-peak detection, dwPLI symmetry/range and
phase-coupling recovery, the per-threshold graph metrics, IO/montage/filter
behaviour, and the full pipeline end-to-end (in-memory and via a FIF file).

## Verified output

Running on the synthetic generator — which injects two hemisphere-clustered,
phase-coupled 10 Hz sources on a 1/f background — the pipeline reproduces the
expected behaviour:

* the spectrum shows a sharp 10 Hz alpha peak with correct band markers;
* alpha dwPLI connectivity is uniformly high (zero diagonal);
* the alpha network resolves into **2 modules** matching the injected clusters,
  while the coupling-free delta band fragments into more modules with
  heterogeneous edge strengths — the band-specific network discrimination that
  is the point of MOHAWK.

## Scope

This port covers the **single-subject** pipeline (`mohawk.m`), which is the
complete analysis for one recording. The original repository also contains
group-level and machine-learning classification scripts
(`groupdata.m`, `buildecc.m`, `testind.m`, `plotclass.m`, `runjobs.m`, …) that
operate on pre-built group datasets and classifier ensembles not shipped with
the repository; those are out of scope here.

## License

GPL-3.0-or-later, matching the original MOHAWK (which builds on EEGLAB and
FieldTrip).
