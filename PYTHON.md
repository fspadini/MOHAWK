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

## Running on EGI MFF recordings

This is the path for recordings like a modern EGI net export
(`Subject_070225.mff`).

**1. An `.mff` is a folder, not a single file.** If yours arrived as a `.zip`,
unzip it so you end up with a directory whose name ends in `.mff` and which
contains `signal1.bin`, `info.xml`, `coordinates.xml`, `sensorLayout.xml`, etc.:

```bash
mkdir Subject_070225.mff
unzip Subject_070225.mff.zip -d Subject_070225.mff
# verify the files landed directly inside (not in a nested subfolder):
ls Subject_070225.mff        # -> signal1.bin  info.xml  coordinates.xml  ...
```

The folder **must** keep the `.mff` extension — that is how the reader
recognises the format.

**2. Run the pipeline** — command line:

```bash
uv run mohawk Subject_070225.mff -n subject -o out_subject
```

or from Python:

```python
from mohawk import run_pipeline, save_results
result = run_pipeline("Subject_070225.mff", basename="subject", outdir="out_subject")
save_results(result, "out_subject")
```

**3. What happens automatically** for these nets:

* the sensor montage is read straight from the net's `coordinates.xml` (no
  manual coordinate work needed, even for modern high-density nets);
* the flat online-reference channel (`REF CZ`) is dropped;
* noisy channels, narrowband-artefact channels, and bad epochs are removed
  automatically, and ICA removes ocular/muscle components;
* figures are written to `out_subject/figures/` (log spectrum, per-band 3D
  "mohawk" topograph, flat network, connectivity matrices) and the numeric
  results to `out_subject/subject_mohawk.npz`.

**Line-noise frequency.** The default notch is **50 Hz** (Europe). For 60 Hz
mains, override it:

```python
from mohawk.config import MohawkConfig
cfg = MohawkConfig()
cfg.preproc.line_freq = 60.0
result = run_pipeline("Subject.mff", basename="subject", outdir="out", config=cfg)
```

**Useful CLI options:** `--montage standard_1005` to force a montage,
`--trials 60` to fix the number of retained epochs (default 60; `0` keeps all),
`--no-graph` to skip the slow graph metrics, `--heuristic N` to set the number
of Louvain repetitions. Run `uv run mohawk --help` for the full list.

**Tuning the automatic rejection.** Every threshold lives in `ArtifactConfig`
(`mohawk/config.py`) and can be overridden on the config passed to
`run_pipeline`, e.g. `cfg.artifact.narrowband_db = 5.0` to be stricter about
narrowband artefacts, or `cfg.artifact.hi_var_zthresh = 2.5` for noisy channels.

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
| `plothead.m` / `plotgraph3d.m` / `plotarc3d.m` | `mohawk/plotting.py` `plot_mohawk_3d`, `plot_head_network` | 3D + flat topographic network plots |
| `freqlist.mat` | `mohawk/config.py` `FREQ_BANDS` | delta/theta/alpha/beta/gamma |
| `groupdata.m` (aggregation) | `mohawk/features.py` `SubjectFeatures` | per-subject feature files |
| `plotauc.m` / `runauc.m` | `mohawk/stats.py` `screen_pair`, `absolute_auc`, `fdr_bh` | AUC / Mann-Whitney / BH-FDR screening |
| (ordered-trend test) | `mohawk/stats.py` `jonckheere_terpstra` | monotonic trend across states |
| `buildecc.m` / `testind.m` / `plotclass.m` | `mohawk/classification.py` `svm_cross_validation` | RBF-SVM 4-fold, Platt, Youden, χ² |

Parameters (filter cut-offs, taper bandwidth, thresholds, band definitions,
epoch length) are collected in `mohawk/config.py` and mirror the constants
hard-coded across the MATLAB sources.

## Automation: what replaced the manual steps

The original app paused for two interactive tasks. To run end-to-end these are
now automatic:

* **Reference / flat channels** — modern EGI nets carry the online reference as
  an all-zero channel (e.g. `REF CZ`); it is dropped up front so it cannot
  corrupt the variance statistics, the average reference, or connectivity.
* **Bad channels / epochs** — flagged by robust (median/MAD) variance
  z-scoring: the original two-tailed threshold of 4, plus a stricter one-sided
  threshold of 3 for *noisy* (high-variance) channels, plus flat-channel
  detection. Bad channels are interpolated from the montage, high-variance
  epochs dropped.
* **Narrowband spectral outliers** — single-channel line-like peaks (e.g. a
  ~25 Hz electronic spike on one electrode) barely raise broadband variance, so
  they slip past the variance test. Each channel's power spectrum is compared to
  its own frequency-smoothed baseline to isolate *narrow* peaks, then to the
  across-channel consensus; a channel is flagged only if a narrow peak exceeds
  `narrowband_db` (default 6 dB) and is a robust-z outlier across channels.
  Genuine rhythms shared across channels (posterior alpha) sit near the
  consensus and are never flagged. Flagged channels are interpolated.
* **ICA components** — FastICA components are scored by MNE's muscle-artefact
  heuristic and by correlation with a frontal channel used as an EOG proxy.
  The proxy is found by flexible name matching (`Fp1`, `FP1`, `129 FP1`, EGI
  `E22`, …) with a most-anterior-electrode fallback, and — crucially — that
  frontal channel is **protected from interpolation**, since blinks make it
  high-variance and interpolating it would erase the very signal ICA needs to
  detect ocular components. Flagged components are removed automatically.

All thresholds are configurable via `ArtifactConfig` in `mohawk/config.py`.

On the real 128-channel recording this tuning drops the flat `REF CZ`,
interpolates the genuinely noisy channels (e.g. the ones at variance z > 5),
and removes three ICA components including a blink component correlating 0.97
with the frontal channel — versus zero components before tuning.

## Tests

```bash
uv run pytest
```

62 tests cover the graph metrics (validated against analytically known graphs
— rings, complete graphs, stars, two-clique modularity), spectrum band-power
and relative-power normalisation and alpha-peak detection, dwPLI symmetry/range
and phase-coupling recovery, the per-threshold graph metrics, IO/montage/filter
behaviour, the automatic and narrowband artefact rejection, faithful (manual)
mode, YAML config loading, the group-level statistics (AUC/FDR/Jonckheere-
Terpstra) and RBF-SVM classification, and the full pipeline end-to-end
(in-memory and via a FIF file).

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

## Cross-check against the paper and real EEG

The method parameters were checked directly against Chennu et al. (2017),
*Brain* 140:2120-2132:

* power estimated 0.5-45 Hz with a **multitaper method using five Slepian
  tapers** — `tapsmofrq = 0.3` over 10 s epochs yields exactly five tapers,
  which the port matches (`bandwidth = 0.6` Hz full-width);
* connectivity is the **debiased weighted phase lag index (dwPLI)**, mapped to
  MNE's `wpli2_debiased`;
* canonical bands delta (0-4), theta (4-8), alpha (8-13) Hz;
* dwPLI matrices proportionally thresholded across connection densities;
* graph metrics: clustering, characteristic path length, Louvain modularity,
  participation coefficient, and modular span — all implemented here.

The pipeline was also run end-to-end on a real 128-channel EGI `.mff` recording
(250 Hz). MNE reads the net's `coordinates.xml` directly into head coordinates,
so no manual coordinate manipulation was needed; the 3D "mohawk" renderer fits a
sphere to those positions. The output reproduces the paper's signature figure —
a scalp map coloured by network hub strength with connectivity arcs rising into
a crest above the head — and shows band-specific modular structure (e.g. fewer,
larger modules in delta than alpha).

The **3D mohawk** plot (`mohawk.plotting.plot_mohawk_3d`) is a Matplotlib
reconstruction of the EEGLAB `headplot`-based rendering; it uses a spherical
scalp model rather than a subject MRI mesh, but preserves the scalp power map,
the strength/module arc colouring, and the raised-arc "mohawk" geometry.

> Note: real patient EEG data is personal/health information and is **not**
> committed to this repository (`data/`, `out_real/`, and `*.mff` are
> git-ignored). Only synthetic example figures are included.

## Faithful (manual) mode

Automatic artefact rejection is a **practical substitute** for the paper's
manual review of channels, epochs and ICA components — it is *not* what the
original study did. For a faithful replication, turn automation off and supply
your own reviewed decisions:

```python
from mohawk import run_pipeline
from mohawk.artifacts import suggest_variance_outliers   # non-mutating review list

result = run_pipeline(
    "subject.mff", basename="subject", outdir="out",
    auto_reject=False,                       # no automatic rejection
    bad_channels=["E31", "E67"],             # your reviewed channels
    bad_segments=[(12.0, 4.0)],              # (onset, duration) seconds to drop
    ica_exclude=[0, 3],                      # your reviewed ICA components
)
```

`suggest_variance_outliers(epochs)` returns a *suggested* bad-channel/epoch list
without modifying anything, mirroring the paper's visual-confirmation workflow.
On the CLI: `mohawk subject.mff --faithful --bad-channels E31,E67 --ica-exclude 0,3`.

## Paper-exact configuration

All parameters live in `mohawk/config.py` dataclasses and can also be loaded
from YAML. The shipped `config/paper.yml` encodes the paper's choices (three
bands, 33 densities from 0.9 to 0.1, Infomax ICA, relative power over 0.5-13 Hz):

```python
from mohawk import load_config, run_pipeline
cfg = load_config("config/paper.yml")
result = run_pipeline("subject.mff", basename="subject", outdir="out", config=cfg)
```

or `mohawk subject.mff --config config/paper.yml`. Two deliberate defaults
differ from `paper.yml` because they follow the *released MOHAWK code* rather
than the paper text: relative power is normalised over all five bands (not
three), and densities start at 1.0 (not 0.9). Both are configurable.

## Group-level analysis

The paper's group ROC screening, ordered-trend test, and RBF-SVM classification
are implemented and operate on per-subject feature files:

```python
from mohawk import run_pipeline, SubjectFeatures
from mohawk.stats import screen_pair, jonckheere_terpstra
from mohawk.classification import FeatureSpec, build_feature_matrix, svm_cross_validation

# 1. per subject: run the pipeline and save features
for path, name in recordings:
    res = run_pipeline(path, basename=name, outdir="out", compute_graph=True)
    SubjectFeatures.from_result(res).save(f"out/{name}_features.npz")

# 2. group: screen every band/metric for a diagnostic contrast
subjects = [SubjectFeatures.load(p) for p in feature_paths]
ranking = screen_pair(subjects, crs_labels, "UWS", "MCS")   # AUC + Mann-Whitney + BH-FDR
best = ranking[0]                                            # highest-AUC metric/band/density

# 3. monotonic trend across the consciousness continuum
obs, p = jonckheere_terpstra([values_uws, values_mcs_minus, values_mcs_plus])

# 4. classify from the winning feature
x = build_feature_matrix(subjects, [FeatureSpec(best.metric, best.band, best.density)])
result = svm_cross_validation(x, crs_labels)
print(result.accuracy, result.confusion, result.p_value)
```

`svm_cross_validation` reproduces the paper's stratified four-fold RBF-SVM with
Platt probabilities and a Youden threshold. As the paper's own design selects
hyperparameters on the same folds used to report accuracy, the estimate is not
nested and can be optimistic — use nested CV for genuinely new claims.

## Scope

The **single-subject** pipeline (`mohawk.m`) is fully covered, and the
**group-level** ROC screening, trend test, and RBF-SVM classification from the
paper are implemented (`mohawk/stats.py`, `mohawk/classification.py`). What is
*not* included: the PET/clinical-label ingestion and the exact MATLAB ECOC
multiclass loss (a one-vs-one SVC is used as the close analogue), since those
depend on data and labels not shipped with the repository.

## License

GPL-3.0-or-later, matching the original MOHAWK (which builds on EEGLAB and
FieldTrip).
