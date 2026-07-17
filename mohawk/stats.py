"""Group-level statistics from Chennu et al. (2017).

ROC / AUC screening of spectral and network metrics, Benjamini-Hochberg FDR,
and the ordered-alternative Jonckheere-Terpstra permutation test used to show
monotonic trends across the consciousness continuum.

Returns lightweight dataclass records (no pandas dependency); convert to a
table yourself if desired.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from .features import SubjectFeatures

NETWORK_METRICS = ("clustering", "path_length", "modularity", "participation", "modular_span")
SCALAR_BAND_METRICS = ("relative_power", "median_connectivity")


@dataclass(slots=True)
class MetricScreen:
    band: str
    metric: str
    density: float          # NaN for band-only metrics
    auc: float
    p_value: float
    p_fdr: float = float("nan")


def absolute_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUC folded to [0.5, 1] (direction-agnostic, as in the paper screening)."""
    auc = float(roc_auc_score(y_true, scores))
    return max(auc, 1.0 - auc)


def fdr_bh(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    out = np.empty_like(adjusted)
    out[order] = np.clip(adjusted, 0, 1)
    return out


def screen_pair(
    features: Sequence[SubjectFeatures],
    labels: Sequence[str],
    group_a: str,
    group_b: str,
) -> list[MetricScreen]:
    """Screen every band/metric family for separating two groups.

    Band metrics (relative power, median connectivity) are scored directly;
    network metrics are optimised over network density (best AUC).  Results are
    returned sorted by descending AUC with BH-FDR-adjusted p-values.
    """
    if not features:
        raise ValueError("No subject features supplied")
    labels = np.asarray(labels)
    sel = np.flatnonzero(np.isin(labels, [group_a, group_b]))
    if sel.size == 0:
        raise ValueError("No subjects match the requested groups")
    y = (labels[sel] == group_b).astype(int)
    template = features[0]
    rows: list[MetricScreen] = []

    for band in template.band_names:
        for metric in SCALAR_BAND_METRICS:
            vals = np.array([features[i].scalar_metric(metric, band) for i in sel])
            p = mannwhitneyu(vals[y == 0], vals[y == 1], alternative="two-sided").pvalue
            rows.append(MetricScreen(band, metric, float("nan"), absolute_auc(y, vals), float(p)))
        for metric in NETWORK_METRICS:
            best = None
            for density in template.densities:
                vals = np.array(
                    [features[i].scalar_metric(metric, band, float(density)) for i in sel]
                )
                auc = absolute_auc(y, vals)
                if best is None or auc > best[0]:
                    best = (auc, float(density), vals)
            auc, density, vals = best
            p = mannwhitneyu(vals[y == 0], vals[y == 1], alternative="two-sided").pvalue
            rows.append(MetricScreen(band, metric, density, auc, float(p)))

    fdr = fdr_bh([r.p_value for r in rows])
    for row, q in zip(rows, fdr):
        row.p_fdr = float(q)
    rows.sort(key=lambda r: r.auc, reverse=True)
    return rows


def jonckheere_terpstra(
    groups: Sequence[np.ndarray],
    *,
    n_permutations: int = 9999,
    random_state: int = 17,
) -> tuple[float, float]:
    """One-sided ordered-alternative Jonckheere-Terpstra permutation test.

    ``groups`` are ordered (e.g. UWS -> MCS- -> MCS+ -> ...); returns the
    observed statistic and a permutation p-value for a monotonic increase.
    """
    arrays = [np.asarray(g, dtype=float) for g in groups]

    def statistic(parts: Sequence[np.ndarray]) -> float:
        total = 0.0
        for li, left in enumerate(parts[:-1]):
            for right in parts[li + 1:]:
                d = right[:, None] - left[None, :]
                total += np.sum(d > 0) + 0.5 * np.sum(d == 0)
        return float(total)

    observed = statistic(arrays)
    values = np.concatenate(arrays)
    sizes = [len(g) for g in arrays]
    cuts = np.cumsum(sizes)[:-1]
    rng = np.random.default_rng(random_state)
    exceed = 0
    for _ in range(n_permutations):
        shuffled = rng.permutation(values)
        if statistic(np.split(shuffled, cuts)) >= observed:
            exceed += 1
    return observed, (exceed + 1) / (n_permutations + 1)
