"""Figure generation (ports of ``plotftspec.m`` and ``plothead`` / ``plotgraph3d``).

Produces the three signature MOHAWK outputs:

* the log power spectrum with canonical band markers,
* the per-band connectivity matrix, and
* the topographic brain-network "mohawk" plot (nodes sized by weighted degree,
  coloured by module; strongest edges coloured by connection strength).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless, file-only rendering
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from . import graph as G
from .config import BAND_NAMES, PlotConfig


def _fit_sphere(p: np.ndarray) -> np.ndarray:
    """Least-squares sphere centre for a set of 3D points."""
    A = np.column_stack([2 * p, np.ones(len(p))])
    b = (p ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    return sol[:3]


def azimuthal_positions(pos3d: np.ndarray) -> np.ndarray:
    """Project 3D electrode positions to 2D via azimuthal-equidistant mapping.

    Standard EEG topomap projection: the vertex (Cz) maps to the origin and
    elevation increases the radius, giving the familiar circular head layout.
    The sphere centre is fit by least squares so off-centre montages project
    correctly.
    """
    p = np.asarray(pos3d, dtype=float)
    p = p - _fit_sphere(p)
    r = np.linalg.norm(p, axis=1)
    r[r == 0] = 1e-12
    elev = np.arccos(np.clip(p[:, 2] / r, -1, 1))  # 0 at +z (vertex)
    az = np.arctan2(p[:, 1], p[:, 0])
    rad = elev / (np.pi / 2)
    return np.column_stack([rad * np.cos(az), rad * np.sin(az)])


def _draw_head(ax, radius: float = 1.0):
    """Draw a head outline (circle, nose, ears) like an EEG topomap."""
    ax.add_patch(Circle((0, 0), radius, fill=False, color="0.4", lw=2, zorder=1))
    # nose
    ax.plot(
        [-0.12 * radius, 0, 0.12 * radius],
        [radius * 0.99, radius * 1.15, radius * 0.99],
        color="0.4", lw=2, zorder=1,
    )
    # ears
    for sign in (-1, 1):
        ax.plot(
            [sign * radius, sign * radius * 1.1, sign * radius],
            [0.1 * radius, 0, -0.1 * radius],
            color="0.4", lw=2, zorder=1,
        )


def _bezier_arc(p0, p1, height, n=40):
    """Quadratic Bezier arc bulging outward, mimicking the 3D connectivity arcs."""
    mid = (p0 + p1) / 2
    norm = np.linalg.norm(mid) + 1e-9
    ctrl = mid + (mid / norm) * height
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p1


def plot_spectrum(result, outpath: str | None = None, basename: str = ""):
    """Log power spectrum with band demarcations (plotftspec.m)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result.freqs, 10 * np.log10(result.spectra.T + 1e-30), lw=1.2)
    ax.set_xlim(0.01, 40)
    ax.set_xlabel("Frequency (Hz)", fontsize=14)
    ax.set_ylabel("Power (dB)", fontsize=14)
    ax.set_title(basename)
    ylim = ax.get_ylim()
    for f in range(min(4, result.bands.shape[0])):
        for edge in result.bands[f]:
            ax.axvline(edge, color="k", ls="-.", lw=0.8)
        ax.text(result.bands[f, 0], ylim[1], BAND_NAMES[f], fontsize=12, va="top")
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
    return fig


def plot_connectivity_matrix(
    conn, band_idx: int, outpath: str | None = None, basename: str = ""
):
    """Heatmap of a band's dwPLI connectivity matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    mat = conn.matrix[band_idx]
    im = ax.imshow(mat, cmap="viridis", vmin=0)
    ax.set_title(f"{basename} {BAND_NAMES[band_idx]} dwPLI")
    ax.set_xlabel("channel")
    ax.set_ylabel("channel")
    fig.colorbar(im, ax=ax, label="dwPLI")
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=150)
        plt.close(fig)
    return fig


def _scalp_field(P, center, elec_pos, values, sigma=0.25):
    """Interpolate electrode ``values`` onto surface points ``P`` (RBF on sphere)."""
    dirs = elec_pos - center
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
    Pd = P - center
    Pn = Pd / (np.linalg.norm(Pd, axis=-1, keepdims=True) + 1e-12)
    field = np.zeros(P.shape[:-1])
    wsum = np.zeros(P.shape[:-1])
    for k in range(len(elec_pos)):
        cosang = np.tensordot(Pn, dirs[k], axes=([-1], [0]))
        w = np.exp(-(1 - cosang) / sigma)
        field += w * values[k]
        wsum += w
    return field / (wsum + 1e-12)


def _arc3d(p0, p1, center, height, up_bias=1.8, n=32):
    """3D quadratic-Bezier arc rising above the scalp into an upward crest.

    The control point is lifted from the edge midpoint along a direction that
    blends the outward scalp normal with a strong ``+z`` (up) bias, so the arcs
    sprout upward like the "mohawk" of Chennu et al. (2017) rather than
    radiating in all directions.
    """
    mid = (p0 + p1) / 2
    out = mid - center
    out = out / (np.linalg.norm(out) + 1e-12)
    lift = out + up_bias * np.array([0.0, 0.0, 1.0])
    lift = lift / (np.linalg.norm(lift) + 1e-12)
    ctrl = mid + lift * height
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p1


def plot_mohawk_3d(
    conn,
    band_idx: int,
    pos3d: np.ndarray,
    cfg: PlotConfig | None = None,
    arcs: str = "strength",
    outpath: str | None = None,
    basename: str = "",
    seed: int | None = None,
    elev: float = 8.0,
    azim: float = 45.0,
):
    """3D "mohawk" connectivity topograph (faithful to plothead.m / plotgraph3d.m).

    A scalp surface coloured by weighted node degree, with the strongest
    ``cfg.plot_quantile`` proportion of within-module edges drawn as arcs that
    rise above the head (height proportional to connection strength), coloured
    either by strength (jet) or by module.  This reproduces the signature
    figure of Chennu et al. (2017).
    """
    cfg = cfg or PlotConfig()
    matrix = np.nan_to_num(conn.matrix[band_idx].copy())
    n = matrix.shape[0]

    matrix = G.threshold_proportional(matrix, cfg.plot_quantile)
    vsize = matrix.sum(axis=1) / (n - 1)
    minfo, _ = G.community_louvain(matrix, seed=seed)

    e0, e1 = cfg.erange
    escaled = np.clip((matrix - e0) / (e1 - e0), 0, 1)

    center = _fit_sphere(pos3d)
    radius = np.linalg.norm(pos3d - center, axis=1).mean()

    fig = plt.figure(figsize=(7, 8), facecolor="black")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")
    ax.set_position([0, 0, 1, 1])  # fill the figure, no wasted margin

    # --- scalp surface coloured by weighted degree (headplot equivalent) ---
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 60)
    sx = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    sy = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    sz = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    P = np.stack([sx, sy, sz], axis=-1)
    field = _scalp_field(P, center, pos3d, vsize)
    fmin, fmax = field.min(), field.max()
    norm = (field - fmin) / (fmax - fmin + 1e-12)
    scalp_colors = plt.get_cmap('turbo')(0.15 + 0.7 * norm)
    ax.plot_surface(
        sx, sy, sz, facecolors=scalp_colors, rcount=60, ccount=60,
        linewidth=0, antialiased=True, shade=False, zorder=1,
    )

    # --- arcs (mohawk), strongest within-module edges ---
    mod_cmap = plt.get_cmap("tab10")
    modules = np.unique(minfo)
    mod_color = {m: mod_cmap(i % 10) for i, m in enumerate(modules)}
    strength_cmap = plt.get_cmap("jet")

    triu = np.array(np.triu_indices(n, k=1)).T
    weights = np.array([escaled[i, j] for i, j in triu])
    for k in np.argsort(weights):
        i, j = triu[k]
        w = escaled[i, j]
        if w <= 0:
            continue
        if minfo[i] != minfo[j]:
            continue  # intra-module edges only (plotinter='off')
        color = strength_cmap(w) if arcs == "strength" else mod_color[minfo[i]]
        height = (0.25 + 1.0 * w) * radius  # lhfactor-like lift
        arc = _arc3d(pos3d[i], pos3d[j], center, height)
        ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color=color,
                lw=0.6 + 1.4 * w, alpha=0.8, zorder=3)

    # Frame the head plus the upward crest: shift the box up and keep equal
    # aspect so the sphere is undistorted.
    r = radius * 1.5
    ax.set_xlim(center[0] - r, center[0] + r)
    ax.set_ylim(center[1] - r, center[1] + r)
    ax.set_zlim(center[2] - r * 0.7, center[2] + r * 1.3)
    try:
        ax.set_box_aspect((1, 1, 1), zoom=1.45)
    except TypeError:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    ax.set_title(f"{basename}: {BAND_NAMES[band_idx]} band  "
                 f"({len(modules)} modules)", color="white", y=0.92)
    if outpath:
        fig.savefig(outpath, dpi=200, facecolor="black")
        plt.close(fig)
    return fig, minfo


def _set_equal_3d(ax, pos, radius):
    """Set equal aspect ratio for a 3D axis around the head."""
    c = pos.mean(axis=0)
    r = radius * 2.2
    ax.set_xlim(c[0] - r, c[0] + r)
    ax.set_ylim(c[1] - r, c[1] + r)
    ax.set_zlim(c[2] - r, c[2] + r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def plot_head_network(
    conn,
    band_idx: int,
    pos3d: np.ndarray,
    cfg: PlotConfig | None = None,
    arcs: str = "strength",
    outpath: str | None = None,
    basename: str = "",
    seed: int | None = None,
):
    """Topographic brain-network plot (plothead.m / plotgraph3d.m).

    Keeps the strongest ``cfg.plot_quantile`` proportion of edges, detects
    modules by Louvain community detection, sizes nodes by weighted degree, and
    draws arcs coloured either by connection strength (``arcs='strength'``) or
    module affiliation (``arcs='module'``).
    """
    cfg = cfg or PlotConfig()
    matrix = np.nan_to_num(conn.matrix[band_idx].copy())
    n = matrix.shape[0]

    matrix = G.threshold_proportional(matrix, cfg.plot_quantile)
    vsize = matrix.sum(axis=1) / (n - 1)
    minfo, _ = G.community_louvain(matrix, seed=seed)

    # rescale edges/vertices (plotgraph3d.m)
    e0, e1 = cfg.erange
    escaled = np.clip((matrix - e0) / (e1 - e0), 0, None)
    v0, v1 = cfg.vrange
    vscaled = np.clip((vsize - v0) / (v1 - v0), 0, None)

    pos = azimuthal_positions(pos3d)
    # normalise positions to the unit head
    span = np.abs(pos).max() + 1e-9
    pos = pos / span

    fig, ax = plt.subplots(figsize=(6, 7), facecolor="black")
    ax.set_facecolor("black")
    _draw_head(ax)

    modules = np.unique(minfo)
    mod_cmap = plt.get_cmap("tab10")
    mod_color = {m: mod_cmap(i % 10) for i, m in enumerate(modules)}
    strength_cmap = plt.get_cmap("jet")

    # edges: draw strongest first so weak ones don't cover them
    triu = np.array(np.triu_indices(n, k=1)).T
    weights = np.array([escaled[i, j] for i, j in triu])
    order = np.argsort(weights)
    for k in order:
        i, j = triu[k]
        w = escaled[i, j]
        if w <= 0:
            continue
        if arcs == "module" and minfo[i] != minfo[j]:
            continue
        color = (
            strength_cmap(np.clip(w, 0, 1))
            if arcs == "strength"
            else mod_color[minfo[i]]
        )
        arc = _bezier_arc(pos[i], pos[j], height=0.18 * w)
        ax.plot(arc[:, 0], arc[:, 1], color=color, lw=0.8 + 1.6 * w,
                alpha=0.85, zorder=2)

    # nodes
    node_sizes = 30 + 300 * vscaled
    node_colors = [mod_color[m] for m in minfo]
    ax.scatter(pos[:, 0], pos[:, 1], s=node_sizes, c=node_colors,
               edgecolors="white", linewidths=0.5, zorder=3)

    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"{basename}: {BAND_NAMES[band_idx]} band  "
        f"({len(modules)} modules)",
        color="white",
    )
    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, dpi=200, facecolor="black")
        plt.close(fig)
    return fig, minfo
