"""
Shared bump-hunt cut / observable / ROC logic.

This module is the single source of truth for the analysis-level selection used
in two places:

  1. ``src.analysis``  — a standalone tool that runs a trained ONNX (or ``.pt``)
     network on signal + background HDF5 samples and produces cutflow tables,
     ROC scans, and the average-mass bump-hunt histogram on the *final* network.
  2. ``src.train``     — the training loop, which animates the same ROC / cutflow
     per epoch using the per-event validation arrays it already collects.

It is intentionally pure ``numpy`` (no ``torch``) so it can be called from either
the array-based training path or the inference-based standalone path without
pulling in the model.

------------------------------------------------------------------------------
Analysis observables (all evaluated on the network's CHOSEN interpretation)
------------------------------------------------------------------------------
For each event the two predicted parent triplets give:

  - ``avg_mass``   : (m1 + m2) / 2          -- the bump-hunt variable
  - ``mass_asym``  : |m1 - m2| / (m1 + m2)  -- signal is balanced (small)
  - ``delta_phi``  : Δφ between the two triplet sums, folded to [0, π]
                     -- pair-produced parents are back-to-back (large)
  - ``avg_boost``  : mean Lorentz boost γ = E/m over the two triplets
                     -- high-mass signal triplets are produced closer to rest
  - Dalitz edge / corner proximity (see below).

------------------------------------------------------------------------------
The Dalitz cut, and why it is two scannable shapes
------------------------------------------------------------------------------
For a 3-body triplet (jets a, b, c, total invariant mass M) the three normalised
pairwise invariant-mass-squared

    d1 = m²(a,b)/M² ,  d2 = m²(a,c)/M² ,  d3 = m²(b,c)/M²

satisfy d1 + d2 + d3 ≈ 1 (massless jets), so each event-triplet lives on the
Dalitz simplex (a triangle).  The triangle's three *edges* are the lines where
one d_i → 0 (one jet pair is soft/collinear); its three *corners* are where two
d_i → 0 simultaneously (two pairs soft/collinear).  QCD multijet splittings are
soft/collinear-enhanced and therefore pile up near the edges and especially the
corners; a genuine 3-body decay fills the interior.

A gluino→q+squark(→qq) cascade puts the on-shell squark mass at a *fixed*
nonzero d_i — a populated band/line in the interior (or near an edge if the
squark is light).  Cutting "the edges" can therefore remove a light-squark
resonant signal.  To let you choose empirically, two scannable cut shapes are
provided (see :data:`DEFAULT_CUTS`):

  - ``dalitz_edge``  : keep events whose *closest-edge distance*
      ``min(d1,d2,d3)`` exceeds a threshold.  Removes a band along ALL three
      edges -- aggressive QCD rejection, but bites into resonant bands that sit
      near an edge.
  - ``dalitz_corner``: keep events whose *second-smallest* coordinate
      ``median(d1,d2,d3)`` exceeds a threshold.  Removes only the three corners
      (where two pairs are soft/collinear) -- the most blatantly QCD-like region
      -- while preserving resonant bands anywhere else in the interior.

Both metrics are aggregated per event by taking the **minimum over the two
triplets** (an event is edge/corner-like if *either* predicted parent is), so a
single edge-like triplet is enough to tag the event as QCD-like.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# numpy >= 2.0 renamed ``np.trapz`` to ``np.trapezoid``; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

# --------------------------------------------------------------------------- #
# Cut configuration
# --------------------------------------------------------------------------- #
# Each cut is (observable_key, direction, threshold).  ``direction`` is:
#   "lower" -> keep events with value <= threshold  (cut removes the high tail)
#   "upper" -> keep events with value >= threshold  (cut removes the low tail)
# Thresholds below are the user's requested nominal working point.
DEFAULT_CUTS: dict[str, tuple[str, float]] = {
    "delta_phi":     ("upper", 2.5),   # Δφ(parent1, parent2) > 2.5
    "mass_asym":     ("lower", 0.4),   # |m1-m2|/(m1+m2) < 0.4
    "avg_boost":     ("lower", 2.0),   # mean γ = E/m of the two triplets < 2
    "dalitz_corner": ("upper", 0.0),   # median(d) > t  (corner removal; off by default)
    "dalitz_edge":   ("upper", 0.0),   # min(d)    > t  (edge-band removal; off by default)
}

# Human-readable axis labels for plots.
OBS_LABELS = {
    "avg_mass":      r"Average candidate mass $(m_1{+}m_2)/2$  [GeV]",
    "mass_asym":     r"Mass asymmetry $|m_1{-}m_2|/(m_1{+}m_2)$",
    "delta_phi":     r"$\Delta\phi$ between parent candidates  [rad]",
    "avg_boost":     r"Average triplet boost  $\gamma = E/m$",
    "dalitz_corner": r"Dalitz corner distance  median$(d_1,d_2,d_3)$",
    "dalitz_edge":   r"Dalitz edge distance  min$(d_1,d_2,d_3)$",
    "y_star":        r"$y^* = |y_1 - y_2|/2$ between parent candidates",
    "chi":           r"$\chi = e^{2y^*}$",
    "score":         r"NN signal score",
}


# --------------------------------------------------------------------------- #
# Dalitz edge / corner metrics
# --------------------------------------------------------------------------- #
def dalitz_edge_corner(dalitz_x: np.ndarray, dalitz_y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-event Dalitz edge- and corner-proximity metrics.

    Args:
        dalitz_x, dalitz_y: ``(N, 2)`` arrays of the two normalised pairwise
            mass-squared coordinates for each of the event's two predicted
            triplets, as produced by the model / training loop
            (``x = m²(lead,sub)/M²``, ``y = m²(lead,third)/M²``).

    Returns:
        ``(min_edge, min_corner)``, each ``(N,)``:
          - ``min_edge``  : minimum over the two triplets of ``min(d1,d2,d3)``
            (small ⇒ near a Dalitz edge / soft-collinear).
          - ``min_corner``: minimum over the two triplets of the *median* of
            ``(d1,d2,d3)`` (small ⇒ near a Dalitz corner).
    """
    x = np.asarray(dalitz_x, dtype=float)
    y = np.asarray(dalitz_y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
        y = y[:, None]
    d1 = np.clip(x, 0.0, None)
    d2 = np.clip(y, 0.0, None)
    d3 = np.clip(1.0 - x - y, 0.0, None)          # third coordinate (sum ≈ 1)
    coords = np.stack([d1, d2, d3], axis=-1)       # (N, n_triplet, 3)
    per_triplet_min = coords.min(axis=-1)          # (N, n_triplet) closest edge
    per_triplet_med = np.median(coords, axis=-1)   # (N, n_triplet) corner proxy
    return per_triplet_min.min(axis=-1), per_triplet_med.min(axis=-1)


# --------------------------------------------------------------------------- #
# Observable container
# --------------------------------------------------------------------------- #
@dataclass
class Observables:
    """Per-event analysis observables for one sample (signal OR background).

    ``y_star``/``chi`` are the rapidity-separation variables of the two chosen
    triplets (``y* = |y1-y2|/2``, ``chi = exp(2 y*)``): QCD's t-channel angular
    shape in chi is approximately mass-invariant (scale invariance), which is
    the factorisation the chi-sideband background transfer relies on.  ``score``
    is the event-level NN signal-vs-QCD probability (sigmoid of the score head),
    when the model provides one.  All three default to ``None`` so the original
    six-observable construction keeps working unchanged.
    """

    avg_mass: np.ndarray
    mass_asym: np.ndarray
    delta_phi: np.ndarray
    avg_boost: np.ndarray
    dalitz_edge: np.ndarray
    dalitz_corner: np.ndarray
    weight: np.ndarray | None = None
    extra: dict = field(default_factory=dict)
    y_star: np.ndarray | None = None
    chi: np.ndarray | None = None
    score: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.avg_mass)

    def get(self, key: str) -> np.ndarray:
        return getattr(self, key)

    @property
    def weights(self) -> np.ndarray:
        if self.weight is None:
            return np.ones(len(self), dtype=float)
        return self.weight


def observables_from_arrays(
    mass_sum: np.ndarray,
    mass_asym: np.ndarray,
    delta_phi: np.ndarray,
    avg_boost: np.ndarray,
    dalitz_x: np.ndarray,
    dalitz_y: np.ndarray,
    weight: np.ndarray | None = None,
) -> Observables:
    """Build :class:`Observables` from the per-event arrays the training loop
    already collects (``mass_sum = m1 + m2``)."""
    edge, corner = dalitz_edge_corner(dalitz_x, dalitz_y)
    return Observables(
        avg_mass=np.asarray(mass_sum, dtype=float) / 2.0,
        mass_asym=np.asarray(mass_asym, dtype=float),
        delta_phi=np.asarray(delta_phi, dtype=float),
        avg_boost=np.asarray(avg_boost, dtype=float),
        dalitz_edge=edge,
        dalitz_corner=corner,
        weight=None if weight is None else np.asarray(weight, dtype=float),
    )


# --------------------------------------------------------------------------- #
# Cut application
# --------------------------------------------------------------------------- #
def _keep_mask(values: np.ndarray, direction: str, threshold: float) -> np.ndarray:
    """Boolean keep-mask for a single cut."""
    if direction == "lower":
        return values <= threshold
    if direction == "upper":
        return values >= threshold
    raise ValueError(f"Unknown cut direction {direction!r} (expected 'lower'/'upper').")


def cut_masks(obs: Observables, cuts: dict[str, tuple[str, float]]) -> dict[str, np.ndarray]:
    """Per-cut boolean keep-masks (independent, not cumulative)."""
    masks = {}
    for key, (direction, thr) in cuts.items():
        masks[key] = _keep_mask(obs.get(key), direction, thr)
    return masks


def combined_mask(obs: Observables, cuts: dict[str, tuple[str, float]]) -> np.ndarray:
    """Logical-AND of all cuts → final selection mask."""
    masks = cut_masks(obs, cuts)
    out = np.ones(len(obs), dtype=bool)
    for m in masks.values():
        out &= m
    return out


def cutflow(
    sig: Observables,
    bkg: Observables,
    cuts: dict[str, tuple[str, float]],
    order: list[str] | None = None,
) -> list[dict]:
    """Sequential cutflow.

    Returns a list of rows (one per stage, starting with "no cut"), each a dict
    with absolute and relative signal/background yields and the running S/√B.
    Yields are weighted when weights are present.
    """
    order = order or list(cuts.keys())
    rows = []

    def _yield(obs, mask):
        w = obs.weights
        return float(w[mask].sum())

    s_mask = np.ones(len(sig), dtype=bool)
    b_mask = np.ones(len(bkg), dtype=bool)
    s0, b0 = _yield(sig, s_mask), _yield(bkg, b_mask)

    def _row(name, sm, bm):
        s, b = _yield(sig, sm), _yield(bkg, bm)
        return {
            "cut": name,
            "sig": s, "bkg": b,
            "sig_frac": s / s0 if s0 else 0.0,
            "bkg_frac": b / b0 if b0 else 0.0,
            "s_over_sqrtb": s / np.sqrt(b) if b > 0 else float("inf"),
        }

    rows.append(_row("no cut", s_mask, b_mask))
    for key in order:
        direction, thr = cuts[key]
        s_mask = s_mask & _keep_mask(sig.get(key), direction, thr)
        b_mask = b_mask & _keep_mask(bkg.get(key), direction, thr)
        rows.append(_row(f"{key} {'>' if direction == 'upper' else '<'} {thr:g}", s_mask, b_mask))
    return rows


def format_cutflow(rows: list[dict]) -> str:
    """Pretty-print a cutflow table (returned as a string)."""
    hdr = f"{'cut':<28}{'signal':>12}{'sig frac':>10}{'bkg':>12}{'bkg frac':>10}{'S/sqrt(B)':>12}"
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append(
            f"{r['cut']:<28}{r['sig']:>12.1f}{r['sig_frac']:>10.3f}"
            f"{r['bkg']:>12.1f}{r['bkg_frac']:>10.3f}{r['s_over_sqrtb']:>12.3f}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# ROC scans
# --------------------------------------------------------------------------- #
def roc_curve_1d(
    sig_vals: np.ndarray,
    bkg_vals: np.ndarray,
    direction: str,
    sig_w: np.ndarray | None = None,
    bkg_w: np.ndarray | None = None,
    n: int = 200,
) -> dict:
    """Single-variable ROC: sweep the threshold and record (sig eff, bkg rej).

    Returns dict with ``thresholds``, ``sig_eff``, ``bkg_eff``, ``bkg_rej``,
    and the trapezoidal ``auc`` of bkg-rejection vs signal-efficiency.
    """
    sig_vals = np.asarray(sig_vals, dtype=float)
    bkg_vals = np.asarray(bkg_vals, dtype=float)
    sig_w = np.ones_like(sig_vals) if sig_w is None else np.asarray(sig_w, dtype=float)
    bkg_w = np.ones_like(bkg_vals) if bkg_w is None else np.asarray(bkg_w, dtype=float)
    lo = float(min(sig_vals.min(), bkg_vals.min()))
    hi = float(max(sig_vals.max(), bkg_vals.max()))
    thresholds = np.linspace(lo, hi, n)
    s_tot, b_tot = sig_w.sum(), bkg_w.sum()
    sig_eff, bkg_eff = [], []
    for t in thresholds:
        sm = _keep_mask(sig_vals, direction, t)
        bm = _keep_mask(bkg_vals, direction, t)
        sig_eff.append(sig_w[sm].sum() / s_tot if s_tot else 0.0)
        bkg_eff.append(bkg_w[bm].sum() / b_tot if b_tot else 0.0)
    sig_eff = np.array(sig_eff)
    bkg_eff = np.array(bkg_eff)
    bkg_rej = 1.0 - bkg_eff
    order = np.argsort(sig_eff)
    auc = float(_trapz(bkg_rej[order], sig_eff[order]))
    return {
        "thresholds": thresholds, "sig_eff": sig_eff,
        "bkg_eff": bkg_eff, "bkg_rej": bkg_rej, "auc": auc,
    }


def roc_curves(
    sig: Observables,
    bkg: Observables,
    variables: list[str] | None = None,
    cuts: dict[str, tuple[str, float]] | None = None,
    n: int = 200,
) -> dict[str, dict]:
    """One ROC curve per cut variable (each scanned independently)."""
    cuts = cuts or DEFAULT_CUTS
    variables = variables or list(cuts.keys())
    out = {}
    for v in variables:
        direction = cuts[v][0]
        out[v] = roc_curve_1d(
            sig.get(v), bkg.get(v), direction,
            sig_w=sig.weights, bkg_w=bkg.weights, n=n,
        )
    return out


def working_point_cloud(
    sig: Observables,
    bkg: Observables,
    cuts: dict[str, tuple[str, float]] | None = None,
    grids: dict[str, np.ndarray] | None = None,
    n_per_axis: int = 6,
    max_points: int = 4000,
    rng_seed: int = 0,
) -> dict:
    """Scatter cloud of MANY combined cut working points in (sig eff, bkg rej).

    Builds a grid of thresholds for each cut variable (auto-derived from the
    sample percentiles when ``grids`` is not given), forms the Cartesian product
    of working points (randomly subsampled to ``max_points`` if huge), and
    evaluates the combined selection for each.  The Pareto-optimal front is also
    returned so the best achievable trade-offs stand out.
    """
    cuts = cuts or DEFAULT_CUTS
    keys = list(cuts.keys())

    if grids is None:
        grids = {}
        qs = np.linspace(5, 95, n_per_axis)
        for k in keys:
            vals = np.concatenate([sig.get(k), bkg.get(k)])
            grids[k] = np.unique(np.percentile(vals, qs))

    mesh = np.meshgrid(*[grids[k] for k in keys], indexing="ij")
    combos = np.stack([m.reshape(-1) for m in mesh], axis=1)  # (P, n_cuts)

    rng = np.random.default_rng(rng_seed)
    if len(combos) > max_points:
        combos = combos[rng.choice(len(combos), max_points, replace=False)]

    s_w, b_w = sig.weights, bkg.weights
    s_tot, b_tot = s_w.sum(), b_w.sum()
    sig_eff = np.empty(len(combos))
    bkg_rej = np.empty(len(combos))
    for i, combo in enumerate(combos):
        sm = np.ones(len(sig), dtype=bool)
        bm = np.ones(len(bkg), dtype=bool)
        for k, thr in zip(keys, combo):
            direction = cuts[k][0]
            sm &= _keep_mask(sig.get(k), direction, thr)
            bm &= _keep_mask(bkg.get(k), direction, thr)
        sig_eff[i] = s_w[sm].sum() / s_tot if s_tot else 0.0
        bkg_rej[i] = 1.0 - (b_w[bm].sum() / b_tot if b_tot else 0.0)

    pareto = _pareto_front(sig_eff, bkg_rej)
    return {"keys": keys, "combos": combos, "sig_eff": sig_eff,
            "bkg_rej": bkg_rej, "pareto": pareto}


def _pareto_front(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Indices of the Pareto-optimal points (maximise both x and y)."""
    order = np.argsort(-x)
    best_y = -np.inf
    keep = []
    for idx in order:
        if y[idx] >= best_y:
            keep.append(idx)
            best_y = y[idx]
    return np.array(sorted(keep))


# --------------------------------------------------------------------------- #
# Analysis regions (chi-sideband transfer)
# --------------------------------------------------------------------------- #
# The bump hunt uses three regions in y* = |y1-y2|/2 between the two chosen
# triplets.  Signal (pair production) is central (low y*); QCD's t-channel
# exchange is forward-peaked, so high y* is QCD-rich and signal-poor:
#
#   SR (low y*)    : search region — bump hunt in m_avg, blinded in the window
#   VR (mid y*)    : validation — transfer-factor derivation & closure
#   CR (high y*)   : QCD m_avg template source
#
# With the |eta| < 2.4 jet acceptance the y* reach is modest, so boundaries are
# defined by *percentiles of the background-like data* rather than fixed chi
# values — this directly controls the region yields (the binding constraint is
# the CR population at high mass, not the SR).
DEFAULT_SR_QUANTILE = 0.60   # SR = bottom 60% of the y* distribution
DEFAULT_VR_QUANTILE = 0.85   # VR = 60-85%; CR = top 15%


def region_boundaries(
    y_star: np.ndarray,
    weights: np.ndarray | None = None,
    sr_quantile: float = DEFAULT_SR_QUANTILE,
    vr_quantile: float = DEFAULT_VR_QUANTILE,
) -> tuple[float, float]:
    """y* boundaries (b1, b2) such that SR: y* < b1, VR: b1 <= y* < b2, CR: y* >= b2.

    Derived from the (weighted) percentiles of the supplied y* sample — use the
    background-like data (or the QCD sample) here, NOT signal, so the region
    populations are controlled.
    """
    y = np.asarray(y_star, dtype=float)
    if weights is None:
        b1 = float(np.percentile(y, 100.0 * sr_quantile))
        b2 = float(np.percentile(y, 100.0 * vr_quantile))
    else:
        w = np.asarray(weights, dtype=float)
        order = np.argsort(y)
        cw = np.cumsum(w[order])
        cw = cw / cw[-1]
        b1 = float(np.interp(sr_quantile, cw, y[order]))
        b2 = float(np.interp(vr_quantile, cw, y[order]))
    return b1, b2


def region_masks(
    obs: Observables,
    boundaries: tuple[float, float],
    sr_cuts: dict[str, tuple[str, float]] | None = None,
    score_min: float | None = None,
) -> dict[str, np.ndarray]:
    """Boolean masks for SR / VR / CR.

    The y* split defines the three regions.  ``sr_cuts`` (e.g. delta_phi > 2.5,
    mass_asym < 0.4) and the NN ``score_min`` requirement are applied to ALL
    three regions identically — the transfer is only valid if the CR and SR see
    the same selection apart from the y* split itself.
    """
    if obs.y_star is None:
        raise ValueError("Observables.y_star is not set; compute it first.")
    b1, b2 = boundaries
    y = np.asarray(obs.y_star, dtype=float)

    common = np.ones(len(obs), dtype=bool)
    if sr_cuts:
        common &= combined_mask(obs, sr_cuts)
    if score_min is not None:
        if obs.score is None:
            raise ValueError("score_min requested but Observables.score is not set.")
        common &= np.asarray(obs.score, dtype=float) >= score_min

    return {
        "SR": common & (y < b1),
        "VR": common & (y >= b1) & (y < b2),
        "CR": common & (y >= b2),
    }
