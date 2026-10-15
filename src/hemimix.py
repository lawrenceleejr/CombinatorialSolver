"""
Hemisphere mixing: build fake 6-jet QCD events from 5-jet control-region events.

Mirrors the analysis' background estimate:
  1. split each 5-jet event into two hemispheres by its transverse thrust axis;
  2. keep the 3-jet hemispheres as a library;
  3. pair each library hemisphere with a *compatible* 3-jet hemisphere from a
     different event (nearest neighbour in hemisphere kinematics), placing the
     partner back-to-back along the thrust axis;
  4. the resulting 6-jet events are fed to the network exactly like data.

Closure is then judged by comparing network outputs on mixed events with real
6-jet QCD.  Anything the network learns from correlations BETWEEN the two
halves of an event (absent in mixed events by construction) shows up as
non-closure.

mix_replace() follows the analysis as described: keep the real 3-jet hemisphere
of a 5-jet event and replace its 2-jet hemisphere by a compatible 3-jet one that
balances it along the thrust axis (axis preserved) at the same hemisphere eta.
mix() is the simpler "pair two random hemispheres" variant, kept for comparison.
"""

import numpy as np
from scipy.spatial import cKDTree


def transverse_thrust_axis(px, py, mask):
    """Exact transverse thrust axis by enumerating sign assignments.

    The axis maximising sum|p_T . n| is parallel to sum_i s_i p_Ti for the best
    choice of signs s_i (jet 0 fixed to +1).  Returns unit vectors (N, 2).
    """
    n_ev, nj = px.shape
    best = np.full(n_ev, -1.0)
    axis = np.zeros((n_ev, 2))
    for code in range(2 ** (nj - 1)):
        s = np.array([1] + [1 if (code >> b) & 1 else -1 for b in range(nj - 1)], dtype=float)
        sx = (px * s * mask).sum(1)
        sy = (py * s * mask).sum(1)
        norm = np.hypot(sx, sy)
        upd = norm > best
        best = np.where(upd, norm, best)
        axis[upd] = np.stack([sx[upd], sy[upd]], 1) / np.maximum(norm[upd], 1e-9)[:, None]
    return axis


def split_hemispheres(pt, eta, phi, m, mask):
    """Returns per-event thrust axis angle and hemisphere side (+1/-1) of each jet."""
    px, py = pt * np.cos(phi), pt * np.sin(phi)
    ax = transverse_thrust_axis(px, py, mask)
    proj = px * ax[:, :1] + py * ax[:, 1:]
    side = np.where(proj >= 0, 1, -1) * mask
    return np.arctan2(ax[:, 1], ax[:, 0]), side


def _hemi_features(pt, eta, phi, m, phi_ax):
    """Compatibility variables for a 3-jet hemisphere expressed in its thrust frame."""
    dphi = phi - phi_ax[:, None]
    px = pt * np.cos(dphi)  # along thrust axis
    py = pt * np.sin(dphi)  # perpendicular
    pz = pt * np.sinh(eta)
    e = np.sqrt((pt * np.cosh(eta)) ** 2 + m**2)
    P = np.stack([e.sum(1), px.sum(1), py.sum(1), pz.sum(1)], 1)
    mass = np.sqrt(np.maximum(P[:, 0] ** 2 - (P[:, 1:] ** 2).sum(1), 0))
    return np.stack([mass, P[:, 1], np.abs(P[:, 2]), np.abs(P[:, 3])], 1)


def build_library(pt, eta, phi, m, mask, w, n_hemi_jets=3):
    """Collect all n_hemi_jets-jet hemispheres from the given events.

    Each hemisphere is stored rotated into its thrust frame (thrust axis -> phi=0,
    hemisphere on the +x side), with eta flipped so its total Pz >= 0.
    """
    phi_ax, side = split_hemispheres(pt, eta, phi, m, mask)
    H = {k: [] for k in ("pt", "eta", "phi", "m", "feat", "ev", "w")}
    for sgn in (1, -1):
        sel = side == sgn
        ok = sel.sum(1) == n_hemi_jets
        idx = np.where(ok)[0]
        if len(idx) == 0:
            continue
        # gather the jets of that hemisphere (pT order preserved)
        order = np.argsort(~sel[idx], axis=1, kind="stable")[:, :n_hemi_jets]
        g = lambda a: np.take_along_axis(a[idx], order, 1)
        hpt, heta, hphi, hm = g(pt), g(eta), g(phi), g(m)
        ax = phi_ax[idx] + (0 if sgn == 1 else np.pi)
        hphi = (hphi - ax[:, None] + np.pi) % (2 * np.pi) - np.pi
        flip = np.where((hpt * np.sinh(heta)).sum(1) < 0, -1.0, 1.0)
        heta = heta * flip[:, None]
        H["pt"].append(hpt)
        H["eta"].append(heta)
        H["phi"].append(hphi)
        H["m"].append(hm)
        H["feat"].append(_hemi_features(hpt, heta, hphi, hm, np.zeros(len(idx))))
        H["ev"].append(idx)
        H["w"].append(w[idx])
    return {k: np.concatenate(v) for k, v in H.items()}


def mix(lib, n_out=None, k_neighbours=4, seed=0, random_eta_flip=True):
    """Pair each hemisphere with a compatible one from a different event.

    Compatibility = nearest neighbours in standardised (mass, p_par, |p_perp|, |Pz|).
    The partner is rotated by pi (back-to-back in the transverse plane) and, with
    probability 1/2, mirrored in eta.  Returns 6-jet events, pT-sorted.
    """
    rng = np.random.default_rng(seed)
    f = lib["feat"]
    z = (f - f.mean(0)) / f.std(0).clip(min=1e-9)
    tree = cKDTree(z)
    n = len(f)
    a_idx = np.arange(n) if n_out is None else rng.choice(n, n_out, replace=n_out > n)
    _, nn = tree.query(z[a_idx], k=k_neighbours + 1)
    pick = nn[np.arange(len(a_idx)), rng.integers(1, k_neighbours + 1, len(a_idx))]
    same = lib["ev"][pick] == lib["ev"][a_idx]
    pick[same] = nn[same, 1 if k_neighbours > 1 else 0]  # fall back to another neighbour
    b_idx = pick

    pt = np.concatenate([lib["pt"][a_idx], lib["pt"][b_idx]], 1)
    phi_b = (lib["phi"][b_idx] + np.pi + np.pi) % (2 * np.pi) - np.pi
    phi = np.concatenate([lib["phi"][a_idx], phi_b], 1)
    eflip = np.where(rng.random(len(a_idx)) < 0.5, -1.0, 1.0)[:, None] if random_eta_flip else 1.0
    eta = np.concatenate([lib["eta"][a_idx], lib["eta"][b_idx] * eflip], 1)
    m = np.concatenate([lib["m"][a_idx], lib["m"][b_idx]], 1)
    # random overall azimuth (irrelevant to the network after canonicalisation)
    phi = (phi + rng.uniform(-np.pi, np.pi, (len(a_idx), 1)) + np.pi) % (2 * np.pi) - np.pi
    order = np.argsort(-pt, 1)
    t = lambda a: np.take_along_axis(a, order, 1)
    w = np.sqrt(lib["w"][a_idx] * lib["w"][b_idx])
    return {"pt": t(pt), "eta": t(eta), "phi": t(phi), "m": t(m),
            "mask": np.ones_like(pt, dtype=bool), "w": w}


def _hemi_summary(pt, eta, phi, m, sel, phi_ax):
    """Per-event 4-momentum of the jets in `sel`, in the frame with thrust axis at phi=0.

    Returns (p_par, p_perp, eta_hemi, mass)."""
    dphi = phi - phi_ax[:, None]
    w = sel.astype(float)
    px = (pt * np.cos(dphi) * w).sum(1)
    py = (pt * np.sin(dphi) * w).sum(1)
    pz = (pt * np.sinh(eta) * w).sum(1)
    e = (np.sqrt((pt * np.cosh(eta)) ** 2 + m**2) * w).sum(1)
    ptot = np.hypot(px, py)
    eta_h = np.arcsinh(pz / np.maximum(ptot, 1e-9))
    mass = np.sqrt(np.maximum(e**2 - px**2 - py**2 - pz**2, 0))
    return px, py, eta_h, mass


def mix_replace(pt, eta, phi, m, mask, w, k_neighbours=4, seed=0,
                match=("p_par", "p_perp", "eta")):
    """Hemisphere replacement (the analysis' method, as described):

    For each 5-jet event splitting 3+2 about its transverse thrust axis, keep the
    3-jet hemisphere A and replace the 2-jet hemisphere B with a 3-jet hemisphere C
    taken from a different event.  C is the nearest neighbour to B in
      p_par  : momentum along the thrust axis (so C balances A as B did and the
               thrust axis of the fake event is preserved),
      p_perp : |momentum perpendicular to the axis|,
      eta    : hemisphere pseudorapidity (C must sit where B sat),
    with C rotated onto B's side of the axis and mirrored in eta to match B's sign.
    Returns 6-jet events, pT-sorted.
    """
    rng = np.random.default_rng(seed)
    phi_ax, side = split_hemispheres(pt, eta, phi, m, mask)
    n_plus, n_minus = (side == 1).sum(1), (side == -1).sum(1)
    three_side = np.where(n_plus == 3, 1, np.where(n_minus == 3, -1, 0))
    ok = (mask.sum(1) == 5) & (((n_plus == 3) & (n_minus == 2)) | ((n_plus == 2) & (n_minus == 3)))
    ev = np.where(ok)[0]

    # library of 3-jet hemispheres in their own thrust frame (+x side, Pz >= 0)
    lib = build_library(pt, eta, phi, m, mask, w, n_hemi_jets=3)
    lpx, lpy, leta, lmass = _hemi_summary(lib["pt"], lib["eta"], lib["phi"], lib["m"],
                                          np.ones_like(lib["pt"], bool), np.zeros(len(lib["pt"])))
    lib_feat = {"p_par": lpx, "p_perp": np.abs(lpy), "eta": np.abs(leta), "mass": lmass}

    # the 2-jet hemisphere B of each event, expressed so B is on the +x side
    sideB = -three_side[ev]
    axB = phi_ax[ev] + np.where(sideB == 1, 0.0, np.pi)
    selB = side[ev] == sideB[:, None]
    bpx, bpy, beta, bmass = _hemi_summary(pt[ev], eta[ev], phi[ev], m[ev], selB, axB)
    b_feat = {"p_par": bpx, "p_perp": np.abs(bpy), "eta": np.abs(beta), "mass": bmass}

    L = np.stack([lib_feat[k] for k in match], 1)
    Q = np.stack([b_feat[k] for k in match], 1)
    mu, sd = L.mean(0), L.std(0).clip(min=1e-9)
    tree = cKDTree((L - mu) / sd)
    _, nn = tree.query((Q - mu) / sd, k=k_neighbours + 1)
    choice = rng.integers(0, k_neighbours + 1, len(ev))
    pick = nn[np.arange(len(ev)), choice]
    for j in range(k_neighbours + 1):  # never reuse the event's own hemisphere
        same = lib["ev"][pick] == ev
        if not same.any():
            break
        pick[same] = nn[same, (choice[same] + j + 1) % (k_neighbours + 1)]

    # place C: rotate its frame onto B's axis, mirror eta to B's sign
    cphi = (lib["phi"][pick] + axB[:, None] + np.pi) % (2 * np.pi) - np.pi
    ceta = lib["eta"][pick] * np.where(beta < 0, -1.0, 1.0)[:, None]
    cpt, cm = lib["pt"][pick], lib["m"][pick]
    selA = side[ev] == three_side[ev][:, None]
    oA = np.argsort(~selA, 1, kind="stable")[:, :3]
    g = lambda a: np.take_along_axis(a[ev], oA, 1)
    out_pt = np.concatenate([g(pt), cpt], 1)
    out_eta = np.concatenate([g(eta), ceta], 1)
    out_phi = np.concatenate([g(phi), cphi], 1)
    out_m = np.concatenate([g(m), cm], 1)
    order = np.argsort(-out_pt, 1)
    t = lambda a: np.take_along_axis(a, order, 1)
    return {"pt": t(out_pt), "eta": t(out_eta), "phi": t(out_phi), "m": t(out_m),
            "mask": np.ones_like(out_pt, dtype=bool), "w": w[ev] * 1.0,
            "phi_axis_orig": phi_ax[ev]}
