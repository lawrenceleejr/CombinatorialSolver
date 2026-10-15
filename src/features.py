"""
Expert four-vector features for a conventional (BDT) signal-vs-QCD baseline.

All features are dimensionless (ratios to HT or masses, angles, shapes), the
same scale-free condition the PairNet inputs satisfy, so a classifier built on
them cannot key on the absolute mass.

event_shape_features(d)          event-level shapes from all jets
pairing_features(d, g1, g2)      features of a chosen (triplet, triplet) pairing
classical_pairing(d, n)          argmin |m1 - m2| over the leading n jets
"""

import numpy as np

from .hemimix import transverse_thrust_axis
from .pairnet import enumerate_interpretations


def four_vectors(d):
    pt, eta, phi, m, mask = d["pt"], d["eta"], d["phi"], d["m"], d["mask"]
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    e = np.sqrt(px**2 + py**2 + pz**2 + m**2)
    p = np.stack([e, px, py, pz], -1) * mask[..., None]
    return p


def inv_mass(P):
    return np.sqrt(np.maximum(P[..., 0] ** 2 - (P[..., 1:] ** 2).sum(-1), 0))


def rapidity(P):
    return 0.5 * np.log(np.maximum(P[..., 0] + P[..., 3], 1e-9) / np.maximum(P[..., 0] - P[..., 3], 1e-9))


def event_shape_features(d):
    p = four_vectors(d)
    mask = d["mask"]
    pt = d["pt"] * mask
    ht = pt.sum(1)
    pabs = np.linalg.norm(p[..., 1:], axis=-1)
    f = {}
    f["njets"] = mask.sum(1).astype(float)
    for i in range(8):
        f[f"ptfrac{i}"] = pt[:, i] / ht if pt.shape[1] > i else np.zeros(len(ht))
    # 3D sphericity tensor
    S = np.einsum("nja,njb->nab", p[..., 1:], p[..., 1:]) / np.maximum((pabs**2).sum(1), 1e-9)[:, None, None]
    lam = np.sort(np.linalg.eigvalsh(S), axis=1)[:, ::-1]
    f["sphericity"] = 1.5 * (lam[:, 1] + lam[:, 2])
    f["aplanarity"] = 1.5 * lam[:, 2]
    # transverse sphericity
    St = np.einsum("nja,njb->nab", p[..., 1:3], p[..., 1:3])
    lt = np.sort(np.linalg.eigvalsh(St), axis=1)[:, ::-1]
    f["sphericity_t"] = 2 * lt[:, 1] / np.maximum(lt.sum(1), 1e-9)
    # transverse thrust from the leading 10 jets
    n10 = min(10, p.shape[1])
    ax = transverse_thrust_axis(p[:, :n10, 1], p[:, :n10, 2], mask[:, :n10])
    proj = np.abs(p[..., 1] * ax[:, :1] + p[..., 2] * ax[:, 1:]) * mask
    f["thrust_t"] = proj.sum(1) / ht
    f["centrality"] = ht / np.maximum(p[..., 0].sum(1), 1e-9)
    Ptot = p.sum(1)
    f["m_all_over_ht"] = inv_mass(Ptot) / ht
    f["mht_over_ht"] = np.hypot(Ptot[:, 1], Ptot[:, 2]) / ht
    # Fox-Wolfram moments (lab frame, |p| weights)
    cos = np.einsum("nia,nja->nij", p[..., 1:], p[..., 1:]) / np.maximum(
        pabs[:, :, None] * pabs[:, None, :], 1e-9)
    w = pabs[:, :, None] * pabs[:, None, :]
    norm = np.maximum(pabs.sum(1) ** 2, 1e-9)
    leg = {2: lambda x: 0.5 * (3 * x**2 - 1), 3: lambda x: 0.5 * (5 * x**3 - 3 * x),
           4: lambda x: (35 * x**4 - 30 * x**2 + 3) / 8}
    for l, P in leg.items():
        f[f"fw{l}"] = (w * P(cos)).sum((1, 2)) / norm
    aeta = np.abs(d["eta"]) * mask
    f["mean_abs_eta6"] = aeta[:, :6].mean(1)
    f["max_abs_eta6"] = aeta[:, :6].max(1)
    return f


def classical_pairing(d, n=7):
    interp = enumerate_interpretations(n, ((3, 3),))
    g1 = np.array([a for a, _ in interp])
    g2 = np.array([b for _, b in interp])
    p = four_vectors(d)[:, :n]
    m1 = inv_mass(p[:, g1].sum(2))
    m2 = inv_mass(p[:, g2].sum(2))
    valid = d["mask"][:, :n][:, g1].all(-1) & d["mask"][:, :n][:, g2].all(-1)
    best = np.where(valid, np.abs(m1 - m2), np.inf).argmin(1)
    return g1[best], g2[best]


def pairing_features(d, g1, g2, n=7):
    """g1, g2: (N, 3) jet indices into the pT-sorted jets."""
    p = four_vectors(d)
    N = len(g1)
    ar = np.arange(N)[:, None]
    J1, J2 = p[ar, g1], p[ar, g2]  # (N, 3, 4)
    P1, P2 = J1.sum(1), J2.sum(1)
    m1, m2 = inv_mass(P1), inv_mass(P2)
    ht = (d["pt"] * d["mask"]).sum(1)
    f = {}
    f["asym"] = np.abs(m1 - m2) / np.maximum(m1 + m2, 1e-9)
    f["mavg_over_ht"] = 0.5 * (m1 + m2) / ht
    y1, y2 = rapidity(P1), rapidity(P2)
    f["dy_tri"] = np.abs(y1 - y2)
    f["cos_theta_star"] = np.abs(np.tanh(0.5 * (y1 - y2)))
    ph1, ph2 = np.arctan2(P1[:, 2], P1[:, 1]), np.arctan2(P2[:, 2], P2[:, 1])
    f["dphi_tri"] = np.abs((ph1 - ph2 + np.pi) % (2 * np.pi) - np.pi)
    pt1, pt2 = np.hypot(P1[:, 1], P1[:, 2]), np.hypot(P2[:, 1], P2[:, 2])
    f["pt_balance"] = np.abs(pt1 - pt2) / np.maximum(pt1 + pt2, 1e-9)
    bo = np.sort(np.stack([pt1 / np.maximum(m1, 1e-9), pt2 / np.maximum(m2, 1e-9)], 1), 1)
    f["boost_lo"], f["boost_hi"] = bo[:, 0], bo[:, 1]
    for k, (J, P, mm) in enumerate(((J1, P1, m1), (J2, P2, m2))):
        pairs = [(0, 1), (0, 2), (1, 2)]
        dal = np.stack([inv_mass(J[:, a] + J[:, b]) ** 2 for a, b in pairs], 1) / np.maximum(mm**2, 1e-9)[:, None]
        dal = np.sort(dal, 1)
        f[f"dalitz{k}_lo"], f[f"dalitz{k}_mid"], f[f"dalitz{k}_hi"] = dal[:, 0], dal[:, 1], dal[:, 2]
        spt = np.hypot(J[..., 1], J[..., 2]).sum(1)
        f[f"cms_delta{k}"] = (spt - mm) / np.maximum(mm, 1e-9)
    # the leading-n jet left out of both groups (ISR candidate)
    used = np.zeros((N, n), bool)
    used[ar, g1] = True
    used[ar, g2] = True
    left = (~used) & d["mask"][:, :n]
    lpt = (d["pt"][:, :n] * left).sum(1)
    f["isr_ptfrac"] = lpt / ht
    f["isr_abs_eta"] = (np.abs(d["eta"][:, :n]) * left).sum(1)
    return f


def stack(fdicts):
    keys = [k for fd in fdicts for k in fd]
    X = np.stack([fd[k] for fd in fdicts for k in fd], 1).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), keys
