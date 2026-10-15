"""
Which mass variable recovers merged / FSR signal?  Compare, after the network's
preselection, the signal peak and the best S/sqrt(B) window for:

  m3        m_avg of the chosen 3+3 groups
  dressedR  chosen groups + every other jet within dR<R of a group axis
            (nearest group), R in {0.8, 1.2, inf}
  thrust    all jets split into hemispheres by the transverse thrust axis
            (no network), m_avg of the two hemisphere masses
  HT        scalar sum of all jet pT

  python scripts/mass_variables.py --model runs/gated_n7/model.pt \
      --sig "data/pythia/sig_*_s2.h5" --bkg data/pythia/qcd_s8.h5 data/pythia/qcd_s9.h5 \
      data/pythia/qcd_s10.h5 --out results/mass_variables
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.hemimix import transverse_thrust_axis  # noqa: E402
from src.pairdata import concat, load_events  # noqa: E402
from src.pairnet import enumerate_interpretations, load_checkpoint, to_cartesian  # noqa: E402

ALL = 20


def m_of(P):
    return np.sqrt(np.maximum(P[..., 0] ** 2 - (P[..., 1:] ** 2).sum(-1), 0))


def eta_phi(P):
    pt = np.hypot(P[..., 1], P[..., 2])
    return np.arcsinh(P[..., 3] / np.maximum(pt, 1e-9)), np.arctan2(P[..., 2], P[..., 1])


@torch.no_grad()
def choose(net, d):
    n = net.n_jets
    t = {k: torch.from_numpy(np.ascontiguousarray(d[k][:, :n])) for k in ("pt", "eta", "phi", "m", "mask")}
    t = {k: (v.float() if v.dtype != torch.bool else v) for k, v in t.items()}
    best, psig = [], []
    for i in range(0, len(t["pt"]), 1024):
        o = net(*(t[k][i : i + 1024] for k in ("pt", "eta", "phi", "m", "mask")))
        A = net.n_assign
        best.append(o["logits"][:, :A].argmax(1))
        psig.append(torch.sigmoid(o["gate_logit"]) if "gate_logit" in o else
                    1 - o["logits"].softmax(1)[:, -1])
    return torch.cat(best).numpy(), torch.cat(psig).numpy()


def variables(d, best, n_net, sizes=((3, 3),)):
    interp = enumerate_interpretations(n_net, sizes)
    N = len(best)
    p4 = to_cartesian(*(torch.from_numpy(d[k]).double() for k in ("pt", "eta", "phi", "m")))
    p4 = (p4 * torch.from_numpy(d["mask"]).unsqueeze(-1)).numpy()  # (N, ALL, 4)
    in1 = np.zeros((N, ALL), bool)
    in2 = np.zeros((N, ALL), bool)
    for i, b in enumerate(best):
        a, c = interp[b]
        in1[i, list(a)] = True
        in2[i, list(c)] = True
    P1 = (p4 * in1[..., None]).sum(1)
    P2 = (p4 * in2[..., None]).sum(1)
    m1, m2 = m_of(P1), m_of(P2)
    out = {"m3": 0.5 * (m1 + m2), "asym": np.abs(m1 - m2) / np.maximum(m1 + m2, 1e-9)}
    # dressing
    jeta, jphi = d["eta"], d["phi"]
    e1, f1 = eta_phi(P1)
    e2, f2 = eta_phi(P2)
    dphi = lambda a, b: (a - b + np.pi) % (2 * np.pi) - np.pi
    dr1 = np.hypot(jeta - e1[:, None], dphi(jphi, f1[:, None]))
    dr2 = np.hypot(jeta - e2[:, None], dphi(jphi, f2[:, None]))
    free = d["mask"] & ~in1 & ~in2
    for R in (0.8, 1.2, np.inf):
        to1 = free & (dr1 <= dr2) & (dr1 < R)
        to2 = free & (dr2 < dr1) & (dr2 < R)
        Q1 = P1 + (p4 * to1[..., None]).sum(1)
        Q2 = P2 + (p4 * to2[..., None]).sum(1)
        key = "dressed" + ("inf" if np.isinf(R) else str(R))
        out[key] = 0.5 * (m_of(Q1) + m_of(Q2))
    # thrust hemispheres (axis from leading 10 jets; all jets assigned)
    px, py = p4[..., 1], p4[..., 2]
    ax = transverse_thrust_axis(px[:, :10], py[:, :10], d["mask"][:, :10])
    side = (px * ax[:, :1] + py * ax[:, 1:]) >= 0
    H1 = (p4 * (side & d["mask"])[..., None]).sum(1)
    H2 = (p4 * (~side & d["mask"])[..., None]).sum(1)
    out["thrust"] = 0.5 * (m_of(H1) + m_of(H2))
    out["HT"] = (d["pt"] * d["mask"]).sum(1)
    return out


def best_window(vs, vb, wb, nsig_tot, wb_tot):
    """Best S/sqrt(B) over windows [q_lo, q_hi] of the signal distribution."""
    if len(vs) < 20:
        return 0.0, (0, 0), 0.0
    qs = np.quantile(vs, np.linspace(0, 1, 41))
    best = (0.0, (0, 0), 0.0)
    for i in range(len(qs)):
        for j in range(i + 2, len(qs)):
            lo, hi = qs[i], qs[j]
            S = ((vs >= lo) & (vs < hi)).sum() / nsig_tot
            B = (wb * ((vb >= lo) & (vb < hi))).sum() / wb_tot
            fom = S / np.sqrt(max(B, 3 * wb.mean() / wb_tot))  # floor: ~3 QCD events
            if fom > best[0]:
                best = (fom, (float(lo), float(hi)), float(S))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sig", nargs="+", required=True)
    ap.add_argument("--bkg", nargs="+", required=True)
    ap.add_argument("--asym-cut", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    net, margs = load_checkpoint(a.model)
    sizes = tuple((int(x[0]), int(x[1])) for x in margs["sizes"].split(",")) if margs.get("sizes") else ((3, 3),)
    sig = load_events(a.sig, ALL, 3, True, None, min_jets=6)
    bkg = load_events(a.bkg, ALL, 3, False, None, min_jets=6)
    d = concat(sig, bkg)
    best, psig = choose(net, d)
    V = variables(d, best, net.n_jets, sizes)
    s = d["is_signal"]
    pre = V["asym"] < a.asym_cut
    wb = d["w"][~s]
    res = {}
    names = ["m3", "dressed0.8", "dressed1.2", "dressedinf", "thrust", "HT"]
    for m in np.unique(d["pmass"][s]):
        sm = s & (d["pmass"] == m)
        r = {}
        for v in names:
            vs = V[v][sm & pre]
            vb = V[v][~s & pre]
            fom, win, S = best_window(vs, vb, wb[pre[~s]], sm.sum(), wb.sum())
            med = np.median(vs)
            q16, q84 = np.quantile(vs, [0.16, 0.84])
            r[v] = {"fom": float(fom), "window": win, "sig_in_window": float(S), "median_over_M": float(med / m),
                    "rel_width": float((q84 - q16) / 2 / med)}
        res[int(m)] = r
        print(int(m), {v: round(r[v]["fom"], 1) for v in names},
              "| S:", {v: round(r[v]["sig_in_window"], 3) for v in names})
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "vars.npz"), is_signal=s, pmass=d["pmass"], w=d["w"],
                        psig=psig, **V)


if __name__ == "__main__":
    main()
