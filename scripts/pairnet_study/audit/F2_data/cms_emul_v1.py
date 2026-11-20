"""F2 item 4b: emulate the CMS-EXO-21-004 resolved-trijet selection (arXiv:2404.02992, Table 1) on Pythia QCD h5 files.
usage: cms_emul.py out.npz nproc file1.h5 [file2.h5 ...]   (or @listfile)
Jets pT>30, |eta|<2.4; H_T>600; six leading; per region: pT6, D2[(6,3)+(3,2)], A_m per triplet pair, Delta and D2[3,2] per triplet;
all passing triplets fill m_jjj.  Variants: 'noqgd' and (if F2/flav present) 'qgd' = quark-gluon discriminator emulated with the
efficiencies quoted in the paper (loose/medium/tight: quark 0.98/0.83/0.61, gluon 0.69/0.30/0.13), nested working points,
one uniform random number per jet; jets failing loose are removed before the six leading are chosen.
Output: per region and variant the per-bin cross-section [fb] and its MC stat error, averaged over files; extra H_T splits.
"""
import sys
from itertools import combinations
from multiprocessing import Pool
import h5py, numpy as np

REG = {1: dict(pt6=40, d63=1.25, am=0.25, delta=250, d32=0.05, edges=np.arange(100, 610.1, 5.0)),
       2: dict(pt6=50, d63=1.00, am=0.175, delta=180, d32=0.175, edges=np.arange(300, 1200.1, 10.0)),
       3: dict(pt6=100, d63=0.9, am=0.15, delta=150, d32=0.2, edges=np.arange(520, 2500.1, 20.0))}
T3 = np.array(list(combinations(range(6), 3)))            # 20 triplets
COMP = np.array([[i for i in range(6) if i not in t] for t in T3])
PAIR = [(i, j) for i in range(20) for j in range(i + 1, 20) if set(T3[i]) | set(T3[j]) == set(range(6))]  # 10 pairs
PARTNER = np.zeros(20, int)
for i, j in PAIR: PARTNER[i], PARTNER[j] = j, i
QEFF = {"q": (0.98, 0.83, 0.61), "g": (0.69, 0.30, 0.13)}
mm = lambda v: np.sqrt(np.maximum(v[..., 0] ** 2 - (v[..., 1:] ** 2).sum(-1), 0))
PTH = [0, 80, 120, 200, 250, 1e9]


def d2_32(t4):
    mi2 = mm(t4) ** 2; mt2 = mm(t4.sum(-2)) ** 2; den = np.maximum(mt2 + mi2.sum(-1), 1e-12)
    mh = np.stack([np.sqrt(mm(t4[..., i, :] + t4[..., j, :]) ** 2 / den) for i, j in ((0, 1), (0, 2), (1, 2))], -1)
    return ((mh - 1 / np.sqrt(3)) ** 2).sum(-1)


def one(f):
    with h5py.File(f, "r") as h:
        s = h["INPUTS/Source"]; pt, eta, phi, m = (s[k][:].astype(np.float64) for k in ("pt", "eta", "phi", "mass"))
        w = h["EventVars/normweight"][:]; norm = h.attrs["sigma_mb"] * 1e12 / h.attrs["weight_sum"]
        flav = h["F2/flav"][:] if "F2/flav" in h else None
        pthat = h["EventVars/pthat"][:] if "EventVars/pthat" in h else None
    w = w * norm
    rng = np.random.RandomState(abs(hash(f)) % (2 ** 31))
    good = (pt > 30) & (np.abs(eta) < 2.4)
    ht = (pt * good).sum(1)
    out = {}
    variants = ["noqgd"] + (["qgd"] if flav is not None else [])
    for var in variants:
        g = good.copy(); wp = None
        if var == "qgd":
            u = rng.uniform(size=pt.shape); isq = (np.abs(flav) >= 1) & (np.abs(flav) <= 5)
            e = np.where(isq[..., None], np.array(QEFF["q"]), np.array(QEFF["g"]))   # (N, MJ, 3)
            wp = (u[..., None] < e).sum(-1)                                          # 0 fail, 1 loose, 2 medium, 3 tight
            g &= wp >= 1
        order = np.argsort(np.where(g, -pt, np.inf), axis=1, kind="stable")[:, :6]
        ok = (g.sum(1) >= 6) & (ht > 600)
        ev = np.where(ok)[0]
        if len(ev) == 0:
            for r in REG: out[f"{var}_r{r}"] = np.zeros((2, len(REG[r]["edges"]) - 1)); out[f"{var}_r{r}_pth"] = np.zeros((len(PTH) - 1,))
            continue
        tk = lambda x: np.take_along_axis(x, order, 1)[ev]
        p, e_, ph, ms = tk(pt), tk(eta), tk(phi), tk(m)
        px, py, pz = p * np.cos(ph), p * np.sin(ph), p * np.sinh(e_)
        p4 = np.stack([np.sqrt(px ** 2 + py ** 2 + pz ** 2 + ms ** 2), px, py, pz], -1)   # (n,6,4)
        tr = p4[:, T3]                                                                    # (n,20,3,4)
        mt = mm(tr.sum(2)); d32 = d2_32(tr); spt = p[:, T3].sum(2)
        m6 = mm(p4.sum(1)); den63 = np.maximum(4 * m6 ** 2 + 6 * (ms ** 2).sum(1), 1e-12)
        d63 = ((np.sqrt(mt ** 2 / den63[:, None] + d32) - 1 / np.sqrt(20)) ** 2).sum(1)
        am = np.abs(mt - mt[:, PARTNER]) / np.maximum(mt + mt[:, PARTNER], 1e-9)
        delta = spt - mt
        if wp is not None:
            wj = tk(wp)[:, T3]; qg = (wj >= 2).all(2) & (wj >= 3).any(2)
        else:
            qg = np.ones_like(mt, bool)
        we = w[ev]; pthat_e = pthat[ev] if pthat is not None else None
        for r, c in REG.items():
            evok = (p[:, 5] > c["pt6"]) & (d63 < c["d63"])
            tok = evok[:, None] & (am < c["am"]) & (delta > c["delta"]) & (d32 < c["d32"]) & qg
            ww = np.broadcast_to(we[:, None], mt.shape)[tok]
            hh, _ = np.histogram(mt[tok], c["edges"], weights=ww); h2, _ = np.histogram(mt[tok], c["edges"], weights=ww ** 2)
            out[f"{var}_r{r}"] = np.stack([hh, h2])
            if pthat_e is not None:
                pp = np.broadcast_to(pthat_e[:, None], mt.shape)[tok]
                out[f"{var}_r{r}_pth"] = np.histogram(pp, PTH, weights=ww)[0]
            else:
                out[f"{var}_r{r}_pth"] = np.zeros(len(PTH) - 1)
    return out


if __name__ == "__main__":
    outp, nproc = sys.argv[1], int(sys.argv[2]); files = sys.argv[3:]
    if files[0].startswith("@"): files = open(files[0][1:]).read().split()
    with Pool(nproc) as pool: res = pool.map(one, files, chunksize=4)
    keys = sorted({k for r in res for k in r}); nf = len(files); o = {"nfiles": nf}
    for k in keys:
        a = np.sum([r[k] for r in res if k in r], 0) / nf
        if a.ndim == 2: o[k] = a[0]; o[k + "_err"] = np.sqrt(a[1] / nf)   # var of mean over files: sum w^2 / nf^2
        else: o[k] = a
    for r, c in REG.items(): o[f"edges_r{r}"] = c["edges"]
    np.savez(outp, **o); print("wrote", outp, nf, "files")
