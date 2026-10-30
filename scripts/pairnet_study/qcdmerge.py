"""Merged QCD for PairNet studies: validation+test files of the standard sample (big_vars.npz, has_pn) plus the high-HT
sample (big_vars_ht.npz). Above HT_CUT the two estimate the same cross section and are averaged with weights
proportional to their effective MC statistics. Returns per-event arrays with a cross-section weight in fb."""
import glob
import h5py
import numpy as np

HT_CUT = 1200.0
P = "/scratch/llee/pairnet/data_r3/prod"


def neff(x):
    return x.sum() ** 2 / max((x ** 2).sum(), 1e-300)


def load(big, ht_npy, big_ht, keys, pn_only=True, test_files_only=False):
    """keys: per-event columns to return (present in both samples). pn_only: old sample restricted to has_pn files.
    test_files_only: use only held-out high-HT files c1700-1999 (for models trained on c0-1399)."""
    v = np.load(big)
    ht = np.load(ht_npy)
    q = ~v["is_signal"] & (v["has_pn"] if pn_only else True)
    fac = {f: h5py.File(f).attrs["sigma_mb"] / h5py.File(f).attrs["weight_sum"] * 1e12 for f in sorted(glob.glob(f"{P}/qcd_c*.h5"))}
    files = [f for f in fac if (f[-4] in "89" or not pn_only)]
    k_old = np.mean([fac[f] for f in files]) / len(files)
    wo = v["w"][q].astype(np.float64) * k_old
    hto = ht[q]
    h = np.load(big_ht)
    nf = int(h["n_files"])
    wn = h["w"].astype(np.float64) * h["fac"] / nf
    sel_new = np.ones(len(wn), bool)
    if test_files_only:
        raise NotImplementedError("needs a file index in big_vars_ht.npz")
    a, b = neff(wo[hto > HT_CUT]), neff(wn[sel_new])
    alpha = a / (a + b)
    wo = np.where(hto > HT_CUT, wo * alpha, wo)
    wn = wn * (1 - alpha)
    out = {k: np.concatenate([v[k][q], h[k][sel_new]]) for k in keys if k != "ht"}
    out["ht"] = np.concatenate([hto, h["ht"][sel_new]])
    out["wfb"] = np.concatenate([wo, wn[sel_new]])
    out["is_new"] = np.concatenate([np.zeros(len(wo), bool), np.ones(sel_new.sum(), bool)])
    info = {"alpha_old": alpha, "neff_old_HT>cut": a, "neff_new": b,
            "check": {x: (float(wo[hto > x].sum() / alpha), float(wn[h["ht"] > x].sum() / (1 - alpha))) for x in (1500, 2000, 2500, 3000)}}
    return out, info
