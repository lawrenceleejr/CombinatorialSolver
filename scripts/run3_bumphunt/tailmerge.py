"""Merge enriched-tail QCD samples (generated with an in-job PairNet filter) into the existing QCD estimate.

A tail sample only contains events passing its filter F (r14A pairing valid, A < 0.1, m_avg in its range, and
P(signal) >= PMIN or min(Delta)/m_avg >= DMIN), with the normalisation of the full generated sample. Inside F and its
complete H_T range it is an unbiased estimate of the same cross section as the existing samples, so per H_T region the
two are combined with weights proportional to their effective MC statistics inside F; outside F (or below its
complete H_T) a tail sample carries no weight and the existing samples are unchanged.
"""
import glob, os, time
import numpy as np

REGIONS = [(400, 900), (900, 1200), (1200, 1e9)]
# m_avg ranges: the generator-level symmetric-split filter (--sym-filter LO,HI) makes a sample complete only for
# LO < m_avg < HI (any A < 0.1 pairing with m_avg in range passes it), so the merge uses exactly that range.
TAILS = {"qcdvlmT": dict(complete=400, pmin=0.8, dmin=0.4, mlo=150, mhi=1150),
         "qcdhtT": dict(complete=900, pmin=0.8, dmin=0.4, mlo=800, mhi=3000)}


def neff(w):
    return w.sum() ** 2 / max((w ** 2).sum(), 1e-300)


def in_filter(valid, m1, m2, psig, d1, d2, spec, use_delta=True):
    ma = 0.5 * (m1 + m2); A = np.abs(m1 - m2) / np.maximum(m1 + m2, 1e-9)
    ok = valid.astype(bool) & (A < 0.1) & (ma > spec["mlo"]) & (ma < spec["mhi"])
    sel = psig >= spec["pmin"]
    if use_delta:
        sel |= np.minimum(d1, d2) / np.maximum(ma, 1e-9) >= spec["dmin"]
    return ok & sel


def load_tail(tdir, name, model, first, last=10 ** 9):
    """Concatenate the scored tail npz files c<first>..c<last> of one tail sample; weights in fb (cross section per
    event), averaged over the files. Returns (dict of arrays, number of files)."""
    parts, nf = [], 0
    for f in sorted(glob.glob(f"{tdir}/{name}_c*.npz")):
        c = int(f.rsplit("_c", 1)[1].split(".")[0])
        if not first <= c <= last:
            continue
        try:
            if time.time() - os.path.getmtime(f) < 120:
                continue
            a = dict(np.load(f))
        except Exception:  # noqa: BLE001  (truncated transfer)
            continue
        nf += 1
        if len(a["w"]) == 0:
            continue
        a["wfb"] = a["w"] * float(a["attr_sigma_mb"]) * 1e12 / float(a["attr_weight_sum"])
        parts.append(a)
    if not parts:
        return None, nf
    keys = [k for k in parts[0] if np.ndim(parts[0][k]) == 1 and len(parts[0][k]) == len(parts[0]["w"])]
    d = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    d["wfb"] = d["wfb"] / nf
    return d, nf


def combine(base, tail, spec):
    """base: list of (w, ht, F) for the existing samples (w modified in place); tail: (w, ht, F). Returns the tail
    weights and a summary of the per-region combination."""
    tw, tht, tF = tail
    out = np.zeros_like(tw); info = {}
    for lo, hi in REGIONS:
        if lo < spec["complete"]:
            continue
        bm = [F & (ht >= lo) & (ht < hi) for w, ht, F in base]
        nb = neff(np.concatenate([w[m] for (w, ht, F), m in zip(base, bm)])) if any(m.any() for m in bm) else 0.0
        tm = tF & (tht >= lo) & (tht < hi)
        nt = neff(tw[tm]) if tm.any() else 0.0
        if nt <= 0:
            continue
        a = nb / (nb + nt)
        for (w, ht, F), m in zip(base, bm):
            w[m] *= a
        out[tm] = tw[tm] * (1 - a)
        info[f"{lo}-{hi:g}"] = {"neff_base_in_F": float(nb), "neff_tail_in_F": float(nt), "alpha_tail": float(1 - a)}
    return out, info
