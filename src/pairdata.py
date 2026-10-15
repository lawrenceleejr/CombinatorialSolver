"""
Loading jet 4-vectors + truth interpretation for PairNet.

Reads MadGraphMLProducer / SPANet-style h5 (INPUTS/Source/*, optional
TARGETS/g{1,2}/j{1,2,3}) and the Pythia testbed files written by
scripts/generate_pythia.py.  Only (pt, eta, phi, mass) of jets are used.

Truth is taken from TARGETS (generator parentage), never from a mass heuristic.
An event is "reconstructable" when all 2k decay products are matched to
distinct jets among the leading n_jets; otherwise its target is -1.
"""

import glob

import h5py
import numpy as np
import torch

from .pairnet import enumerate_pairings


def _pair_lookup(n_jets, k):
    g1, g2 = enumerate_pairings(n_jets, k)
    return {(tuple(a), tuple(b)): i for i, (a, b) in enumerate(zip(g1.tolist(), g2.tolist()))}


def load_events(paths, n_jets=8, k=3, is_signal=True, max_events=None):
    files = []
    for p in paths if isinstance(paths, (list, tuple)) else [paths]:
        files += sorted(glob.glob(p)) or [p]
    lut = _pair_lookup(n_jets, k)
    out = {k_: [] for k_ in ["pt", "eta", "phi", "m", "mask", "target", "pmass", "w", "njall"]}
    for fp in files:
        with h5py.File(fp, "r") as f:
            src = f["INPUTS/Source"]
            pt, eta, phi, m = (src[x][:] for x in ("pt", "eta", "phi", "mass"))
            mask = src["MASK"][:].astype(bool)
            n = len(pt)
            # pT-sort (descending) within real jets
            key = np.where(mask, -pt, np.inf)
            order = np.argsort(key, axis=1, kind="stable")
            take = lambda a: np.take_along_axis(a, order, 1)
            pt, eta, phi, m, mask = map(take, (pt, eta, phi, m, mask))
            inv = np.argsort(order, axis=1)  # original index -> sorted position
            njall = mask.sum(1)
            target = np.full(n, -1, dtype=np.int64)
            if is_signal and "TARGETS" in f:
                tg = np.stack(
                    [f[f"TARGETS/g{g}/j{j}"][:] for g in (1, 2) for j in (1, 2, 3)], 1
                ).astype(np.int64)
                kk = tg.shape[1] // 2
                for i in range(n):
                    t = tg[i]
                    if (t < 0).any() or not mask[i, t].all():
                        continue
                    s = inv[i, t]
                    if (s >= n_jets).any() or len(set(s.tolist())) != 2 * kk:
                        continue
                    a, b = tuple(sorted(s[:kk].tolist())), tuple(sorted(s[kk:].tolist()))
                    a, b = min(a, b), max(a, b)
                    target[i] = lut[(a, b)]
            if "EventVars/parent_mass" in f:
                pm = f["EventVars/parent_mass"][:]
            else:
                pm = np.full(n, float(f.attrs.get("mass", 0.0)) if is_signal else 0.0)
            w = f["EventVars/normweight"][:] if "EventVars/normweight" in f else np.ones(n)

            sl = slice(0, n_jets)
            pad = max(0, n_jets - pt.shape[1])
            padf = lambda a: np.pad(a[:, sl], ((0, 0), (0, pad)))
            out["pt"].append(padf(pt))
            out["eta"].append(padf(eta))
            out["phi"].append(padf(phi))
            out["m"].append(padf(m))
            out["mask"].append(padf(mask).astype(bool))
            out["target"].append(target)
            out["pmass"].append(pm.astype(np.float32))
            out["w"].append(w.astype(np.float64))
            out["njall"].append(njall)
    res = {k_: np.concatenate(v) for k_, v in out.items()}
    if max_events:
        res = {k_: v[:max_events] for k_, v in res.items()}
    res["is_signal"] = np.full(len(res["pt"]), is_signal)
    return res


def concat(*ds):
    return {k: np.concatenate([d[k] for d in ds]) for k in ds[0]}


def to_tensors(d):
    t = {}
    for k, v in d.items():
        if v.dtype == np.float64 and k != "w":
            v = v.astype(np.float32)
        t[k] = torch.from_numpy(np.ascontiguousarray(v))
    t["w"] = t["w"].float()
    return t


def split(d, fracs=(0.7, 0.15, 0.15), seed=0):
    n = len(d["pt"])
    idx = np.random.default_rng(seed).permutation(n)
    a = int(fracs[0] * n)
    b = int((fracs[0] + fracs[1]) * n)
    pick = lambda ii: {k: v[ii] for k, v in d.items()}
    return pick(idx[:a]), pick(idx[a:b]), pick(idx[b:])
