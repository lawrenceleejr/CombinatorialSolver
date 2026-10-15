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

from .pairnet import enumerate_interpretations


def _lookup(n_jets, sizes):
    return {(a, b): i for i, (a, b) in enumerate(enumerate_interpretations(n_jets, sizes))}


def canonical(a, b):
    a, b = tuple(sorted(a)), tuple(sorted(b))
    if len(a) != len(b):
        return (a, b) if len(a) > len(b) else (b, a)
    return (min(a, b), max(a, b))


def truth_sets_from_quarks(quarks, eta, phi, mask, dr_max=0.4):
    """Each decay quark -> nearest jet within dr_max (several quarks may share a jet).

    Returns (N, 6) jet indices in the given (sorted) jet order, -1 if unmatched.
    """
    qeta, qphi = quarks[..., 1][:, :, None], quarks[..., 2][:, :, None]
    dphi = (qphi - phi[:, None] + np.pi) % (2 * np.pi) - np.pi
    dr = np.sqrt((qeta - eta[:, None]) ** 2 + dphi**2)
    dr = np.where(mask[:, None], dr, np.inf)
    j = dr.argmin(-1)
    return np.where(np.take_along_axis(dr, j[..., None], -1)[..., 0] < dr_max, j, -1)


def _proxy_targets(tq, pt, eta, phi, m, mask, pmass, n_jets, sizes):
    """'Most signal-like' interpretation when the exact one is unavailable.

    Score = (#jets grouped consistently with their true parent) first, then
    closeness of m1, m2 to the true parent mass.  Returned encoded as -2 - idx.
    Events with no usable interpretation get -1.
    """
    import torch

    from .pairnet import minkowski_m2, to_cartesian

    interp = enumerate_interpretations(n_jets, sizes)
    A = len(interp)
    M1 = np.zeros((A, n_jets))
    M2 = np.zeros((A, n_jets))
    for i, (a, b) in enumerate(interp):
        M1[i, list(a)] = 1
        M2[i, list(b)] = 1
    nev = len(tq)
    if nev == 0:
        return np.zeros(0, dtype=np.int64)
    # parent membership of each jet: bit 1 = parent 1, bit 2 = parent 2
    par = np.zeros((nev, n_jets), dtype=np.int64)
    for q in range(6):
        j = tq[:, q]
        ok = (j >= 0) & (j < n_jets)
        par[np.where(ok)[0], j[ok]] |= 1 if q < 3 else 2
    p1 = (par & 1 > 0).astype(float)
    p2 = (par & 2 > 0).astype(float)
    ov = np.maximum(p1 @ M1.T + p2 @ M2.T, p2 @ M1.T + p1 @ M2.T)  # (nev, A)
    p4 = to_cartesian(*(torch.from_numpy(x).double() for x in (pt, eta, phi, m)))
    p4 = (p4 * torch.from_numpy(mask).unsqueeze(-1)).numpy()
    P1 = np.einsum("an,enc->eac", M1, p4)
    P2 = np.einsum("an,enc->eac", M2, p4)
    m1 = np.sqrt(np.maximum(minkowski_m2(torch.from_numpy(P1)).numpy(), 0))
    m2 = np.sqrt(np.maximum(minkowski_m2(torch.from_numpy(P2)).numpy(), 0))
    dist = (np.abs(m1 - pmass[:, None]) + np.abs(m2 - pmass[:, None])) / np.maximum(pmass[:, None], 1)
    valid = (mask.astype(float) @ (M1 + M2).T) >= (M1 + M2).sum(1)[None] - 0.5
    score = np.where(valid, ov * 100.0 - np.minimum(dist, 50.0), -np.inf)
    best = score.argmax(1)
    good = np.isfinite(score.max(1))
    return np.where(good, -2 - best, -1)


def load_events(paths, n_jets=8, k=3, is_signal=True, max_events=None, sizes=None,
                truth="auto", min_jets=None, max_jets=None, proxy=False, n_assign=None):
    """truth: "quarks" (nearest-jet, merging allowed; needs truth/quarks),
    "targets" (TARGETS block, unique matching), or "auto" (quarks if present)."""
    sizes = tuple(tuple(x) for x in (sizes or ((k, k),)))
    allowed = {tuple(sorted(x, reverse=True)) for x in sizes}
    files = []
    for p in paths if isinstance(paths, (list, tuple)) else [paths]:
        files += sorted(glob.glob(p)) or [p]
    na = n_assign or n_jets  # truth / interpretations use the leading na jets
    lut = _lookup(na, sizes)
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
            use_q = is_signal and "truth/quarks" in f and truth in ("auto", "quarks")
            if use_q:
                tq = truth_sets_from_quarks(f["truth/quarks"][:], eta, phi, mask)
                for i in range(n):
                    t = tq[i]
                    if (t < 0).any() or (t >= na).any():
                        continue
                    a_, b_ = set(t[:3].tolist()), set(t[3:].tolist())
                    if a_ & b_:
                        continue  # quarks from different parents share a jet
                    ca = canonical(a_, b_)
                    if (len(ca[0]), len(ca[1])) not in allowed:
                        continue
                    target[i] = lut[ca]
                if proxy:
                    pmass_f = (f["EventVars/parent_mass"][:] if "EventVars/parent_mass" in f
                               else np.full(n, float(f.attrs.get("mass", 0.0))))
                    need = target < 0
                    pt_ = target.copy()
                    pt_[need] = _proxy_targets(tq[need], pt[need, :na], eta[need, :na],
                                               phi[need, :na], m[need, :na],
                                               mask[need, :na], pmass_f[need], na, sizes)
                    target = pt_
            elif is_signal and "TARGETS" in f:
                tg = np.stack(
                    [f[f"TARGETS/g{g}/j{j}"][:] for g in (1, 2) for j in (1, 2, 3)], 1
                ).astype(np.int64)
                kk = tg.shape[1] // 2
                for i in range(n):
                    t = tg[i]
                    if (t < 0).any() or not mask[i, t].all():
                        continue
                    s_ = inv[i, t]
                    if (s_ >= na).any() or len(set(s_.tolist())) != 2 * kk:
                        continue
                    ca = canonical(s_[:kk].tolist(), s_[kk:].tolist())
                    if ca in lut:
                        target[i] = lut[ca]
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
    keep = np.ones(len(res["pt"]), dtype=bool)
    if min_jets is not None:
        keep &= res["njall"] >= min_jets
    if max_jets is not None:
        keep &= res["njall"] <= max_jets
    res = {k_: v[keep] for k_, v in res.items()}
    if max_events and max_events < len(res["pt"]):
        pick = np.sort(np.random.default_rng(12345).choice(len(res["pt"]), max_events, replace=False))
        res = {k_: v[pick] for k_, v in res.items()}
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
