"""
Loading jets + per-jet ancestry labels for JetAssign.

Signal files written with generate_pythia.py --ancestry carry INPUTS/Source/anc (N, J, 3): the parton
pT fractions of each stored jet from parent 1 / parent 2 / neither.  Labels are put in canonical order
(parent 1 owns the leading jet that is mostly from a parent) so that they are a function of the jets.
QCD and unlabelled files get all-zero labels.
"""

import glob

import h5py
import numpy as np


def canonical_labels(anc, mask):
    """anc (N, J, 3) in pT order -> labels with parent 1 = parent of the leading parent-dominated jet.
    Real jets with no parton information are 'other'."""
    lab = anc.astype(np.float32).copy()
    s = lab.sum(-1)
    lab[(s < 1e-6) & mask, 2] = 1.0
    lab[~mask] = 0.0
    dom = lab.argmax(-1)
    from_parent = (dom < 2) & mask
    has = from_parent.any(1)
    first = np.where(has, from_parent.argmax(1), 0)
    swap = has & (dom[np.arange(len(dom)), first] == 1)
    lab[swap] = lab[swap][:, :, [1, 0, 2]]
    return lab


def load(paths, n_jets=12, is_signal=True, min_jets=6, max_events=None, seed=12345):
    files = []
    for p in paths if isinstance(paths, (list, tuple)) else [paths]:
        files += sorted(glob.glob(p)) or [p]
    out = {k: [] for k in ("pt", "eta", "phi", "m", "mask", "labels", "pmass", "w", "njall")}
    for fp in files:
        with h5py.File(fp, "r") as f:
            src = f["INPUTS/Source"]
            pt, eta, phi, m = (src[x][:] for x in ("pt", "eta", "phi", "mass"))
            mask = src["MASK"][:].astype(bool)
            anc = src["anc"][:] if "anc" in src else np.zeros(pt.shape + (3,), np.float32)
            n = len(pt)
            keep = np.ones(n, bool)
            if "EventVars/pass_sel" in f:
                keep &= f["EventVars/pass_sel"][:].astype(bool)
            order = np.argsort(np.where(mask, -pt, np.inf), axis=1, kind="stable")
            take = lambda a: np.take_along_axis(a, order if a.ndim == 2 else order[..., None], 1)
            pt, eta, phi, m, mask, anc = map(take, (pt, eta, phi, m, mask, anc))
            njall = mask.sum(1)
            lab = canonical_labels(anc, mask) if is_signal else np.zeros_like(anc, dtype=np.float32)
            if "EventVars/parent_mass" in f:
                pm = f["EventVars/parent_mass"][:]
            else:
                pm = np.full(n, float(f.attrs.get("mass", 0.0)) if is_signal else 0.0)
            w = f["EventVars/normweight"][:] if "EventVars/normweight" in f else np.ones(n)
            pad = max(0, n_jets - pt.shape[1])
            cut = lambda a: np.pad(a[:, :n_jets], ((0, 0), (0, pad)) + ((0, 0),) * (a.ndim - 2))[keep]
            for k, v in (("pt", pt), ("eta", eta), ("phi", phi), ("m", m), ("mask", mask), ("labels", lab)):
                out[k].append(cut(v))
            out["pmass"].append(pm.astype(np.float32)[keep])
            out["w"].append(w.astype(np.float64)[keep])
            out["njall"].append(njall[keep])
    res = {k: np.concatenate(v) for k, v in out.items()}
    res["mask"] = res["mask"].astype(bool)
    sel = res["njall"] >= min_jets
    res = {k: v[sel] for k, v in res.items()}
    if max_events and max_events < len(res["pt"]):
        pick = np.sort(np.random.default_rng(seed).choice(len(res["pt"]), max_events, replace=False))
        res = {k: v[pick] for k, v in res.items()}
    res["is_signal"] = np.full(len(res["pt"]), is_signal)
    return res
