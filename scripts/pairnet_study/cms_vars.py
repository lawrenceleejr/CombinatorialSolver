"""Per-event CMS-style trijet variables on the r8B test set, for the PairNet pairing and the argmin |m1-m2| pairing.

Definitions follow CMS-EXO-17-030 (arXiv:1810.10092), used again in CMS-EXO-21-004:
  mhat(3,2)^2_ij  = m_ij^2 / (m_ijk^2 + m_i^2 + m_j^2 + m_k^2)          D2_32 = sum_{i<j} (mhat(3,2)_ij - 1/sqrt3)^2
  mhat(6,3)^2_ijk = m_ijk^2 / (4 m_6j^2 + 6 sum_i m_i^2)                D2_63_32 = sum_{20 triplets} (sqrt(mhat(6,3)^2 + D2_32) - 1/sqrt20)^2
  Delta = (scalar pT sum of a triplet) - m_jjj  (CMS cut: m_jjj < sum|pT| - Delta_cut)
Per event and pairing: D2_32 max over the two chosen triplets, Delta min over them, D2_63_32 over the six chosen jets,
lowest pT among the six chosen jets, and the pT of the 6th leading jet. Row order matches the run's eval.npz.
"""
import glob, sys
from itertools import combinations
import numpy as np
import torch

sys.path.insert(0, "/scratch/llee/CombinatorialSolver")
from src.pairdata import load_events  # noqa: E402
from src.pairnet import enumerate_interpretations, to_cartesian  # noqa: E402

npz, out = sys.argv[1:3]
P = "/scratch/llee/pairnet/data_r3/prod"
e = np.load(npz)
d = [load_events(sorted(glob.glob(f"{P}/sig_*_c*9.h5")), 12, 3, True, None, min_jets=6, truth="none", n_assign=7),
     load_events(sorted(glob.glob(f"{P}/qcd_c*9.h5")), 12, 3, False, None, min_jets=6, truth="none", n_assign=7)]
pt, eta, phi, m, mask = (np.concatenate([x[k] for x in d])[:, :7] for k in ("pt", "eta", "phi", "m", "mask"))
is_sig = np.concatenate([x["is_signal"] for x in d]); pmass = np.concatenate([x["pmass"] for x in d])
assert len(pt) == len(e["best"]) and (is_sig == e["is_signal"]).all() and np.allclose(pmass, e["pmass"]), "row order mismatch"
p4 = (to_cartesian(*(torch.from_numpy(v).double() for v in (pt, eta, phi, m))) * torch.from_numpy(mask).unsqueeze(-1)).numpy()

interp = enumerate_interpretations(7)
G = np.array([list(a) + list(b) for a, b in interp])  # (70, 6): triplet 1 then triplet 2
mm = lambda v: np.sqrt(np.maximum(v[..., 0] ** 2 - (v[..., 1:] ** 2).sum(-1), 0))

# argmin |m1 - m2| over valid interpretations (all six jets real)
m1 = mm(p4[:, G[:, :3]].sum(2)); m2 = mm(p4[:, G[:, 3:]].sum(2))
ok = mask[:, G].all(-1)
amin = np.where(ok, np.abs(m1 - m2), np.inf).argmin(1)

T3 = list(combinations(range(6), 3))  # 20 triplets of six jets
PR = [(0, 1), (0, 2), (1, 2)]


def d2_32(t4):
    """t4: (..., 3, 4) jet four-vectors of a triplet -> D2_[3,2]."""
    mi2 = mm(t4) ** 2
    mt2 = mm(t4.sum(-2)) ** 2
    den = mt2 + mi2.sum(-1)
    mh = np.stack([np.sqrt(mm(t4[..., i, :] + t4[..., j, :]) ** 2 / np.maximum(den, 1e-12)) for i, j in PR], -1)
    return ((mh - 1 / np.sqrt(3)) ** 2).sum(-1)


def variables(best):
    ar = np.arange(len(best))
    j6 = G[best]  # (N, 6) jet indices
    v = p4[ar[:, None], j6]  # (N, 6, 4)
    ptc = pt[ar[:, None], j6]
    ta, tb = v[:, :3], v[:, 3:]
    ma, mb = mm(ta.sum(1)), mm(tb.sum(1))
    da = ptc[:, :3].sum(1) - ma; db = ptc[:, 3:].sum(1) - mb
    d32 = np.maximum(d2_32(ta), d2_32(tb))
    mi2 = (mm(v) ** 2).sum(1)
    m6 = mm(v.sum(1))
    den63 = 4 * m6 ** 2 + 6 * mi2
    tr = v[:, np.array(T3)]  # (N, 20, 3, 4)
    mh63_2 = mm(tr.sum(2)) ** 2 / np.maximum(den63, 1e-12)[:, None]
    d63 = ((np.sqrt(mh63_2 + d2_32(tr)) - 1 / np.sqrt(20)) ** 2).sum(1)
    return {"m1": ma, "m2": mb, "delta_min": np.minimum(da, db), "d2_32_max": d32, "d2_63_32": d63, "pt_low6": ptc.min(1)}


res = {"pt6_lead": pt[:, 5], "is_signal": is_sig, "pmass": pmass, "w": e["w"], "psig": e["psig"]}
for name, best in (("pn", e["best"]), ("am", amin)):
    vv = variables(best)
    for k, x in vv.items():
        res[f"{name}_{k}"] = x.astype(np.float32)
    res[f"{name}_mavg"] = (0.5 * (vv["m1"] + vv["m2"])).astype(np.float32)
    res[f"{name}_asym"] = (np.abs(vv["m1"] - vv["m2"]) / (vv["m1"] + vv["m2"])).astype(np.float32)
res["am_valid"] = ok[np.arange(len(amin)), amin]
print("PairNet mavg check: max |rel diff| vs eval.npz = %.2e" % np.max(np.abs(res["pn_mavg"] - e["mavg"]) / np.maximum(e["mavg"], 1)))
print("PairNet asym check: max |diff| = %.2e" % np.max(np.abs(res["pn_asym"] - e["asym"])))
np.savez_compressed(out, **res)
