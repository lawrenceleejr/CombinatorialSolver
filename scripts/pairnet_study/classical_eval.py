"""Per-event argmin|m1-m2| pairing (leading 7 jets) on the 10M-QCD test files, saved in eval.npz form (no psig)."""
import glob, sys
import numpy as np, torch
sys.path.insert(0, "/scratch/llee/CombinatorialSolver")
from src.pairdata import load_events, concat
from src.pairnet import to_cartesian
from src.combinatorics import enumerate_assignments
P = "/scratch/llee/pairnet/data_r3/prod"; out = sys.argv[1]
d = concat(load_events(sorted(glob.glob(f"{P}/sig_*_c*9.h5")), 7, 3, True, None, min_jets=6),
           load_events(sorted(glob.glob(f"{P}/qcd_c*9.h5")), 7, 3, False, None, min_jets=6))
p4 = (to_cartesian(*(torch.from_numpy(d[k]).float() for k in ("pt", "eta", "phi", "m"))) * torch.from_numpy(d["mask"]).unsqueeze(-1)).numpy().astype(np.float64)
asg = enumerate_assignments(7); g1 = np.array([a[1] for a in asg]); g2 = np.array([a[2] for a in asg])
mm = lambda v: np.sqrt(np.maximum(v[..., 0] ** 2 - (v[..., 1:] ** 2).sum(-1), 0))
m1 = mm(p4[:, g1].sum(2)); m2 = mm(p4[:, g2].sum(2)); ok = d["mask"][:, g1].all(-1) & d["mask"][:, g2].all(-1)
b = np.where(ok, np.abs(m1 - m2), np.inf).argmin(1); ar = np.arange(len(b))
np.savez_compressed(out, mavg=0.5 * (m1[ar, b] + m2[ar, b]), asym=np.abs(m1[ar, b] - m2[ar, b]) / (m1[ar, b] + m2[ar, b]),
                    is_signal=d["is_signal"], pmass=d["pmass"], w=d["w"])
print("events", len(b))
