"""Corrected argmin|m1-m2| metrics on the 10M-QCD test files (pairings must use real jets only):
overall pairing efficiency, A<0.1 pass rates, signal in window, and per-mass S/sqrt(B) with A<0.1 + window."""
import glob, json, math, sys
import numpy as np, torch
sys.path.insert(0, "/scratch/llee/CombinatorialSolver")
from src.pairdata import load_events, concat
from src.pairnet import to_cartesian, enumerate_interpretations
from src.combinatorics import enumerate_assignments
P = "/scratch/llee/pairnet/data_r3/prod"; out = sys.argv[1]
d = concat(load_events(sorted(glob.glob(f"{P}/sig_*_c*9.h5")), 7, 3, True, None, min_jets=6),
           load_events(sorted(glob.glob(f"{P}/qcd_c*9.h5")), 7, 3, False, None, min_jets=6))
p4 = (to_cartesian(*(torch.from_numpy(d[k]).float() for k in ("pt", "eta", "phi", "m"))) * torch.from_numpy(d["mask"]).unsqueeze(-1)).numpy().astype(np.float64)
asg = enumerate_assignments(7); g1 = np.array([a[1] for a in asg]); g2 = np.array([a[2] for a in asg])
interp = enumerate_interpretations(7, ((3, 3),)); key = lambda a, b: frozenset([frozenset(a), frozenset(b)])
pos = {key(a, b): j for j, (a, b) in enumerate(interp)}; L2P = np.array([pos[key(a[1], a[2])] for a in asg])
mm = lambda v: np.sqrt(np.maximum(v[..., 0] ** 2 - (v[..., 1:] ** 2).sum(-1), 0))
m1 = mm(p4[:, g1].sum(2)); m2 = mm(p4[:, g2].sum(2)); ok = d["mask"][:, g1].all(-1) & d["mask"][:, g2].all(-1)
b = np.where(ok, np.abs(m1 - m2), np.inf).argmin(1); ar = np.arange(len(b))
A = np.abs(m1[ar, b] - m2[ar, b]) / (m1[ar, b] + m2[ar, b]); mavg = 0.5 * (m1[ar, b] + m2[ar, b])
s = d["is_signal"]; q = ~s; w = d["w"]; W = w[q].sum(); tgt = d["target"]; pm = d["pmass"]; rec = s & (tgt >= 0)
rel = mavg / np.maximum(pm, 1); win = (rel > 0.85) & (rel < 1.1); pas = A < 0.1
r = {"eff_on_reconstructable": float((L2P[b] == tgt)[rec].mean()), "sig_pass_asym": float(pas[s].mean()),
     "bkg_pass_asym": float((w * pas)[q].sum() / W), "sig_pass_asym_in_window": float((pas & win)[s].mean())}
r["ratio"] = r["sig_pass_asym_in_window"] / math.sqrt(r["bkg_pass_asym"])
lo, hi = np.quantile(rel[s], [0.16, 0.84]); r["mavg_rel_halfwidth68"] = float((hi - lo) / (hi + lo))
mass = {}
for M in range(500, 2401, 100):
    sm = s & (pm == M); wm = (mavg > 0.85 * M) & (mavg < 1.1 * M)
    S = (pas & wm)[sm].mean(); B = (w * (pas & wm))[q].sum() / W
    mass[M] = {"S": float(S), "B": float(B), "fom": float(S / math.sqrt(B)), "n_qcd": int((pas & wm & q).sum())}
json.dump({"classical": r, "classical_mass": mass}, open(out, "w"), indent=1)
print({k: round(v, 4) for k, v in r.items()})
print({M: round(v["fom"], 2) for M, v in mass.items() if M in (500, 800, 1000, 1500, 2000, 2400)})
