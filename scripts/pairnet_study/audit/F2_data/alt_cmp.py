"""F2 item 4c: alternative shower/tune samples (local or OSPool, C700 generator filter: pTHat>120, bias 4, H_T>900, sym 450,1000; scored unfiltered)
vs the production qcd700T (same generator settings, default Monash simple shower).  usage: alt_cmp.py name 'glob of h5 or npz' [M=700]"""
import sys, glob, numpy as np, h5py, subprocess, os
name, pat = sys.argv[1], sys.argv[2]; M = float(sys.argv[3]) if len(sys.argv) > 3 else 700.
o = {k: [] for k in ("ht", "p", "m", "A", "v", "w")}; fs = sorted(glob.glob(pat)); sig = []
for f in fs:
    a = np.load(f); k = float(a["attr_sigma_mb"]) * 1e12 / float(a["attr_weight_sum"]); m1, m2 = a["r14A_m1"], a["r14A_m2"]; sig.append((float(a["attr_sigma_mb"]), float(a["attr_n_tried"]), int(a["n_all"])))
    o["ht"].append(a["ht"]); o["p"].append(a["r14A_psig"]); o["m"].append(0.5 * (m1 + m2)); o["A"].append(np.abs(m1 - m2) / np.maximum(m1 + m2, 1e-9)); o["v"].append(a["r14A_valid"] > 0); o["w"].append(a["w"] * k)
o = {k: np.concatenate(v) for k, v in o.items()}; o["w"] /= len(fs)
print(name, len(fs), "files; sigma_gen[mb], tried, written:", sig[:3])
d = np.load("cache/q700T.npz"); m1, m2 = d["r14A_m1"], d["r14A_m2"]
b = dict(ht=d["ht"], p=d["r14A_psig"], m=0.5 * (m1 + m2), A=np.abs(m1 - m2) / np.maximum(m1 + m2, 1e-9), v=d["r14A_valid"] > 0, w=d["wfb"])
def rate(c, p, ht):
    s = c["v"] & (c["A"] < 0.1) & (c["m"] > 0.85 * M) & (c["m"] < 1.1 * M) & (c["ht"] > ht) & (c["p"] > p); w = c["w"][s]; return w.sum(), np.sqrt((w ** 2).sum()), int(s.sum())
for ht in (900, 1200):
    for p in (0.9, 0.95, 0.99, 0.9966):     # production sample only exists above its tail filter P>0.9
        ra, ea, na = rate(o, p, ht); rb, eb, nb = rate(b, p, ht)
        if na < 3: print(f"  H_T>{ht} P>{p}: {name} n={na} ({ra:.2f} fb), default {rb:.2f}+-{eb:.2f} fb"); continue
        x = ra / rb; e = x * np.hypot(ea / ra, eb / rb); print(f"  H_T>{ht} P>{p}: {name} {ra:9.2f}+-{ea:7.2f} fb (n={na})  default {rb:9.2f}+-{eb:6.2f} fb  ratio {x:.2f} +- {e:.2f}")
ra, ea, na = rate(o, -1, 900); print(f"  no P cut, H_T>900: {name} {ra:.0f}+-{ea:.0f} fb (n={na})")
