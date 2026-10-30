"""Quick test-set check: argmin and PairNet(+P) with an HT > c*M cut tuned on one half, evaluated on the other."""
import glob, json, sys
import h5py
import numpy as np
sys.path.insert(0, "/tmp/claude-12421/-scratch-llee/a35d08e7-9d87-5923-8644-0d53c5b0f767/scratchpad")
from limits import s_up
v = np.load("/scratch/llee/pairnet/cuts_study/vars_r8B.npz"); ht = np.load("/scratch/llee/pairnet/run3/ht_test_r8B.npy")
s = v["is_signal"]; q = ~s; w = v["w"].astype(float); pm = v["pmass"]; half = np.arange(len(s)) % 2
qf = sorted(glob.glob("/scratch/llee/pairnet/data_r3/prod/qcd_c*9.h5"))
kq = 2 * np.mean([h5py.File(f).attrs["sigma_mb"] / h5py.File(f).attrs["weight_sum"] for f in qf]) * 1e12 / len(qf)
filt = {int(k): x for k, x in json.load(open("/scratch/llee/pairnet/xs_study/xs_study.json"))["filter_eff"].items()}
L = 300.0
C = [0, .8, 1.0, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 2.0]
out = {}
for M in range(500, 2401, 100):
    sg = s & (pm == M)
    r = {}
    for p in ("am", "pn"):
        base = (v[f"{p}_asym"] < 0.1) & (v[f"{p}_mavg"] > 0.85 * M) & (v[f"{p}_mavg"] < 1.1 * M)
        pq = [-1.0] + list(np.quantile(v["psig"][sg & base], 1 - np.linspace(0.96, 0.2, 20))) if p == "pn" else [-1.0]

        def lim(c, t, h):
            sel = base & (ht > c * M) & (v["psig"] >= t)
            m = half == h
            eS = (sel & sg & m).sum() / (sg & m).sum() * filt[M]
            nB = int((sel & q & m).sum()); B = kq * w[sel & q & m].sum() * L
            return (s_up(B) / (L * eS) if eS > 0 else np.inf), nB
        for tag, cs in (("", [0]), ("+HT", C)):
            ls = []
            for h in (0, 1):
                best = min(((lim(c, t, h)[0], c, t) for c in cs for t in pq if lim(c, t, h)[1] >= 10), default=(np.inf, 0, -1))
                ls.append(lim(best[1], best[2], 1 - h)[0])
            r[p + tag] = float(np.exp(np.mean(np.log(ls))))
    out[M] = r
    print(M, {k: round(x, 2) for k, x in r.items()}, "argmin gain from HT %.1fx, PairNet %.2fx" % (r["am"] / r["am+HT"], r["pn"] / r["pn+HT"]), flush=True)
json.dump(out, open("/scratch/llee/pairnet/run3/quick_ht_gain.json", "w"), indent=1)
