"""Attach F1's nominal r14A psig (audit/F1_stats/leak) to my chunks by matching H_T within each file. Writes out/nomS/{chunk}.npz with v0_psig, v0_valid."""
import glob, os, numpy as np
D = "/scratch/llee/pairnet/audit/F3_robust"; F = "/scratch/llee/pairnet/audit/F1_stats/leak"
os.makedirs(f"{D}/out/nomS", exist_ok=True)
SRC = {"Uvlm": lambda i, b: f"qcdvlm_c{i}", "Uht": lambda i, b: f"qcdht_c{i}"}
for M in (400, 700, 1000):
    SRC[f"sig{M}"] = (lambda M: lambda i, b: f"siglm_m{M}_{'boost' if b else 'incl'}_c{i}")(M)
for M in (1500, 2000):
    SRC[f"sig{M}"] = (lambda M: lambda i, b: f"sig_m{M}_c{i}")(M)
cache = {}
for g, fn in SRC.items():
    nmiss = ntot = 0
    for f in sorted(glob.glob(f"{D}/cache/chunks/{g}_[0-9]*.npz")):
        c = np.load(f); ht = (c["pt"] * c["mask"]).sum(1); b = c["boost"] if "boost" in c else np.zeros(len(ht), bool)
        ps = np.full(len(ht), -1.0); va = np.zeros(len(ht), bool)
        for i, bb, j in zip(c["fidx"], b, range(len(ht))):
            k = fn(int(i), bool(bb))
            if k not in cache:
                z = np.load(f"{F}/{k}.npz") if os.path.exists(f"{F}/{k}.npz") else None
                cache[k] = None if z is None else (z["ht"], z["r14A_psig"], z["r14A_valid"])
            z = cache[k]
            if z is None:
                continue
            d = np.abs(z[0] - ht[j]); a = d.argmin()
            if d[a] < 1e-3 * ht[j]:
                ps[j], va[j] = z[1][a], z[2][a]
        nmiss += (ps < 0).sum(); ntot += len(ht)
        np.savez(f"{D}/out/nomS/" + os.path.basename(f), v0_psig=ps, v0_valid=va)
    print(g, "events", ntot, "unmatched (outside F1 window filter or <6 jets)", nmiss, flush=True)
