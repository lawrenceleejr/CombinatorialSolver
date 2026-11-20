"""Reduced (local) event set: prescale by the nominal P so the CPU goes to events that can matter. Weights carry the prescale.
usage: prepL.py T | S"""
import glob, sys, os, numpy as np
D = "/scratch/llee/pairnet/audit/F3_robust"; rng = np.random.default_rng(5); CH = 2500
K = ("pt", "eta", "phi", "m", "mask", "w", "row", "fidx")


def build(name, files, pfun, tiers, extra=()):
    parts = []
    for f in files:
        c = np.load(f); p = pfun(f, c); ps = np.ones(len(p))
        for lo, hi, k in tiers:
            ps[(p >= lo) & (p < hi)] = k
        keep = rng.random(len(p)) < 1.0 / ps
        d = {k: c[k][keep] for k in K + tuple(extra) if k in c}; d["w"] = d["w"] * ps[keep]; d["pnom"] = p[keep].astype(np.float32); d["presc"] = ps[keep].astype(np.float32)
        parts.append(d)
    d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    n = len(d["w"]); nall = len(glob.glob(f"{D}/cache/chunks/{name}_[0-9]*.npz"))
    d["w"] = d["w"] * nall / len(files)            # in case not all chunks were available
    o = np.argsort(-d["pnom"], kind="stable")       # highest P first: partial results are still usable tier by tier
    perm = rng.permutation(n); d = {k: v[perm] for k, v in d.items()}
    nch = int(np.ceil(n / CH))
    for i in range(nch):
        np.savez(f"{D}/cache/chunksL/{name}_{i}.npz", **{k: v[i * CH:(i + 1) * CH] for k, v in d.items()})
    print(name, "events", n, "chunks", nch, {t: int((d['presc'] == t[2]).sum()) for t in tiers}, "sum w", d["w"].sum(), flush=True)
    return [f"{name}_{i}" for i in range(nch)]


names = []
if sys.argv[1] == "T":
    p0 = lambda f, c: c["p0"]
    names += build("TvlmT", sorted(glob.glob(f"{D}/cache/chunks/TvlmT_*.npz")), p0, [(0.99, 2, 1), (0.97, 0.99, 8), (0.0, 0.97, 60)], ("p0",))
    names += build("ThtT", sorted(glob.glob(f"{D}/cache/chunks/ThtT_*.npz")), p0, [(0.995, 2, 1), (0.98, 0.995, 5), (0.0, 0.98, 50)], ("p0",))
elif sys.argv[1] == "U":
    def pn(f, c):
        z = np.load(f"{D}/out/nomS/" + os.path.basename(f)); return np.where(z["v0_valid"], z["v0_psig"], 0.0)
    names += build("Uvlm", sorted(glob.glob(f"{D}/cache/chunks/Uvlm_*.npz")), pn, [(0.97, 2, 1), (0.9, 0.97, 10), (-2, 0.9, 200)])
    names += build("Uht", sorted(glob.glob(f"{D}/cache/chunks/Uht_*.npz")), pn, [(0.98, 2, 1), (0.9, 0.98, 10), (-2, 0.9, 200)])
else:
    def pn(f, c):
        z = np.load(f"{D}/out/nomS/" + os.path.basename(f)); return np.where(z["v0_valid"], z["v0_psig"], 0.0)
    for g, tiers in (("sig700", [(0.98, 2, 1), (0.9, 0.98, 5), (-2, 0.9, 40)]), ("sig400", [(0.98, 2, 1), (0.9, 0.98, 5), (-2, 0.9, 40)]), ("sig1000", [(0.98, 2, 1), (0.9, 0.98, 5), (-2, 0.9, 40)]),
                     ("sig1500", [(0.98, 2, 1), (0.9, 0.98, 5), (-2, 0.9, 40)]), ("sig2000", [(0.98, 2, 1), (0.9, 0.98, 5), (-2, 0.9, 40)])):
        fs = [f for f in sorted(glob.glob(f"{D}/cache/chunks/{g}_*.npz")) if os.path.exists(f"{D}/out/nomS/" + os.path.basename(f))]
        names += build(g, fs, pn, tiers, ("boost",))
open(f"{D}/chunksL_{sys.argv[1]}.txt", "w").write("\n".join(names) + "\n")
