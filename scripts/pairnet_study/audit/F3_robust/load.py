"""Merge the scan outputs of one group into a compact cache: out/merged/{group}.npz"""
import sys, glob, os, numpy as np
K = ("psig", "mest", "valid", "ht", "nj", "mavg", "asym", "dmax", "dmin", "d32", "d63", "ptlow")
D = "/scratch/llee/pairnet/audit/F3_robust"
SD, CD, PAT = (sys.argv.pop(1), sys.argv.pop(1), sys.argv.pop(1)) if len(sys.argv) > 1 and sys.argv[1] == "-L" and sys.argv.pop(1) else ("out/scan", "cache/chunks", "_scan")
MD = os.environ.get("F3_MERGED", "merged"); os.makedirs(f"{D}/out/{MD}", exist_ok=True)
for grp in sys.argv[1:]:
    fs = sorted(glob.glob(f"{D}/{SD}/{grp}_[0-9]*{PAT}.npz"), key=lambda f: int(os.path.basename(f).replace(PAT + ".npz", "").split("_")[-1]))
    nall = len(glob.glob(f"{D}/{CD}/{grp}_[0-9]*.npz"))
    out = {k: [] for k in K}; meta = {k: [] for k in ("w", "fidx", "row", "p0", "boost", "pnom", "presc")}; gap = []
    for f in fs:
        z = np.load(f); c = np.load(f"{D}/{CD}/" + (os.path.basename(f).replace(PAT, "") if PAT else os.path.basename(f)))
        V = len(z["variants"]); assert int(z["n"]) == len(c["w"])
        for k in K:
            a = np.stack([z[f"v{i}_{k}"] for i in range(V)], 1)
            out[k].append(a.astype(np.float32) if a.dtype == np.float64 and k != "psig" else a)
        gap.append(z["v0_gap"])
        for k in meta:
            if k in c:
                meta[k].append(c[k])
    r = {k: np.concatenate(v) for k, v in out.items()}; r.update({k: np.concatenate(v) for k, v in meta.items() if v})
    r["gap"] = np.concatenate(gap); r["variants"] = z["variants"]; r["frac"] = len(fs) / nall
    for k in ("nfiles", "ntried_incl", "ntried_boost"):
        if k in c:
            r[k] = c[k]
    np.savez(f"{D}/out/{MD}/{grp}.npz", **r)
    print(grp, "chunks", len(fs), "/", nall, "events", len(r["w"]), flush=True)
