"""Concatenate scored npz of one family into a compact cache (per-event fb weights; file-level normalisation).
usage: cache_fam.py name 'glob' [model=r14A]"""
import sys, glob, numpy as np
name, pat = sys.argv[1], sys.argv[2]; M = sys.argv[3] if len(sys.argv) > 3 else "r14A"
fs = sorted(glob.glob(pat)); cols = ["ht", "pt6_lead", f"{M}_psig", f"{M}_m1", f"{M}_m2", f"{M}_delta1", f"{M}_delta2", f"{M}_valid", "am_m1", "am_m2", "am_valid", "pmass"]
opt = ["pthat", "sym_pass"]
out = {c: [] for c in cols + opt + ["wfb", "fid", "w"]}; nf = 0; ntried = 0; nall = 0; sig = []; bad = 0
for i, f in enumerate(fs):
    try: a = np.load(f); w = a["w"]
    except Exception: bad += 1; continue
    nf += 1; ntried += float(a["attr_n_tried"]); nall += int(a["n_all"]) if "n_all" in a.files else len(w)
    k = float(a["attr_sigma_mb"]) * 1e12 / float(a["attr_weight_sum"]); sig.append(float(a["attr_sigma_mb"]))
    if len(w) == 0: continue
    for c in cols: out[c].append(a[c])
    for c in opt:
        if c in a.files: out[c].append(a[c])
    out["wfb"].append(w * k); out["w"].append(w); out["fid"].append(np.full(len(w), i, np.int32))
res = {c: np.concatenate(v) for c, v in out.items() if v}
res["wfb"] = res["wfb"] / nf
res.update(nfiles=nf, ntried=ntried, nall=nall, sigma_mb=np.mean(sig), bad=bad)
np.savez(f"cache/{name}.npz", **res); print(name, nf, "files", len(res["wfb"]), "events", "bad", bad, flush=True)
