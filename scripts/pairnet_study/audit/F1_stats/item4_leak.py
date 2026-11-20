"""Item 4: P(signal) tail efficiency of r14A on its training files vs validation / test / never-seen files (same sample type)."""
import glob, json, re, sys
import numpy as np
D = "/scratch/llee/pairnet/audit/F1_stats"
K = ["ht", "w", "r14A_psig", "r14A_m1", "r14A_m2", "r14A_valid"]


def load(pattern, split):
    parts = {}
    for f in glob.glob(f"{D}/leak/{pattern}"):
        c = int(re.search(r"_c(\d+)\.npz$", f).group(1)); s = split(c)
        if s is None:
            continue
        a = np.load(f)
        if len(a["w"]) == 0:
            continue
        parts.setdefault(s, []).append({k: a[k] for k in K})
    return {s: {k: np.concatenate([p[k] for p in v]) for k in K} | {"nfiles": len(v)} for s, v in parts.items()}


def eff(d, mlo, mhi, t, htmin=0.0):
    ma = 0.5 * (d["r14A_m1"] + d["r14A_m2"]); A = np.abs(d["r14A_m1"] - d["r14A_m2"]) / np.maximum(d["r14A_m1"] + d["r14A_m2"], 1e-9)
    den = d["ht"] > htmin
    k = den & d["r14A_valid"].astype(bool) & (A < 0.1) & (ma > mlo) & (ma < mhi) & (d["r14A_psig"] >= t)
    w = d["w"]; W = w[den].sum(); p = w[k].sum() / W
    # ratio estimator variance
    e = np.sqrt(((w[k] * (1 - p)) ** 2).sum() + ((w[den & ~k] * p) ** 2).sum()) / W
    return p, e, int(k.sum())


out = {}
def table(name, sets, ranges, ths, htmin=0.0, ref="train"):
    print(f"\n== {name}: weighted fraction of events with valid pairing, A < 0.1, m_avg in range, P(signal) >= t" + (f", H_T > {htmin:g}" if htmin else ""))
    print("   files:", {s: d["nfiles"] for s, d in sets.items()}, " events:", {s: len(d["w"]) for s, d in sets.items()})
    for (mlo, mhi) in ranges:
        for t in ths:
            row = {s: eff(d, mlo, mhi, t, htmin) for s, d in sets.items()}
            txt = "  ".join(f"{s} {p:.3e} ± {e:.1e} (n {n})" for s, (p, e, n) in row.items())
            rat = "  ".join(f"{s}/{ref} {row[s][0] / row[ref][0]:.3f} ± {np.hypot(row[s][1] / row[s][0], row[ref][1] / row[ref][0]) * row[s][0] / row[ref][0]:.3f}" for s in row if s != ref and row[ref][0] > 0 and row[s][0] > 0)
            print(f"   m_avg {mlo:g}-{mhi:g}  P >= {t}:  {txt}   |  {rat}")
            out.setdefault(name, []).append({"m": [mlo, mhi], "t": t, **{s: list(v) for s, v in row.items()}})


which = sys.argv[1:] or ["vlm", "old", "ht", "siglm", "sig"]
if "vlm" in which:
    s = load("qcdvlm_c*.npz", lambda c: "train" if c < 1000 else "val" if c < 1050 else "test")
    a = np.load(f"{D}/data/qcdvlm.npz"); s["unseen"] = {k: a[k] for k in K} | {"nfiles": len(a["files_c"])}
    s["val+test+unseen"] = {k: np.concatenate([s[x][k] for x in ("val", "test", "unseen")]) for k in K} | {"nfiles": 0}
    table("qcdvlm (pTHat > 80, H_T > 400, sym filter)", s, [(250, 450), (450, 800), (800, 1150)], [0.9, 0.99, 0.997])
if "old" in which:
    s = load("qcd_c*.npz", lambda c: "train" if c % 10 < 8 else "val" if c % 10 == 8 else "test")
    s["val+test"] = {k: np.concatenate([s[x][k] for x in ("val", "test")]) for k in K} | {"nfiles": 0}
    table("prod/qcd (pTHat > 200)", s, [(450, 800), (800, 1200), (1200, 2500)], [0.9, 0.99, 0.997], htmin=900)
if "ht" in which:
    s = load("qcdht_c*.npz", lambda c: "train" if c < 1400 else "val" if c < 1700 else "test")
    s["val+test"] = {k: np.concatenate([s[x][k] for x in ("val", "test")]) for k in K} | {"nfiles": 0}
    table("qcdht (H_T > 1.2 TeV)", s, [(800, 1200), (1200, 1700), (1700, 2500)], [0.9, 0.99, 0.997, 0.999], htmin=1200)
if "siglm" in which:
    for M in (300, 400, 500, 700, 1000):
        for kind in ("incl", "boost"):
            s = load(f"siglm_m{M}_{kind}_c*.npz", lambda c: "train" if c < 16 else "val+test")
            table(f"signal siglm m{M} {kind}", s, [(0.85 * M, 1.1 * M)], [0.0, 0.9, 0.99, 0.997])
if "sig" in which:
    for M in (1000, 1500, 2000):
        s = load(f"sig_m{M}_c*.npz", lambda c: "train" if c % 10 < 8 else "val" if c % 10 == 8 else "test")
        table(f"signal prod m{M}", s, [(0.85 * M, 1.1 * M)], [0.0, 0.9, 0.99, 0.997, 0.999])
json.dump(out, open(f"{D}/item4_leak.json", "w"), indent=1, default=float)
