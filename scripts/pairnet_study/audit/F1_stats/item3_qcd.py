"""Item 3: agreement between QCD samples where two are complete, and completeness near the H_T thresholds.
Uses the sample-alone cross-section weights w0 (before any merge factor). Phase space: PairNet pairing valid, A < 0.1, H_T > 400."""
import sys, numpy as np, pipe
import tailmerge as TM
tag = sys.argv[1] if len(sys.argv) > 1 else "main"
Q = dict(np.load(f"{pipe.D}/data/qcd_sel_{tag}.npz")); S = pipe.SRC
src, ht, w0, ma, p = Q["src"], Q["ht"], Q["w0"], Q["pn_mavg"], Q["psig"]
print("columns:", sorted(Q)); print("events per sample:", {S[i]: int((src == i).sum()) for i in np.unique(src)})
def y(k):
    w = w0[k]; return w.sum(), np.sqrt((w ** 2).sum()), int(k.sum())
def cmp(names, k, label):
    r = {n: y(k & (src == S.index(n))) for n in names}
    ref = names[0]
    txt = "  ".join(f"{n} {v[0]:.4g} ± {v[1]:.2g} ({v[2]})" for n, v in r.items())
    rat = "  ".join(f"{n}/{ref} {r[n][0] / r[ref][0]:.3f} ± {r[n][0] / r[ref][0] * np.hypot(r[n][1] / r[n][0], r[ref][1] / r[ref][0]):.3f}" for n in names[1:] if r[n][0] > 0 and r[ref][0] > 0)
    print(f"  {label}: {txt} | {rat}")
print("\n(1) yields in fb (sample-alone weights), window [0.85, 1.1] M, P >= t; samples complete in the H_T range")
for M in (500, 600, 700, 800, 900, 1000):
    for t in (-1, 0.9, 0.99):
        win = (ma > 0.85 * M) & (ma < 1.1 * M) & (p >= t)
        cmp(["vlm", "old"], win & (ht >= 900) & (ht < 1200), f"M {M} P>={t} H_T 900-1200")
        cmp(["vlm", "old", "ht"], win & (ht >= 1200), f"M {M} P>={t} H_T > 1200")
for M in (1200, 1500, 2000):
    for t in (-1, 0.9, 0.99):
        win = (ma > 0.85 * M) & (ma < 1.1 * M) & (p >= t)
        cmp(["old", "ht"], win & (ht >= 1200), f"M {M} P>={t} H_T > 1200")
print("\n(2) H_T turn-on near the completeness thresholds (A < 0.1, 150 < m_avg < 1150)")
base = (ma > 150) & (ma < 1150)
for lo, hi in ((900, 950), (950, 1000), (1000, 1100), (1100, 1200)):
    cmp(["vlm", "old"], base & (ht >= lo) & (ht < hi), f"H_T {lo}-{hi}")
for lo, hi in ((1200, 1250), (1250, 1300), (1300, 1400), (1400, 1600), (1600, 2000), (2000, 1e9)):
    cmp(["vlm", "old", "ht"], base & (ht >= lo) & (ht < hi), f"H_T {lo}-{hi:g}")
print("   same, m_avg > 1150 (old vs ht only)")
for lo, hi in ((1200, 1250), (1250, 1300), (1300, 1400), (1400, 1600), (1600, 2000), (2000, 1e9)):
    cmp(["old", "ht"], (ma > 1150) & (ht >= lo) & (ht < hi), f"H_T {lo}-{hi:g}")
print("\n(3) tail samples inside their filter F vs the base samples (sample-alone weights)")
for tn in [t for t in ("qcdvlmT", "qcdhtT", "qcd700T", "qcd700U") if (src == S.index(t)).any()]:
    spec = TM.TAILS[tn]
    F = TM.in_filter(Q["pn_valid"], Q["pn_m1"], Q["pn_m2"], p, Q["pn_delta1"], Q["pn_delta2"], spec)
    Fp = F & (p >= spec["pmin"])
    edges = {400: [(400, 450), (450, 500), (500, 600), (600, 900), (900, 1200), (1200, 1e9)], 900: [(900, 950), (950, 1000), (1000, 1100), (1100, 1200), (1200, 1300), (1300, 1e9)], 1200: [(1200, 1250), (1250, 1300), (1300, 1500), (1500, 1e9)]}[spec["complete"]]
    for lo, hi in edges:
        names = [n for n, c in (("vlm", 400), ("old", 900), ("ht", 1200)) if lo >= c]
        cmp(names + [tn], F & (ht >= lo) & (ht < hi), f"{tn} in F, H_T {lo}-{hi:g}")
        cmp(names + [tn], Fp & (ht >= lo) & (ht < hi), f"{tn} in F and P >= {spec['pmin']}, H_T {lo}-{hi:g}")
    for M in (400, 700, 1000) if spec["mlo"] < 500 else (900, 1000):
        if not spec["mlo"] < 0.85 * M or not 1.1 * M < spec["mhi"]:
            continue
        for t in (0.9, 0.99):
            k = F & (ma > 0.85 * M) & (ma < 1.1 * M) & (p >= t) & (ht >= max(spec["complete"], 450))
            names = [n for n, c in (("vlm", 400), ("old", 900), ("ht", 1200)) if spec["complete"] >= c]
            cmp(names + [tn], k, f"{tn} window M {M} P>={t} H_T>{max(spec['complete'], 450)}")
