"""Item 5: signal normalisation cross-checks: (a) low-mass chain (siglm, 1/n_tried) vs high-mass chain (prod, filter_eff / n_stored)
efficiencies of the same selection, (b) incl/boost merge, (c) power-shower variation, (d) siglm vs sigX samples."""
import pickle, json, numpy as np, pipe
HA, _ = pickle.load(open(f"{pipe.D}/data/high_arrays.pkl", "rb"))
def sel(d, M, pmin=-1, htmin=0):
    return (d["pn_asym"] < 0.1) & d["pn_valid"].astype(bool) & (d["pn_mavg"] > 0.85 * M) & (d["pn_mavg"] < 1.1 * M) & (d["psig"] >= pmin) & (d["ht"] > htmin)
def eff(d, k):
    w = d["wfb"]; return w[k].sum(), np.sqrt((w[k] ** 2).sum())
print("(a) efficiency of [valid, A < 0.1, m_avg in [0.85, 1.1] M, P >= t]: low-mass chain vs high-mass chain")
for M in (500, 600, 700, 800, 900, 1000):
    s = pipe.merged_signal(M); A = HA[M]; hs = A["s"]; nt = A["ntot"].sum()
    for t in (-1, 0.9, 0.99):
        e1, d1 = eff(s, sel(s, M, t))
        k = hs["valid"] & (hs["m"] > 0.85) & (hs["m"] < 1.1) & (hs["p"] >= t); e2 = k.sum() / nt * A["filt"]; d2 = np.sqrt(k.sum()) / nt * A["filt"]
        k3 = (hs["m"] > 0.85) & (hs["m"] < 1.1) & (hs["p"] >= t); e3 = k3.sum() / nt * A["filt"]
        print(f"  M {M} P>={t}: low {e1:.5f} ± {d1:.5f}   high {e2:.5f} ± {d2:.5f}  (without valid {e3:.5f})   high/low {e2 / e1:.3f} ± {e2 / e1 * np.hypot(d1 / e1, d2 / e2):.3f}   filter_eff {A['filt']:.4f} n_stored {nt}")
print("(b) incl/boost merge: boost events below the merge threshold 2.3 M; inclusive vs boost estimates above it")
for M in (300, 400, 500, 700, 1000):
    inc, bst = pipe.load_sig(M, "incl"), pipe.load_sig(M, "boost"); hb = 2.3 * M
    fi, di = eff(inc, inc["ht"] > hb); fb, db = eff(bst, bst["ht"] > hb); lowb = eff(bst, bst["ht"] <= hb)[0]
    m = pipe.merged_signal(M); k = sel(m, M); e_m, d_m = eff(m, k); e_i, d_i = eff(inc, sel(inc, M))
    print(f"  M {M}: frac(H_T > 2.3M): incl {fi:.4f} ± {di:.4f}, boost {fb:.4f} ± {db:.4f}, ratio {fb / fi:.3f} ± {fb / fi * np.hypot(di / fi, db / fb):.3f}; boost events with H_T <= 2.3M: {(bst['ht'] <= hb).mean():.4f} of sample "
          f"(adds {lowb * (1 - (inc['ht'] > hb).sum() / ((inc['ht'] > hb).sum() + len(bst['ht']))):.2e} to a total of 1); window eff merged {e_m:.5f} ± {d_m:.5f} vs incl only {e_i:.5f} ± {d_i:.5f}; sum of merged weights {m['wfb'].sum():.4f}")
print("(c) power shower vs default (inclusive samples), efficiency of the window selection at P >= t")
for M in (400, 700, 1000):
    inc, pw = pipe.load_sig(M, "incl"), pipe.load_sig(M, "power")
    if pw is None:
        print("  no power sample for", M); continue
    for t in (-1, 0.9, 0.99):
        a, da = eff(inc, sel(inc, M, t)); b, db = eff(pw, sel(pw, M, t))
        print(f"  M {M} P>={t}: default {a:.5f} ± {da:.5f}, power {b:.5f} ± {db:.5f}, power/default {b / a:.3f} ± {b / a * np.hypot(da / a, db / b):.3f}")
print("(d) siglm (4 files) vs sigX (100 files), merged incl+boost separately")
for M in (650, 700, 750):
    for kind in ("incl", "boost"):
        import os
        a, nf, raw = pipe.load_cache(f"siglm_m{M}_{kind}") if os.path.exists(f"{pipe.D}/data/siglm_m{M}_{kind}.npz") else (None, 0, None)
        if a is None or not os.path.exists(f"{pipe.D}/data/sig{M}X_{kind}.npz"):
            continue
        b, nfb, rawb = pipe.load_cache(f"sig{M}X_{kind}")
        for t in (-1, 0.9, 0.99):
            r = []
            for d, rw in ((a, raw), (b, rawb)):
                m1, m2 = d["r14A_m1"], d["r14A_m2"]; ma = 0.5 * (m1 + m2); As = np.abs(m1 - m2) / np.maximum(m1 + m2, 1e-9)
                k = d["r14A_valid"].astype(bool) & (As < 0.1) & (ma > 0.85 * M) & (ma < 1.1 * M) & (d["r14A_psig"] >= t); nt = rw["files_n_tried"].sum()
                r.append((k.sum() / nt, np.sqrt(k.sum()) / nt, len(ma) / nt))
            print(f"  M {M} {kind} P>={t}: siglm {r[0][0]:.5f} ± {r[0][1]:.5f} (stored/tried {r[0][2]:.3f}), sigX {r[1][0]:.5f} ± {r[1][1]:.5f} (stored/tried {r[1][2]:.3f}), X/lm {r[1][0] / r[0][0]:.3f} ± {r[1][0] / r[0][0] * np.hypot(r[0][1] / r[0][0], r[1][1] / r[1][0]):.3f}")
