#!/usr/bin/env python
"""Baselines (P, P+boost, P+all, +categories) and items 3 (search variable) and 4 (pairing-distribution discriminants)
with the r14A argmax pairing. Nested two-fold procedure, paired bootstrap of the evaluation folds.
usage: base34.py out.json [masses]"""
import json, sys, time
import numpy as np
import f6lib as F

out = sys.argv[1]
MASSES = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else F.LOW_MASSES + F.HIGH_MASSES
FS = (0.10, 0.05)
t0 = time.time()
Q = F.load_qcd()
print("merge low", Q["mergeL"], "\nmerge high", Q["mergeH"], f"{time.time() - t0:.0f}s", flush=True)
KEEP = [0.98, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
# item 4 variables: name -> (column or function, direction)
X4 = {"ent<": (lambda d: d["ent"], "<"), "margin>": (lambda d: d["margin"], ">="), "pbest>": (lambda d: d["pbest"], ">="),
      "n_sym<": (lambda d: d["n_sym"].astype(np.float32), "<"), "n_sym>": (lambda d: d["n_sym"].astype(np.float32), ">="),
      "p_sym>": (lambda d: d["p_sym"], ">="), "p_near>": (lambda d: d["p_near"], ">="),
      "agree": (lambda d: d["agree"].astype(np.float32), ">="),
      "mspread3<": (lambda d: d["mspread3"], "<"), "mspread5<": (lambda d: d["mspread5"], "<"),
      "sl_margin>": (lambda d: d["sl_margin"], ">="), "sl_ent<": (lambda d: d["sl_ent"], "<"),
      "sl_dm<": (lambda d: np.abs(d["sl_ma_sig"] / np.maximum(d["pn_mavg"], 1e-9) - 1), "<"),
      "mest_dm<": (lambda d: np.abs(d["m_est"] / np.maximum(d["pn_mavg"], 1e-9) - 1), "<")}


def xgrid(S, n, h):
    k = (S.h == h) & (S.m > F.WIN[0]) & (S.m < F.WIN[1]); v = S.x[n][k]
    if S.dir[n] == "<":
        return [np.inf] + sorted(set(np.quantile(v, KEEP)), reverse=True)
    return [-np.inf] + sorted(set(np.quantile(v, 1 - np.array(KEEP))))


def run_extra(St, Bt, free, f, name, start_from=None):
    """free + one extra variable with its own quantile grid (per tuning fold)."""
    sels = []
    for h in (0, 1):
        g = {name: xgrid(St, name, h)}
        cuts, best = F.tune(St, Bt, free + [name], h, f, grids=g)
        s = F.Sel(St, Bt, cuts, 1 - h); s.feasible = bool(np.isfinite(best)) and F.shape_ok(St, Bt, cuts, h)
        sels.append(s)
    return sels


res = {}
for M in MASSES:
    S = F.load_sig(M)
    if S is None:
        print("no signal", M); continue
    w = F.weights(Q, M)
    Bt, St = F.make_tab(Q, M, w), F.make_tab(S, M, S["wfb"])
    for n, (fn, dr) in X4.items():
        Bt.add(n, fn(Q)[Bt.eid], dr); St.add(n, fn(S)[St.eid], dr)
    # item 3 tables: same events, other search variables
    alt = {}
    for nm, a in (("mest", 0.0), ("wavg25", 0.25), ("wavg50", 0.5), ("wavg75", 0.75)):
        mk = lambda d, T: (a * d["pn_mavg"][T.eid] + (1 - a) * d["m_est"][T.eid]) / M
        alt[nm] = (F.Tab(St.w, St.h, mk(S, St), St.eid, St.x, St.dir), F.Tab(Bt.w, Bt.h, mk(Q, Bt), Bt.eid, Bt.x, Bt.dir))
    for T, d in ((St, S), (Bt, Q)):
        T.add("mest_lo", d["m_est"][T.eid] / M, ">"); T.add("mest_hi", d["m_est"][T.eid] / M, "<")
    WG = {"wlo": [0.85, 0.80, 0.83, 0.88, 0.90, 0.92, 0.94], "whi": [1.10, 1.03, 1.05, 1.07, 1.15]}
    res[M] = {}
    for f in FS:
        folds = [{}, {}]
        def put(name, sels):
            for i in (0, 1):
                folds[i][name] = sels[i]
        for name, free in F.CONF.items():
            s, sc = F.run_conf(St, Bt, free, f, cats=3, smin=0.8)
            put(name, s); put(name + "+cats", sc)
        # ---- item 3
        for base in ("P", "P+all"):
            free = F.CONF[base]
            put(f"{base}|win", F.run_conf(St, Bt, free + ["wlo", "whi"], f, grids=WG))
            put(f"{base}|mest", F.run_conf(*alt["mest"], free, f))
            put(f"{base}|mest|win", F.run_conf(*alt["mest"], free + ["wlo", "whi"], f, grids=WG))
            put(f"{base}|both", F.run_conf(St, Bt, free + ["mest_lo", "mest_hi"], f,
                                           grids={"mest_lo": [-1e9, 0.75, 0.8, 0.85, 0.88, 0.9, 0.92], "mest_hi": [1e9, 1.25, 1.2, 1.15, 1.1, 1.07, 1.05]}))
            # weighted average: the mixing fraction is chosen on the tuning fold (best tuned limit), window re-tuned
            sels = []
            for h in (0, 1):
                best = None
                for nm in ("wavg25", "wavg50", "wavg75"):
                    cuts, lim = F.tune(*alt[nm], free + ["wlo", "whi"], h, f, grids=WG)
                    if best is None or lim < best[0]:
                        best = (lim, nm, cuts)
                lim, nm, cuts = best
                s = F.Sel(*alt[nm], cuts, 1 - h, info={"mix": nm}); s.feasible = bool(np.isfinite(lim)) and F.shape_ok(*alt[nm], cuts, h)
                s.cuts = {**cuts, "mix_mavg": float(nm[4:]) / 100}
                sels.append(s)
            put(f"{base}|wavg|win", sels)
        # ---- item 4
        for base in ("P", "P+all"):
            for n in X4:
                put(f"{base}+{n}", run_extra(St, Bt, F.CONF[base], f, n))
        r = F.summarize(folds, f, len(S["w"]), len(Q["w"]), "P+all")
        fam = lambda k: k == "P" or k.startswith("P|") or (k.startswith("P+") and not k.startswith(("P+all", "P+boost")))
        rP = F.summarize([{k: v for k, v in folds[i].items() if fam(k)} for i in (0, 1)], f, len(S["w"]), len(Q["w"]), "P")
        for k in rP:
            r[k]["ratio_vs_P"] = rP[k]["ratio"]; r[k]["ratio_vs_P_err"] = rP[k]["ratio_err"]
        # diagnostics for item 3: signal fraction and QCD in the window at the P-only cut of fold 0, m_avg vs m_est
        c0 = {"psig": folds[0]["P"].cuts["psig"]}
        dg = {}
        for nm, (s_, b_) in {"mavg": (St, Bt), "mest": alt["mest"]}.items():
            ks, kb = F.passes(s_, c0), F.passes(b_, c0)
            wi = lambda T: (T.m > 0.85) & (T.m < 1.1)
            dg[nm] = {"sig_eff_in_window": float(s_.w[ks & wi(s_)].sum()), "sig_frac_in_window_of_0.4-1.6": float(s_.w[ks & wi(s_)].sum() / max(s_.w[ks].sum(), 1e-300)),
                      "qcd_fb_in_window": float(b_.w[kb & wi(b_)].sum()),
                      "sig_rms_core": float(np.sqrt(np.average((s_.m[ks & wi(s_)] - np.average(s_.m[ks & wi(s_)], weights=s_.w[ks & wi(s_)])) ** 2, weights=s_.w[ks & wi(s_)]))) if (ks & wi(s_)).any() else None}
        res[M][f"f{f:g}"] = {"sel": r, "diag3": dg}
        b = r["P+all"]; p = r["P"]
        print(f"M={M} f={f:g}: P {p['limit_fb']:.4g} (neff {p['neffB']:.0f})  P+boost {r['P+boost']['limit_fb']:.4g}  P+all {b['limit_fb']:.4g} (nS {b['nS']}, neff {b['neffB']:.0f}, +-{b['limit_relerr']:.2f})"
              f"  +cats {r['P+all+cats']['limit_fb']:.4g}  | " + "  ".join(f"{k} {r[k]['ratio']:.2f}+-{r[k]['ratio_err']:.2f}" for k in r if k.startswith("P+all|")) + f"  {time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(out, "w"), indent=1, default=float)
