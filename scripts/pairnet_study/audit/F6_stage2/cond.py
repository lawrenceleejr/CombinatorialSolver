#!/usr/bin/env python
"""Item 2: hypothesis-conditioned pairing from the stored symmetric interpretations (A < 0.1, sorted by gated logit).
For hypothesis M the candidate is the highest-scored symmetric interpretation with m_avg in [0.85, 1.1] M.
  arg    it must be the network's argmax (current analysis)
  cond   any rank (i)
  condd  log-probability within delta of the argmax, delta tuned (ii)
  pwin   discriminant = P x summed pairing probability of the symmetric interpretations in the window (iii)
  2cat   argmax-in-window events and conditioned-only events as two counting categories with their own P cuts
QCD events need P >= 0.5 (stored interpretations), so all P thresholds are >= 0.5. No spectrum-shape requirement is
applied here (every candidate is in the window by construction); sculpting is measured instead.
usage: cond.py out.json [masses]"""
import json, sys, time
import numpy as np
import f6lib as F

out = sys.argv[1]
MASSES = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [400, 700, 1000, 1500, 2000]
t0 = time.time()
Q = F.load_qcd(with_S=True)
print(f"loaded {time.time() - t0:.0f}s, rich QCD {len(Q['rich_idx'])}", flush=True)


def cond_tab(d, M, w, lo=F.WIN[0], hi=F.WIN[1]):
    r = d["rich_idx"]; ma = d["S_ma"] / M
    inw = (ma > lo) & (ma < hi)
    j = inw.argmax(1)
    rows = np.where(inw.any(1) & (d["ht"][r] > F.TRIG) & (w[r] > 0) & (d["psig"][r] >= 0.5))[0]
    eid, jj = r[rows], j[rows]
    isarg = d["S_idx"][rows, jj] == d["best"][eid]
    lp = d["S_lp"][rows, jj].astype(np.float32)
    dlp = np.where(isarg, 0.0, np.minimum(lp - np.log(np.maximum(d["pbest"][eid], 1e-30)), -1e-3)).astype(np.float32)
    pw = (np.exp(d["S_lp"][rows].astype(np.float32)) * inw[rows]).sum(1)
    m = ma[rows, jj]
    T = F.Tab(w[eid], d["half"][eid], m, eid)
    T.add("psig", d["psig"][eid], ">="); T.add("dlp", dlp, ">="); T.add("pwinP", (pw * d["psig"][eid]).astype(np.float32), ">=")
    T.add("ht", d["ht"][eid] / M, ">")
    T.add("delta_min", d["S_dmin"][rows, jj].astype(np.float32) * m, ">"); T.add("delta_max", d["S_dmax"][rows, jj].astype(np.float32) * m, ">")
    T.rank = jj; T.isarg = isarg
    return T


def pgrid(n):
    def g(S, h):
        v = S.x[n][S.h == h]
        return [0.5] + [float(q) for q in np.quantile(v, F.PQ) if q > 0.5] if len(v) else [0.5]
    return g


DG = [-np.inf, -5, -4, -3, -2, -1.5, -1, -0.5, -0.25, 0.0]
BOOST = ["ht", "delta_max", "delta_min"]


def two_cat(St, Bt, f, extra):
    """Category 1: argmax in window (tuned as 'arg'); category 2: conditioned-only events, own cuts tuned on the tuning
    fold to minimise the combined limit with category 1 fixed (dropped if it does not help by 2%)."""
    sels = []
    for h in (0, 1):
        c1, b1 = F.tune(St, Bt, ["psig"] + extra, h, f, grids={"psig": pgrid("psig")}, shape=False, start={"dlp": 0.0})
        a_s, a_b = ~St.isarg, ~Bt.isarg

        def yields(c, hh, floors):
            ks = F.passes(St, c) & a_s & (St.h == hh); kb = F.passes(Bt, c) & a_b & (Bt.h == hh)
            if floors and (kb.sum() < F.MIN_NB or F.neff(Bt.w[kb]) < F.MIN_NEFF or ks.sum() < F.MIN_NS):
                return None
            return ks, kb
        k1s = F.passes(St, c1) & (St.h == h); k1b = F.passes(Bt, c1) & (Bt.h == h)
        S1, B1 = 2 * F.L * St.w[k1s].sum(), 2 * F.L * Bt.w[k1b].sum()
        cur = F.mu_up_multi([S1], [B1], f); c2 = None
        G = {"psig": pgrid("psig")(St, h), "dlp": DG[:-1], **{n: F.VARS[n][2] for n in extra}}
        trial0 = {n: g[0] for n, g in G.items()}
        for _ in range(2):
            for n, g in G.items():
                for t in g:
                    tr = {**(c2 or trial0), n: t}
                    y = yields(tr, h, True)
                    if y is None:
                        continue
                    lim = F.mu_up_multi([S1, 2 * F.L * St.w[y[0]].sum()], [B1, 2 * F.L * Bt.w[y[1]].sum()], f)
                    if lim < cur * (0.98 if c2 is None else 1 - 1e-9):
                        cur, c2 = lim, tr
        s = F.Sel(St, Bt, c1, 1 - h)
        if c2 is not None:
            ks, kb = yields(c2, 1 - h, False)
            s.cats.append((St.eid[ks], St.w[ks], Bt.eid[kb], Bt.w[kb]))
            s.cuts = {**{f"c1_{k}": v for k, v in c1.items()}, **{f"c2_{k}": v for k, v in c2.items()}}
        sels.append(s)
    return sels


res = {}
for M in MASSES:
    S = F.load_sig(M, with_S=True)
    if S is None:
        continue
    S["rich_idx"] = np.arange(len(S["w"])) if len(S["rich_idx"]) == len(S["w"]) else S["rich_idx"]
    w = F.weights(Q, M)
    Bt, St = cond_tab(Q, M, w), cond_tab(S, M, S["wfb"])
    res[M] = {}
    for f in (0.10, 0.05):
        folds = [{}, {}]
        def put(name, sels):
            for i in (0, 1):
                folds[i][name] = sels[i]
        for tag, extra in (("P", []), ("P+boost", BOOST)):
            run = lambda free, **kw: F.run_conf(St, Bt, free + extra, f, shape=False, **kw)
            put(f"arg:{tag}", run(["psig"], grids={"psig": pgrid("psig")}, start={"dlp": 0.0}))
            put(f"cond:{tag}", run(["psig"], grids={"psig": pgrid("psig")}))
            put(f"condd:{tag}", run(["psig", "dlp"], grids={"psig": pgrid("psig"), "dlp": DG}))
            put(f"pwin:{tag}", run(["pwinP"], grids={"pwinP": pgrid("pwinP")}, start={}))
            put(f"2cat:{tag}", two_cat(St, Bt, f, extra))
        r = F.summarize(folds, f, len(S["w"]), len(Q["w"]), "arg:P")
        rb = F.summarize([{k: v for k, v in folds[i].items() if k.endswith("P+boost")} for i in (0, 1)], f, len(S["w"]), len(Q["w"]), "arg:P+boost")
        for k in rb:
            r[k]["ratio_vs_argboost"] = rb[k]["ratio"]; r[k]["ratio_vs_argboost_err"] = rb[k]["ratio_err"]
        # sculpting at the P cut of arg:P (fold 0 tuning), all events (both folds)
        pc = folds[0]["arg:P"].cuts["psig"]
        def y(T, sel):
            k = (T.x["psig"] >= pc) & sel
            return float(T.w[k].sum()), int(k.sum()), F.neff(T.w[k])
        sc = {"p_cut": float(pc)}
        for nm, fs in (("arg", lambda T: T.isarg), ("cond_all", lambda T: np.ones(len(T.w), bool)), ("cond_d1", lambda T: T.x["dlp"] >= -1), ("cond_d0.5", lambda T: T.x["dlp"] >= -0.5),
                       ("pwinP>=pc", None)):
            if fs is None:
                ks, kb = St.x["pwinP"] >= pc, Bt.x["pwinP"] >= pc
                sc[nm] = {"sig_eff": float(St.w[ks].sum()), "qcd_fb": float(Bt.w[kb].sum()), "qcd_n": int(kb.sum()), "qcd_neff": F.neff(Bt.w[kb])}
            else:
                a, b = y(St, fs(St)), y(Bt, fs(Bt))
                sc[nm] = {"sig_eff": a[0], "qcd_fb": b[0], "qcd_n": b[1], "qcd_neff": b[2]}
        # sliding hypothesis: QCD yield arg vs cond for neighbouring hypotheses (same weights)
        sl = {}
        for k in (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4):
            Tb = cond_tab(Q, M * k, w)
            kk = Tb.x["psig"] >= pc
            sl[f"{k:g}"] = {"arg_fb": float(Tb.w[kk & Tb.isarg].sum()), "cond_fb": float(Tb.w[kk].sum()), "cond_d1_fb": float(Tb.w[kk & (Tb.x["dlp"] >= -1)].sum()),
                            "arg_neff": F.neff(Tb.w[kk & Tb.isarg]), "cond_neff": F.neff(Tb.w[kk])}
        res[M][f"f{f:g}"] = {"sel": r, "sculpt": sc, "sliding": sl,
                             "rank_sig": np.bincount(St.rank[St.x["psig"] >= pc], minlength=6)[:6].tolist(), "rank_qcd_w": [float(Bt.w[(Bt.x["psig"] >= pc) & (Bt.rank == i)].sum()) for i in range(6)]}
        print(f"M={M} f={f:g}: " + "  ".join(f"{k} {v['limit_fb']:.4g} ({v['ratio']:.2f}+-{v['ratio_err']:.2f}, neff {v['neffB']:.0f})" for k, v in r.items()), flush=True)
        print("   sculpt", json.dumps(sc, default=float), f"{time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(out, "w"), indent=1, default=float)
