"""Items 2-3: per working point and variant: signal efficiency ratio, QCD background ratio (tail core + unfiltered
in-migration), limit ratio with fixed cuts and with the P cut re-tuned to the nominal background."""
import json, sys, numpy as np
from wps import *
D = "/scratch/llee/pairnet/audit/F3_robust"
import os
MD = os.environ.get("F3_MERGED", "merged"); SUF = os.environ.get("F3_SUF", "")
_c = {}


def L(g):
    if g not in _c:
        z = np.load(f"{D}/out/{MD}/{g}.npz"); _c[g] = {k: z[k] for k in z.files}
    return _c[g]


def ratio(w, a, b):
    """R = sum w a / sum w b with the paired error."""
    A, B = (w * a).sum(), (w * b).sum()
    if B <= 0:
        return np.nan, np.nan
    R = A / B
    return R, np.sqrt((w ** 2 * (a - R * b) ** 2).sum()) / B


def neff(w):
    return w.sum() ** 2 / max((w ** 2).sum(), 1e-300)


VAR = [str(v) for v in L("sig700")["variants"]]
BASE = {v: ("thr:40" if v.endswith("@40") else "nom") for v in VAR}
rows = []
rng = np.random.default_rng(11)
for name, wp in WPS.items():
    M = wp["M"]; S = L(f"sig{M}"); tn, un, rngm = QCD[M]; T = L(tn)
    U = L(un) if os.path.exists(f"{D}/out/{MD}/{un}.npz") else {k: v[:0] for k, v in T.items() if hasattr(v, "ndim") and v.ndim >= 1 and len(v) == len(T["w"])} | {"frac": 1.0}
    wS, wT, wU = S["w"], T["w"] / T["frac"], U["w"] / U["frac"]
    # U events that are NOT in the tail-core definition at nominal (those are represented by T)
    inF = U["valid"][:, 0] & (U["psig"][:, 0] >= 0.9) & (U["asym"][:, 0] < 0.4) & (U["mavg"][:, 0] > rngm[0]) & (U["mavg"][:, 0] < rngm[1])
    pS = {i: aux(S, wp, i) for i in range(len(VAR))}; pT = {i: aux(T, wp, i) for i in range(len(VAR))}; pU = {i: aux(U, wp, i) & ~inF for i in range(len(VAR))}
    t = wp["psig"]
    for i, v in enumerate(VAR):
        b = VAR.index(BASE[v])
        s1, s0 = pS[i] & (S["psig"][:, i] >= t), pS[b] & (S["psig"][:, b] >= t)
        t1, t0 = pT[i] & (T["psig"][:, i] >= t), pT[b] & (T["psig"][:, b] >= t)
        u1, u0 = pU[i] & (U["psig"][:, i] >= t), pU[b] & (U["psig"][:, b] >= t)
        rS, eS = ratio(wS, s1, s0)
        rT, eT = ratio(wT, t1, t0)                       # tail-core part (out-migration + in-migration from P_nom >= 0.9)
        B0 = (wT * t0).sum() + (wU * u0).sum(); B1 = (wT * t1).sum() + (wU * u1).sum()
        eB = np.sqrt((wT ** 2 * (t1 * 1.0 - B1 / B0 * t0) ** 2).sum() + (wU ** 2 * (u1 * 1.0 - B1 / B0 * u0) ** 2).sum()) / B0
        rB = B1 / B0
        lim = rB / rS; elim = lim * np.hypot(eS / rS, eB / rB) if rS > 0 else np.nan
        limT = rT / rS; elimT = limT * np.hypot(eS / rS, eT / rT) if rS > 0 and rT > 0 else np.nan
        # ---- re-tune the P cut so that B_var(t') = B_base; limit ratio = eps_S(base) / eps_S(var, t')
        def retune(wt, wu):
            b0 = (wt * t0).sum() + (wu * u0).sum()
            ps = np.concatenate([T["psig"][pT[i], i], U["psig"][pU[i], i]]); ww = np.concatenate([wt[pT[i]], wu[pU[i]]])
            o = np.argsort(-ps); c = np.cumsum(ww[o]); k = np.searchsorted(c, b0, side="right")
            tt = ps[o][min(k, len(o) - 1)] if len(o) else 1.0
            if k == 0:
                tt = 1.0
            return tt, b0
        tt, _ = retune(wT, wU)
        sden = (wS * s0).sum()
        rSt = (wS * (pS[i] & (S["psig"][:, i] > tt))).sum() / sden
        bs = []
        for _ in range(40):
            kt, ku, ks = rng.poisson(1, len(wT)), rng.poisson(1, len(wU)), rng.poisson(1, len(wS))
            t2, _ = retune(wT * kt, wU * ku)
            d = (wS * ks * s0).sum()
            bs.append((wS * ks * (pS[i] & (S["psig"][:, i] > t2))).sum() / d if d > 0 else np.nan)
        bs = np.array(bs); bs = bs[np.isfinite(bs) & (bs > 0)]
        limR = 1 / rSt if rSt > 0 else np.inf
        elimR = 0.5 * (np.quantile(1 / bs, 0.84) - np.quantile(1 / bs, 0.16)) if len(bs) > 10 else np.nan
        rows.append(dict(wp=name, M=M, var=v, base=BASE[v], effS0=float((wS * s0).sum()), nS0=int(s0.sum()), rS=rS, eS=eS,
                         B0=float(B0), B0_T=float((wT * t0).sum()), nT0=int(t0.sum()), neffT0=float(neff(wT[t0])) if t0.any() else 0.0,
                         B0_U=float((wU * u0).sum()), nU0=int(u0.sum()), rT=rT, eT=eT, B1_U=float((wU * u1).sum()), nU1=int(u1.sum()),
                         wU_typ=float(np.median(wU)), rB=rB, eB=eB, lim=lim, elim=elim, limT=limT, elimT=elimT, t_retuned=float(tt), limR=limR, elimR=elimR))
    print(name, "done", flush=True)
json.dump(rows, open(f"{D}/out/tab/variations{SUF}.json", "w"), indent=0, default=float)

with open(f"{D}/out/tab/variations{SUF}.txt", "w") as f:
    for name in WPS:
        r0 = next(r for r in rows if r["wp"] == name and r["var"] == "nom")
        f.write(f"\n### {name}: eff_S {r0['effS0']:.5f} (n {r0['nS0']}), B_nom {r0['B0']:.1f} fb (tail core {r0['B0_T']:.1f}, n {r0['nT0']}, neff {r0['neffT0']:.0f}; "
                f"outside-core U {r0['B0_U']:.1f} fb, n {r0['nU0']}; one U event = {r0['wU_typ']:.1f} fb)\n")
        f.write(f"{'variant':24s} {'base':7s} {'S ratio':>15s} {'B core ratio':>15s} {'U in (n, fb)':>14s} {'B ratio':>15s} {'limit x (fixed cuts)':>21s} {'core only':>14s} {'limit x (P re-tuned)':>21s} {'P cut':>8s}\n")
        for r in rows:
            if r["wp"] != name:
                continue
            f.write(f"{r['var']:24s} {r['base']:7s} {r['rS']:7.3f}±{r['eS']:.3f}  {r['rT']:7.3f}±{r['eT']:.3f}  {r['nU1']:3d} {r['B1_U']:8.1f}  {r['rB']:7.3f}±{r['eB']:.3f}  {r['lim']:9.2f}±{r['elim']:.2f}"
                    f"  {r['limT']:7.2f}±{r['elimT']:.2f}  {r['limR']:9.2f}±{r['elimR']:.2f}   {r['t_retuned']:.5f}\n")
print(open(f"{D}/out/tab/variations{SUF}.txt").read())
