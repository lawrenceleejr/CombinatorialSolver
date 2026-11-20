"""Items 4-5: mass-variable response and mass / H_T dependence of P; QCD m_avg sculpting. Nominal variant only."""
import json, numpy as np
from wps import *
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
D = "/scratch/llee/pairnet/audit/F3_robust"; out = open(f"{D}/out/tab/item45.txt", "w")
R = {}


def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.write(s + "\n")


def L(g):
    z = np.load(f"{D}/out/merged/{g}.npz"); return {k: (z[k][:, 0] if z[k].ndim == 2 else z[k]) for k in z.files if k != "variants"}


def wq(x, w, q):
    o = np.argsort(x); c = np.cumsum(w[o]) / w.sum(); return np.interp(q, c, x[o])


def shortest(x, w, f=0.68):
    """Shortest interval holding fraction f; returns (lo, hi)."""
    o = np.argsort(x); x, c = x[o], np.cumsum(w[o]) / w.sum()
    j = np.searchsorted(c, c - 0 + f); ok = j < len(x); i = np.where(ok)[0]
    k = i[np.argmin(x[j[ok]] - x[i])]; return x[k], x[j[k]]


def peak(x, w):
    lo, hi = shortest(x, w, 0.3); return 0.5 * (lo + hi)


def eff(w, a, b):
    d = (w * b).sum(); r = (w * (a & b)).sum() / d; return r, np.sqrt((w[b] ** 2 * ((a[b]) - r) ** 2).sum()) / d


PWP = {400: 0.98833, 700: 0.99705, 1000: 0.997445, 1500: 0.998078, 2000: 0.99928}
S = {M: L(f"sig{M}") for M in PWP}
P("== Item 4: signal response (weights per produced event). pre = valid, A<0.1, H_T>450; +P = working-point P cut")
P(f"{'M':>5s} {'sel':5s} {'var':6s} {'acc':>8s} {'peak/M':>7s} {'med/M':>7s} {'68% width/M':>12s} {'in[.85,1.1]':>12s} {'in[.92,1.04]':>12s}")
for M, s in S.items():
    pre = s["valid"] & (s["asym"] < 0.1) & (s["ht"] > TRIG)
    for sel, m in (("pre", pre), ("+P", pre & (s["psig"] >= PWP[M]))):
        for var in ("mavg", "mest"):
            x, w = s[var][m] / M, s["w"][m]
            lo, hi = shortest(x, w)
            r = dict(acc=w.sum(), peak=peak(x, w), med=wq(x, w, 0.5), w68=hi - lo, fw=w[(x >= .85) & (x < 1.1)].sum() / w.sum(), fn=w[(x >= .92) & (x < 1.04)].sum() / w.sum(), n=int(m.sum()))
            R[f"sig{M}_{sel}_{var}"] = r
            P(f"{M:5d} {sel:5s} {var:6s} {r['acc']:8.5f} {r['peak']:7.3f} {r['med']:7.3f} {r['w68']:12.3f} {r['fw']:12.3f} {r['fn']:12.3f}   n={r['n']}")
P("\n   m_est vs m_avg on the SAME events (pre + P cut): fraction in the narrow window if the window were placed on each variable's own peak (±6% around the peak)")
for M, s in S.items():
    m = s["valid"] & (s["asym"] < 0.1) & (s["ht"] > TRIG) & (s["psig"] >= PWP[M]); w = s["w"][m]
    r = []
    for var in ("mavg", "mest"):
        x = s[var][m] / M; pk = peak(x, w); r.append((pk, w[np.abs(x / pk - 1) < 0.06].sum() / w.sum()))
    P(f"   M={M}: m_avg peak {r[0][0]:.3f}, frac ±6% {r[0][1]:.3f};  m_est peak {r[1][0]:.3f}, frac ±6% {r[1][1]:.3f}")

P("\n== Item 4/5: drift with H_T/M (signal, pre + wide window): P efficiency at the working-point cut and at a common cut 0.997; m_avg median/M")
HB = [0, 2.0, 2.3, 2.6, 3.0, 3.5, 4.5, 99]
for M, s in S.items():
    pre = s["valid"] & (s["asym"] < 0.1) & (s["ht"] > TRIG); win = pre & (s["mavg"] / M >= .85) & (s["mavg"] / M < 1.1)
    P(f" M={M}  (P cut {PWP[M]})")
    for a, b in zip(HB[:-1], HB[1:]):
        hb = (s["ht"] / M >= a) & (s["ht"] / M < b)
        if (s["w"] * (win & hb)).sum() <= 0 or (win & hb).sum() < 30:
            continue
        e1, d1 = eff(s["w"], s["psig"] >= PWP[M], win & hb); e2, d2 = eff(s["w"], s["psig"] >= 0.997, win & hb)
        mm_ = pre & hb; md = wq(s["mavg"][mm_] / M, s["w"][mm_], 0.5)
        mp = pre & hb & (s["psig"] >= 0.99)
        mdp = wq(s["mavg"][mp] / M, s["w"][mp], 0.5) if mp.sum() > 20 else np.nan
        fr = s["w"][hb].sum() / s["w"].sum()
        R[f"ht_sig{M}_{a}"] = dict(frac=fr, eP=e1, dP=d1, e997=e2, d997=d2, med=md, medP=mdp)
        P(f"    H_T/M [{a:3.1f},{b:4.1f}): frac of events {fr:6.3f}  eff(P_wp) {e1:7.4f}±{d1:.4f}  eff(P>=0.997) {e2:7.4f}±{d2:.4f}   m_avg med/M all {md:.3f}, P>=0.99 {mdp:.3f}")

P("\n== Item 5: efficiency of a FIXED P cut vs true mass (signal; given pre + wide window; and absolute per produced event)")
for t in (0.99, 0.997, 0.999):
    r = []
    for M, s in S.items():
        win = s["valid"] & (s["asym"] < 0.1) & (s["ht"] > TRIG) & (s["mavg"] / M >= .85) & (s["mavg"] / M < 1.1)
        e, d = eff(s["w"], s["psig"] >= t, win); r.append(f"M{M}: {e:.4f}±{d:.4f} (abs {(s['w'] * (win & (s['psig'] >= t))).sum():.5f})"); R[f"effP{t}_{M}"] = (e, d)
    P(f"  P>={t}: " + "; ".join(r))

P("\n== Item 5: QCD m_avg sculpting (tail samples: valid, A<0.1, H_T>450; B in fb per 5% m_avg bin; ratio to P>=0.9)")
for tn, lo, hi, cuts in (("TvlmT", 300, 1120, (0.99, 0.997, 0.9985)), ("ThtT", 1150, 2500, (0.99, 0.998, 0.9993))):
    T = L(tn); w = T["w"] / T["frac"]; pre = T["valid"] & (T["asym"] < 0.1) & (T["ht"] > TRIG)
    edges = np.exp(np.arange(np.log(lo), np.log(hi), 0.05)); c = np.sqrt(edges[1:] * edges[:-1])
    H = {}
    for t in (0.9,) + cuts:
        m = pre & (T["psig"] >= t); h, _ = np.histogram(T["mavg"][m], edges, weights=w[m]); h2, _ = np.histogram(T["mavg"][m], edges, weights=w[m] ** 2)
        H[t] = (h, np.sqrt(h2))
    P(f" {tn}:  m_avg  " + "  ".join(f"B(P>={t})±" for t in H) + "   pass fraction of the tightest cut rel. to P>=0.9")
    for i in range(len(c)):
        P(f"   {c[i]:7.0f}  " + "  ".join(f"{H[t][0][i]:9.3g}±{H[t][1][i]:7.2g}" for t in H) + f"   {H[cuts[-1]][0][i] / max(H[0.9][0][i], 1e-30):.2e}")
    R[f"sculpt_{tn}"] = dict(c=c, **{str(t): H[t][0] for t in H}, **{f"e{t}": H[t][1] for t in H})
    # smoothness: fit log B vs log m with a quadratic for the tightest cuts, report chi2 and the largest local excess
    for t in cuts:
        h, e = H[t]; ok = h > 0
        if ok.sum() < 6:
            continue
        x = np.log(c[ok]); y = np.log(h[ok]); s = e[ok] / h[ok]
        co = np.polyfit(x, y, 2, w=1 / s); pull = (y - np.polyval(co, x)) / s
        P(f"   P>={t}: quadratic log-log fit chi2/ndf = {(pull ** 2).sum():.1f}/{ok.sum() - 3}; largest pull {pull.max():+.1f} at m_avg {c[ok][pull.argmax()]:.0f}, most negative {pull.min():+.1f} at {c[ok][pull.argmin()]:.0f}")
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for t in H:
        h, e = H[t]; ax.errorbar(c, h, e, fmt="o-", ms=2.5, lw=0.8, label=f"P ≥ {t}")
    ax.set_yscale("log"); ax.set_xscale("log"); ax.set_xlabel("m_avg [GeV]"); ax.set_ylabel("QCD [fb per 5% bin]")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(f"{D}/out/sculpt_{tn}.pdf"); plt.close(fig)

P("\n== where QCD sits relative to the window after the working-point selection (no window cut; tail samples; m/M quantiles 16/50/84% and fraction in window)")
for name, wp in WPS.items():
    M = wp["M"]; tn = QCD[M][0]; T = L(tn); w = T["w"] / T["frac"]
    wp2 = dict(wp, win=(QCD[M][2][0] / M * 1.02, QCD[M][2][1] / M * 0.98))
    m = aux(T, wp2) & (T["psig"] >= wp["psig"])
    if m.sum() < 5:
        P(f"  {name}: n={m.sum()} too few"); continue
    for var in ("mavg", "mest"):
        x = T[var][m] / M; q = wq(x, w[m], [0.16, 0.5, 0.84]); fw = w[m][(x >= wp["win"][0]) & (x < wp["win"][1])].sum() / w[m].sum()
        # local slope: B in the upper vs lower half of the window
        a, b = wp["win"]; mid = np.sqrt(a * b); up = w[m][(x >= mid) & (x < b)].sum(); dn = w[m][(x >= a) & (x < mid)].sum()
        P(f"  {name:16s} {var}: range [{wp2['win'][0]:.2f},{wp2['win'][1]:.2f}]M n={m.sum():5d}  q16/50/84 = {q[0]:.2f}/{q[1]:.2f}/{q[2]:.2f}  frac in window {fw:.3f}  upper/lower half of window {up / max(dn, 1e-30):.2f}")
json.dump(R, open(f"{D}/out/tab/item45.json", "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))
