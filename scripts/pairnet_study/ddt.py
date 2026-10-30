"""m_avg-dependent P(signal) threshold at fixed QCD efficiency ("DDT"-style), PairNet pairing, A < 0.1.

t_eps(m): the (1 - eps) cross-section-weighted quantile of P(signal) for QCD in log-spaced m_avg bins, smoothed
(running median over 3 bins) and interpolated in log m. Derived on one half of the merged QCD, applied to the other.
Checks: (1) the QCD m_avg spectrum after the cut divided by eps x the spectrum before (should be flat at 1);
(2) the dijet-function fit of the QCD spectrum after the cut; (3) expected limits per mass with the best eps per mass
(tuned on one half, evaluated on the other), against the per-mass tuned P cut of cuts_study3.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

style, big, ht_npy, big_ht, xs_json, cs3_json, out = sys.argv[1:8]
sys.path.insert(0, style); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
import qcdmerge  # noqa: E402
from limits import s_up  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")
L = 300.0
FSYS = float(sys.argv[8]) if len(sys.argv) > 8 else 0.0

d, info = qcdmerge.load(big, ht_npy, big_ht, ["pn_m1", "pn_m2", "psig"])
qm = 0.5 * (d["pn_m1"] + d["pn_m2"]); qA = np.abs(d["pn_m1"] - d["pn_m2"]) / np.maximum(d["pn_m1"] + d["pn_m2"], 1e-9)
qb = qA < 0.1
qm, qp, qw = qm[qb], d["psig"][qb], d["wfb"][qb]
qh = np.arange(len(qm)) % 2
v = np.load(big)
sg = v["is_signal"]
sm = 0.5 * (v["pn_m1"][sg] + v["pn_m2"][sg]); sA = np.abs(v["pn_m1"][sg] - v["pn_m2"][sg]) / (v["pn_m1"][sg] + v["pn_m2"][sg])
sp, spm = v["psig"][sg], v["pmass"][sg]
sh = np.arange(len(sm)) % 2
filt = {int(k): x for k, x in json.load(open(xs_json))["filter_eff"].items()}
EDGES = np.exp(np.linspace(np.log(300), np.log(4000), 41)); LC = 0.5 * (np.log(EDGES[1:]) + np.log(EDGES[:-1]))
EPS = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 3e-4, 1e-4]


def wquantile(x, w, qq):
    o = np.argsort(x); c = np.cumsum(w[o]); c /= c[-1]
    return np.interp(qq, c, x[o])


def threshold(eps, h):
    m = qh == h
    t = np.full(len(LC), np.nan)
    for i in range(len(LC)):
        b = m & (qm >= EDGES[i]) & (qm < EDGES[i + 1])
        if b.sum() >= max(50, int(5 / eps)):  # >= 5 MC events expected above the threshold
            t[i] = wquantile(qp[b], qw[b], 1 - eps)
    ok = ~np.isnan(t)
    if ok.sum() < 3:
        return None
    t = np.interp(LC, LC[ok], t[ok])  # flat extrapolation outside the populated range
    t = np.array([np.median(t[max(0, i - 1):i + 2]) for i in range(len(t))])
    return lambda mm: np.interp(np.log(np.maximum(mm, 1)), LC, t)


T = {(e, h): threshold(e, h) for e in EPS for h in (0, 1)}
EPS = [e for e in EPS if T[(e, 0)] is not None and T[(e, 1)] is not None]
print("usable QCD efficiencies:", EPS)

# (1)+(2): closure and shape on the half not used to derive the threshold
bins = np.arange(500, 3600, 50.0); ctr = 0.5 * (bins[1:] + bins[:-1])
fig, ax = plt.subplots(figsize=(4.8, 3.0))
closure = {}
cb = np.arange(500, 3600, 100.0); cc = 0.5 * (cb[1:] + cb[:-1])
for e, c in ((0.03, "#c9a227"), (0.01, "#5f8a5f"), (0.003, "#9b2c2c")):
    num = den = 0
    for h in (0, 1):
        m = qh == 1 - h
        pas = m & (qp >= T[(e, h)](qm))
        n1, _ = np.histogram(qm[pas], cb, weights=qw[pas]); n0, _ = np.histogram(qm[m], cb, weights=qw[m])
        num = num + n1; den = den + n0
    r = num / np.maximum(den * e, 1e-30)
    ax.plot(cc, r, color=c, lw=1.2, drawstyle="steps-mid", label=f"QCD efficiency {e:g}")
    closure[e] = {"median_ratio": float(np.median(r[den > 0])), "spread_16_84": [float(x) for x in np.quantile(r[den > 0], [.16, .84])]}
ax.axhline(1, color="0.6", lw=0.6); ax.set_ylim(0, 2.5)
ax.set_xlabel(r"$m_{\rm avg}$ [GeV]"); ax.set_ylabel(r"QCD after cut / ($\varepsilon\times$ before)")
ax.set_title("m$_{\\rm avg}$-dependent P(signal) threshold: derived on one half of the QCD, applied to the other", fontsize=6.5, loc="left")
ax.legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
spines_to_data_range(ax)
save_pdf(fig, f"{out}/ddt_closure.pdf"); fig.savefig(f"{out}/ddt_closure.png", dpi=170, bbox_inches="tight")

# (3) limits per mass: best eps tuned on one half, evaluated on the other
cs3 = json.load(open(cs3_json))
res = {}
for M in range(500, 2401, 100):
    sM = spm == M
    lims = []
    for h in (0, 1):
        def lim(e, hh):
            """Threshold derived on half h (the tuning half); limit computed on half hh."""
            t = T[(e, h)]
            sw = sM & (sh == hh) & (sA < 0.1) & (sm > 0.85 * M) & (sm < 1.1 * M) & (sp >= t(sm))
            eS = sw.sum() / (sM & (sh == hh)).sum() * filt[M]
            qq = (qh == hh) & (qm > 0.85 * M) & (qm < 1.1 * M) & (qp >= t(qm))
            B = 2 * qw[qq].sum() * L
            return (s_up(B, FSYS) / (L * eS) if eS > 0 else np.inf), int(qq.sum())
        cand = [(lim(e, h)[0], e) for e in EPS if lim(e, h)[1] >= 10]
        if not cand:
            lims.append(np.inf); continue
        best_e = min(cand)[1]
        lims.append(lim(best_e, 1 - h)[0])
    r = {"ddt_limit_fb": float(np.exp(np.mean(np.log(lims)))), "tuned_P_limit_fb": cs3[str(M)]["pn"]["limit_fb"]}
    res[M] = r
    print(M, {k: round(x, 3) for k, x in r.items()}, "ratio DDT/tuned %.2f" % (r["ddt_limit_fb"] / r["tuned_P_limit_fb"]), flush=True)
json.dump({"closure": closure, "limits": res, "fsys": FSYS}, open(f"{out}/ddt_f{FSYS:g}.json", "w"), indent=1)

# spectrum after DDT cuts with dijet fits (both halves, each with the threshold from the other)
fig, ax = plt.subplots(figsize=(4.8, 3.2))
for e, c in ((0.03, "#c9a227"), (0.01, "#5f8a5f"), (0.003, "#9b2c2c")):
    y = e2 = n = 0
    for h in (0, 1):
        m = qh == 1 - h
        pas = m & (qp >= T[(e, h)](qm))
        y = y + np.histogram(qm[pas], bins, weights=qw[pas] * L)[0]; e2 = e2 + np.histogram(qm[pas], bins, weights=(qw[pas] * L) ** 2)[0]
        n = n + np.histogram(qm[pas], bins)[0]
    x = ctr / 13600.0; i0 = int(np.argmax(y)); ok = (np.arange(len(y)) >= i0) & (n >= 3)
    f = lambda p: (np.exp(p[0]) * (1 - x[ok]) ** p[1] / x[ok] ** (p[2] + p[3] * np.log(x[ok])) - y[ok]) / np.sqrt(e2[ok])
    fit = least_squares(f, [np.log(max(y[ok][0], 1e-9) * x[ok][0] ** 5), 10.0, 5.0, 0.0], method="lm", max_nfev=20000)
    chi2 = float((fit.fun ** 2).sum()); ndf = int(ok.sum() - 4)
    closure[e]["dijet_chi2_ndf"] = [chi2, ndf]
    ax.stairs(y / np.diff(bins), bins, color=c, lw=1.2, label=f"QCD efficiency {e:g} ($\\chi^2$/ndf {chi2:.0f}/{ndf})")
ax.set_yscale("log"); ax.set_xlabel(r"$m_{\rm avg}$ of the PairNet pairing [GeV]"); ax.set_ylabel(r"QCD events / GeV, 300 fb$^{-1}$")
ax.set_title("QCD after $A<0.1$ and the m$_{\\rm avg}$-dependent P(signal) cut", fontsize=6.5, loc="left")
ax.legend(fontsize=6.3, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
spines_to_data_range(ax)
save_pdf(fig, f"{out}/ddt_spectra.pdf"); fig.savefig(f"{out}/ddt_spectra.png", dpi=170, bbox_inches="tight")
json.dump({"closure": closure, "limits": res, "fsys": FSYS}, open(f"{out}/ddt_f{FSYS:g}.json", "w"), indent=1)
print(json.dumps(closure, indent=1))
