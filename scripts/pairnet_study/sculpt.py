"""Does the P(signal) cut sculpt the QCD m_avg spectrum? PairNet pairing, A < 0.1, fixed P thresholds.

Each spectrum (merged QCD, cross-section weighted, 300 fb^-1 yields) is fitted above m_min with the 4-parameter dijet
function f(x) = p0 (1-x)^p1 / x^(p2 + p3 ln x), x = m_avg / 13.6 TeV, by weighted least squares using the MC
uncertainty per bin. Outputs: one spectrum plot (all thresholds) and one pull plot per threshold, plus chi2/ndf.
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

style, big, ht_npy, big_ht, out = sys.argv[1:6]
sys.path.insert(0, style); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
import qcdmerge  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")

L = 300.0
d, info = qcdmerge.load(big, ht_npy, big_ht, ["pn_m1", "pn_m2", "psig"])
print(json.dumps(info, indent=1, default=float))
mavg = 0.5 * (d["pn_m1"] + d["pn_m2"]); A = np.abs(d["pn_m1"] - d["pn_m2"]) / np.maximum(d["pn_m1"] + d["pn_m2"], 1e-9)
base = A < 0.1
bins = np.arange(500, 3600, 50.0)
ctr = 0.5 * (bins[1:] + bins[:-1]); x = ctr / 13600.0
THR = [(None, "no P cut", "0.55"), (0.9, "$P>0.9$", "#c9a227"), (0.99, "$P>0.99$", "#5f8a5f"), (0.998, "$P>0.998$", "#4a87a8"), (0.9995, "$P>0.9995$", "#9b2c2c")]


def dijet(p, x):
    return np.exp(p[0]) * (1 - x) ** p[1] / x ** (p[2] + p[3] * np.log(x))


res = {}
fig, ax = plt.subplots(figsize=(4.8, 3.2))
for t, lab, c in THR:
    sel = base & ((d["psig"] >= t) if t is not None else True)
    y, _ = np.histogram(mavg[sel], bins, weights=d["wfb"][sel] * L)
    e2, _ = np.histogram(mavg[sel], bins, weights=(d["wfb"][sel] * L) ** 2)
    n, _ = np.histogram(mavg[sel], bins)
    # fit from the spectrum peak upward, bins with >= 3 MC events
    i0 = int(np.argmax(y)) + 1
    ok = (np.arange(len(y)) >= i0) & (n >= 3)
    if ok.sum() < 8:
        continue
    err = np.sqrt(e2[ok])
    f = lambda p: (dijet(p, x[ok]) - y[ok]) / err
    p0 = [np.log(max(y[ok][0], 1e-9) * x[ok][0] ** 5), 10.0, 5.0, 0.0]
    fit = least_squares(f, p0, method="lm", max_nfev=20000)
    chi2 = float((fit.fun ** 2).sum()); ndf = int(ok.sum() - 4)
    res[lab] = {"chi2": chi2, "ndf": ndf, "fit_from_GeV": float(bins[i0]), "nMC": int(sel.sum()), "events_300fb": float(y.sum())}
    ax.stairs(y / np.diff(bins), bins, color=c, lw=1.2, label=f"{lab}  ($\\chi^2$/ndf {chi2:.0f}/{ndf})")
    ax.plot(ctr[ok], dijet(fit.x, x[ok]) / 50.0, color=c, lw=0.6, ls=":")
    # pulls
    fg, a2 = plt.subplots(figsize=(4.6, 2.2))
    pull = (y[ok] - dijet(fit.x, x[ok])) / err
    a2.axhline(0, color="0.6", lw=0.6)
    a2.bar(ctr[ok], pull, width=40, color=c)
    a2.set_ylim(-4, 4); a2.set_xlabel(r"$m_{\rm avg}$ [GeV]"); a2.set_ylabel(r"(MC $-$ fit) / MC unc.")
    a2.set_title(f"QCD, PairNet $A<0.1$, {lab}: dijet-function fit, $\\chi^2$/ndf = {chi2:.0f}/{ndf}", fontsize=6.5, loc="left")
    spines_to_data_range(a2)
    tag = "none" if t is None else str(t)
    save_pdf(fg, f"{out}/sculpt_pull_{tag}.pdf"); fg.savefig(f"{out}/sculpt_pull_{tag}.png", dpi=170, bbox_inches="tight")
    plt.close(fg)
ax.set_yscale("log"); ax.set_xlabel(r"$m_{\rm avg}$ of the PairNet pairing [GeV]"); ax.set_ylabel(r"QCD events / GeV, 300 fb$^{-1}$")
ax.set_title("QCD after $A<0.1$ and fixed P(signal) cuts; dotted: 4-parameter dijet fits", fontsize=6.5, loc="left")
ax.legend(fontsize=6.3, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
spines_to_data_range(ax)
save_pdf(fig, f"{out}/sculpt_spectra.pdf"); fig.savefig(f"{out}/sculpt_spectra.png", dpi=170, bbox_inches="tight")
json.dump({"merge": info, "fits": res}, open(f"{out}/sculpt.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1))
