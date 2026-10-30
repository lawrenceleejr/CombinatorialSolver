"""Compare the expected gluino-pair cross-section limits of xs_study.py with CMS-EXO-21-004 (arXiv:2404.02992).

CMS: resolved trijet search with data scouting, 128 fb^-1 at 13 TeV, HEPData Figure 4c (ins2774247). HEPData gives limits
on sigma x B x acceptance [pb] and the acceptance per mass, so sigma_95 = limit / acceptance. The table has three search
regions (200-500, 500-900, 900-2000 GeV); masses at a boundary appear twice, so each region is drawn as its own segment.
Naive 300 fb^-1 projection of the CMS expected limit: background-dominated scaling, x sqrt(128/300).
This study: limits recomputed at 128 fb^-1 from the stored signal efficiencies and background cross sections.
"""
import json, math, sys
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

style, xs_json, cms_yaml, out = sys.argv[1:5]
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")

L_CMS, L_HL = 128.0, 300.0
SCALE = math.sqrt(L_CMS / L_HL)
PB_TO_FB = 1e3

# ---------------- CMS
d = yaml.safe_load(open(cms_yaml))
cm = np.array([x["value"] for x in d["independent_variables"][0]["values"]], float)
col = {v["header"]["name"]: np.array([x["value"] for x in v["values"]], float) for v in d["dependent_variables"]}
acc = col["Acceptance"]
cms = {k: col[c] / acc * PB_TO_FB for k, c in (("exp", "Expected XSCN"), ("obs", "Observed XSCN"), ("p1", "+1 sigma Expected XSCN"),
                                               ("m1", "-1 sigma Expected XSCN"), ("p2", "+2 sigma Expected XSCN"), ("m2", "-2 sigma Expected XSCN"))}
cms["theory"] = col["SignalXscn_X_acceptance"] / acc * PB_TO_FB
cuts = [i for i in range(1, len(cm)) if cm[i] == cm[i - 1]]
segs = [s for s in np.split(np.arange(len(cm)), cuts) if cm[s].min() >= 500]  # 500-900 and 900-2000 GeV regions

# ---------------- this study
r = json.load(open(xs_json))
M = np.array(sorted(int(k) for k in r["rows"]))
sig_lo = np.array([r["sigma_LO_fb"][str(m)] for m in M])


def s_up(B):
    if B <= 0:
        return 3.0
    f = lambda s: 2 * (s - B * math.log1p(s / B)) - 1.96 ** 2
    return max(brentq(f, 1e-9, 1e3 + 50 * math.sqrt(B) + 10 * B), 3.0)


ours = {}
for key in ("argmin", "pairnet_A", "pairnet_AP"):
    for L in (L_CMS, L_HL):
        ours[key, L] = np.array([s_up(r["rows"][str(m)][key]["xsB_fb"] * L) / (L * r["rows"][str(m)][key]["effS"]) for m in M])


def reach(mass, lim, xs):
    """Largest mass where lim < xs, log-linear interpolation between the last excluded and the next point."""
    ex = np.where(lim < xs)[0]
    if len(ex) == 0:
        return None
    i = ex[-1]
    if i == len(mass) - 1:
        return float(mass[i])
    f = lambda x: math.log(np.interp(x, mass[i:i + 2], lim[i:i + 2])) - math.log(np.interp(x, mass[i:i + 2], xs[i:i + 2]))
    try:
        return brentq(f, mass[i], mass[i + 1])
    except ValueError:
        return float(mass[i])


hi = segs[-1]  # 900-2000 GeV region, where the reach lies
summary = {
    "cms_reach_vs_own_theory": {"exp_128": reach(cm[hi], cms["exp"][hi], cms["theory"][hi]), "obs_128": reach(cm[hi], cms["obs"][hi], cms["theory"][hi]),
                                "exp_300_naive": reach(cm[hi], cms["exp"][hi] * SCALE, cms["theory"][hi])},
    "ours_reach_vs_LO_13p6": {f"{k}_{int(L)}": reach(M, v, sig_lo) for (k, L), v in ours.items()},
    "ratio_cms_exp_over_ours": {},
}
for m in (1000, 1200, 1500, 1800, 2000):
    j = hi[cm[hi] == m][0]; i = int(np.where(M == m)[0][0])
    summary["ratio_cms_exp_over_ours"][m] = {
        "128": {k: float(cms["exp"][j] / ours[k, L_CMS][i]) for k in ("argmin", "pairnet_A", "pairnet_AP")},
        "300_naive": {k: float(cms["exp"][j] * SCALE / ours[k, L_HL][i]) for k in ("argmin", "pairnet_A", "pairnet_AP")},
        "cms_exp_fb": float(cms["exp"][j]), "cms_exp_300_fb": float(cms["exp"][j] * SCALE),
        "pairnet_AP_128_fb": float(ours["pairnet_AP", L_CMS][i]), "pairnet_AP_300_fb": float(ours["pairnet_AP", L_HL][i])}
summary["sigma_theory_fb"] = {"cms_13TeV": {int(a): float(b) for a, b in zip(cm[hi], cms["theory"][hi])},
                              "pythia_LO_13p6TeV": {int(a): float(b) for a, b in zip(M, sig_lo)}}
json.dump(summary, open(f"{out}/cms_compare.json", "w"), indent=1)
print(json.dumps({k: summary[k] for k in ("cms_reach_vs_own_theory", "ours_reach_vs_LO_13p6")}, indent=1))
for m, v in summary["ratio_cms_exp_over_ours"].items():
    print(m, {k: {kk: round(vv, 2) for kk, vv in x.items()} if isinstance(x, dict) else round(x, 3) for k, x in v.items()})

# ---------------- plots: one at 128 fb^-1 (like for like), one at 300 fb^-1 with the naive CMS projection
CMS_C, CMS_P = "#2b5d8a", "#7aa6c9"
OURS = (("argmin", r"this study: argmin $|m_1-m_2|$, $A<0.1$", "0.5", "--"), ("pairnet_A", r"this study: PairNet, $A<0.1$", "#c46a5f", "--"),
        ("pairnet_AP", r"this study: PairNet, $A<0.1$ + $P$ cut", "#9b2c2c", "-"))


def draw(L, fname):
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    for n, s in enumerate(segs):
        lab = (lambda t: t if n == 0 else None)
        if L == L_HL:  # naive projection with its scaled +-1 sigma band; the published expected curve for reference
            ax.fill_between(cm[s], cms["m1"][s] * SCALE, cms["p1"][s] * SCALE, color=CMS_P, alpha=0.25, lw=0,
                            label=lab(r"CMS expected $\pm1\sigma$ $\times\sqrt{128/300}$ (naive 300 fb$^{-1}$)"))
            ax.plot(cm[s], cms["exp"][s] * SCALE, color=CMS_C, lw=1.6, ls="-", label=lab(r"CMS expected $\times\sqrt{128/300}$ (naive 300 fb$^{-1}$)"))
            ax.plot(cm[s], cms["exp"][s], color=CMS_C, lw=1.0, ls=":", label=lab(r"CMS expected, published 128 fb$^{-1}$"))
        else:
            ax.fill_between(cm[s], cms["m1"][s], cms["p1"][s], color=CMS_P, alpha=0.25, lw=0, label=lab(r"CMS expected $\pm1\sigma$, 128 fb$^{-1}$"))
            ax.plot(cm[s], cms["exp"][s], color=CMS_C, lw=1.4, ls=":", label=lab(r"CMS expected, 128 fb$^{-1}$"))
            ax.plot(cm[s], cms["obs"][s], color=CMS_C, lw=0.8, marker="o", ms=2.2, label=lab(r"CMS observed, 128 fb$^{-1}$"))
    for key, lab, c, ls in OURS:
        ax.plot(M, ours[key, L], color=c, lw=1.3, ls=ls, label=lab + f", {int(L)} fb$^{{-1}}$")
    ax.plot(M, sig_lo, color="0.1", lw=1.4, label=r"$\sigma(\tilde g\tilde g)$, Pythia LO, 13.6 TeV")
    ax.plot(cm[hi], cms["theory"][hi], color="0.1", lw=1.0, ls="-.", label=r"$\sigma(\tilde g\tilde g)$ used by CMS, NLO+NLL, 13 TeV")
    ax.set_yscale("log"); ax.set_xlim(480, 2450)
    ax.set_xlabel(r"$m_{\tilde g}$ [GeV]"); ax.set_ylabel(r"95% CL upper limit on $\sigma(pp\to\tilde g\tilde g)$ [fb]")
    ax.set_title(("CMS-EXO-21-004 (128 fb$^{-1}$) vs this study at %d fb$^{-1}$; this study: median expected, no systematics" % L),
                 fontsize=7, loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6.8, frameon=False)
    spines_to_data_range(ax)
    save_pdf(fig, f"{out}/{fname}.pdf"); fig.savefig(f"{out}/{fname}.png", dpi=170, bbox_inches="tight")


draw(L_HL, "xs_limits_cms_300")
draw(L_CMS, "xs_limits_cms_128")
