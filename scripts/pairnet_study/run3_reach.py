"""Run 3 expected reach: limits per selection against the 13.6 TeV NNLOapprox+NNLL gluino-pair cross section.

Inputs: a json {M: {selection: limit_fb at 300 fb^-1}} (e.g. quick_ht_gain.json), the NNLL-fast json, cms_compare.json.
The CMS curve is the CMS-EXO-21-004 expected limit (13 TeV, 128 fb^-1) x sqrt(128/300): a naive Run 3 projection that
ignores the 13 -> 13.6 TeV change of the background (+10-20%) and of the acceptance.
Reach = largest mass with limit < sigma (log-linear interpolation), also against sigma -1 std (theory band).
"""
import json, math, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

style, lim_json, nnll_json, cms_json, out, tag = sys.argv[1:7]
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")

th = json.load(open(nnll_json))["data"]
tm = np.array(sorted(int(float(k)) for k in th)); tx = np.array([th[str(m)]["xsec_pb"] * 1e3 for m in tm]); tu = np.array([th[str(m)]["unc_pb"] * 1e3 for m in tm])
sig = lambda m: np.exp(np.interp(m, tm, np.log(tx)))
sig_lo = lambda m: np.exp(np.interp(m, tm, np.log(tx - tu)))
lims = json.load(open(lim_json))
M = np.array(sorted(int(k) for k in lims))
cms = json.load(open(cms_json))
cm = np.array(sorted(int(k) for k in cms["ratio_cms_exp_over_ours"]))
cl = np.array([cms["ratio_cms_exp_over_ours"][str(k)]["cms_exp_300_fb"] for k in cm])


def reach(m, lim, xs):
    r = np.log(lim) - np.log(xs(m))
    ex = np.where(r < 0)[0]
    if len(ex) == 0:
        return None
    i = ex[-1]
    if i == len(m) - 1:
        return float(m[i])
    return float(m[i] + (m[i + 1] - m[i]) * (-r[i]) / (r[i + 1] - r[i]))


SEL = json.loads(sys.argv[7]) if len(sys.argv) > 7 else [["am", "argmin, $A<0.1$", "0.6", "--"], ["am+HT", "argmin + $H_T$ cut", "0.3", "-"],
                                                         ["pn", "PairNet + P cut", "#9b2c2c", "-"]]
summary = {}
fig, ax = plt.subplots(figsize=(5.0, 3.4))
mm = np.linspace(500, 2600, 200)
ax.fill_between(mm, sig_lo(mm), np.exp(np.interp(mm, tm, np.log(tx + tu))), color="0.85", lw=0)
ax.plot(mm, sig(mm), color="0.1", lw=1.4, label=r"$\sigma(\tilde g\tilde g)$ NNLO$_{\rm approx}$+NNLL, 13.6 TeV")
for k, lab, c, ls in SEL:
    y = np.array([lims[str(m)][k] for m in M])
    ax.plot(M, y, color=c, ls=ls, lw=1.4, label=lab)
    summary[k] = {"reach_GeV": reach(M, y, sig), "reach_minus1sd_GeV": reach(M, y, sig_lo)}
ax.plot(cm, cl, color="#2b5d8a", lw=1.3, marker="o", ms=2.5, label=r"CMS-EXO-21-004 exp. $\times\sqrt{128/300}$ (13 TeV)")
summary["cms_naive_300"] = {"reach_GeV": reach(cm, cl, sig), "reach_minus1sd_GeV": reach(cm, cl, sig_lo)}
ax.set_yscale("log"); ax.set_xlim(480, 2650)
ax.set_xlabel(r"$m_{\tilde g}$ [GeV]"); ax.set_ylabel(r"expected 95% CL limit on $\sigma$ [fb], 300 fb$^{-1}$")
fs = "no systematics" if tag == "f0" else "5% background uncertainty" if tag == "f0.05" else tag
ax.set_title(f"300 fb$^{{-1}}$ at 13.6 TeV; this study: {fs}; CMS: naive luminosity scaling", fontsize=6.5, loc="left")
ax.legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
spines_to_data_range(ax)
save_pdf(fig, f"{out}/run3_reach_{tag}.pdf"); fig.savefig(f"{out}/run3_reach_{tag}.png", dpi=170, bbox_inches="tight")
json.dump(summary, open(f"{out}/run3_reach_{tag}.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
