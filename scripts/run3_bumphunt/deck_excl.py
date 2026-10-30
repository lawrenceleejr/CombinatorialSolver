"""Deck sensitivity plot: PairNet r14A mass-window search (10% on B) at 300 fb^-1, 13.6 TeV, against what CMS Run 2
already excludes (CMS-EXO-21-004 observed, 128 fb^-1, unscaled, with its own 13 TeV signal cross section).
usage: deck_excl.py style out window10.json cms_fig4c.yaml [known_low.json known_high.json]"""
import json, sys
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

style, out, wj, cy = sys.argv[1:5]
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")
w = json.load(open(wj))["curves"]
d = yaml.safe_load(open(cy))
cm = np.array([x["value"] for x in d["independent_variables"][0]["values"]], float)
col = {v["header"]["name"]: np.array([x["value"] for x in v["values"]], float) for v in d["dependent_variables"]}
A = col["Acceptance"]; obs = col["Observed XSCN"] / A * 1e3; exp = col["Expected XSCN"] / A * 1e3; th13 = col["SignalXscn_X_acceptance"] / A * 1e3
segs = np.split(np.arange(len(cm)), [i for i in range(1, len(cm)) if cm[i] == cm[i - 1]])
# CMS observed exclusion edge (last crossing of observed limit and the CMS signal cross section), log interpolation
r = obs / th13; o = np.argsort(cm, kind="stable"); mo, ro = cm[o], r[o]
edge = None
for i in range(len(mo) - 1):
    if ro[i] < 1 <= ro[i + 1]:
        edge = mo[i] + (mo[i + 1] - mo[i]) * np.log(1 / ro[i]) / np.log(ro[i + 1] / ro[i])
summary = {"cms_observed_excluded_GeV": [float(mo[0]), float(edge)]}

fig, ax = plt.subplots(figsize=(5.0, 3.4))
ax.axvspan(mo[0], edge, color="#2b5d8a", alpha=0.07, lw=0)
ax.text(edge - 30, 0.25, f"excluded by CMS Run 2 (observed): {mo[0]:.0f} to {edge:.0f} GeV", fontsize=6.3, color="#2b5d8a", va="bottom", ha="right")
sig = w["PairNet, 10% on B"]
mm = np.array(sorted(int(m) for m in sig)); st = np.array([sig[str(m)]["sigma_th_fb"] for m in mm])
ax.plot(mm[mm >= 500], st[mm >= 500], color="0.1", lw=1.4, label=r"$\sigma(\tilde g\tilde g)$ NNLO$_{\rm approx}$+NNLL, 13.6 TeV")
ax.plot(mm[mm <= 500], st[mm <= 500], color="0.1", lw=1.2, ls="--", label=r"  below 500 GeV: LO $\times$ 2.07")
for n, s in enumerate(segs):
    lab = (lambda t: t if n == 0 else None)
    ax.plot(cm[s], obs[s], color="#2b5d8a", lw=1.5, label=lab(r"CMS-EXO-21-004 observed, 128 fb$^{-1}$ (13 TeV)"))
    ax.plot(cm[s], exp[s] * np.sqrt(128 / 300), color="#2b5d8a", lw=0.9, ls="--", label=lab(r"CMS expected $\times\sqrt{128/300}$ (naive 300 fb$^{-1}$)"))
    ax.plot(cm[s], th13[s], color="0.5", lw=0.8, ls="-.", label=lab(r"$\sigma$ used by CMS, 13 TeV"))
STY = [("PairNet, 10% on B, P categories", "#9b2c2c", "-", 2.2, "PairNet r14A, 10% on B, P categories"),
       ("PairNet, 10% on B", "#c26a5a", "-", 1.1, "PairNet r14A, 10% on B, one category")]
for key, c, ls, lw, lab in STY:
    r_ = w[key]; M = np.array(sorted(int(m) for m in r_ if np.isfinite(r_[m]["limit_fb"]))); y = np.array([r_[str(m)]["limit_fb"] for m in M])
    ax.plot(M, y, color=c, ls=ls, lw=lw, marker="o" if "categories" in key else None, ms=2.3, label=lab)
if len(sys.argv) > 6:
    kl = json.load(open(sys.argv[5]))["results"]; kh = json.load(open(sys.argv[6]))["results"]
    kn = {int(m): min(x[k]["limit_fb"] for k in ("P", "P+boost", "P+all") if k in x) for m, x in kl.items()}
    kn.update({int(m): x["limit_fb"] for m, x in kh.items() if int(m) > 1000})
    M = np.array(sorted(m for m in kn if np.isfinite(kn[m]) and m >= 250)); y = np.array([kn[m] for m in M])
    ax.plot(M, y, color="#9b2c2c", ls=":", lw=1.1, label="PairNet r14A, B known exactly")
    summary["known"] = {int(m): float(v) for m, v in zip(M, y)}
ax.set_yscale("log"); ax.set_xlim(180, 2450)
ax.set_xlabel(r"$m_{\tilde g}$ [GeV]"); ax.set_ylabel(r"95% CL limit on $\sigma$ [fb]")
ax.set_title(r"mass window [0.85, 1.1] M, offline $H_T>450$ GeV; PairNet: expected, 300 fb$^{-1}$, 13.6 TeV", fontsize=6.3, loc="left")
ax.legend(fontsize=5.7, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
spines_to_data_range(ax)
save_pdf(fig, f"{out}/deck_excl.pdf"); fig.savefig(f"{out}/deck_excl.png", dpi=170, bbox_inches="tight")
json.dump(summary, open(f"{out}/deck_excl.json", "w"), indent=1)
print(summary["cms_observed_excluded_GeV"])
