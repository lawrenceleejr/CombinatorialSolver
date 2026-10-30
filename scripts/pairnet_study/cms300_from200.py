"""Slide 15 replacement: naive CMS-EXO-21-004 projection to 300 fb^-1 from 200 GeV, against this study at 300 fb^-1.

CMS: HEPData Fig. 4c (all three resolved regions, 200-2000 GeV), sigma limit = (sigma x B x A limit) / A, expected x
sqrt(128/300) with the scaled +-1 sigma band; the published 128 fb^-1 expected curve for reference; CMS signal sigma.
This study (300 fb^-1, no systematics, held-out MC): PairNet r14A (preliminary) best selection per mass with offline
H_T > 450 GeV up to 1 TeV and the per-mass P cut above; argmin + tuned cuts (low-mass study up to 1 TeV, the full-QCD
cut study above). Theory at 13.6 TeV: NNLOapprox+NNLL from 500 GeV, Pythia LO x 2.07 below.
"""
import json, os, sys
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

style, cms_yaml, excl_json, low_json, cuts_json, out = sys.argv[1:7]
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")
SC = np.sqrt(128 / 300)
d = yaml.safe_load(open(cms_yaml))
cm = np.array([x["value"] for x in d["independent_variables"][0]["values"]], float)
col = {v["header"]["name"]: np.array([x["value"] for x in v["values"]], float) for v in d["dependent_variables"]}
acc = col["Acceptance"]
cms = {k: col[c] / acc * 1e3 for k, c in (("exp", "Expected XSCN"), ("p1", "+1 sigma Expected XSCN"), ("m1", "-1 sigma Expected XSCN"))}
cms_th = col["SignalXscn_X_acceptance"] / acc * 1e3
segs = np.split(np.arange(len(cm)), [i for i in range(1, len(cm)) if cm[i] == cm[i - 1]])

ex = json.load(open(excl_json))
MODEL = os.environ.get("MODEL", "r14Ap"); FK = os.environ.get("FK", "f0")
rows = ex["rows"][f"{MODEL}_{FK}"]
pm = np.array(sorted(int(m) for m in rows)); pl = np.array([rows[str(m)]["limit_fb"] for m in pm])
low = json.load(open(low_json))["results"]
cuts = json.load(open(cuts_json))
am_m = sorted([int(m) for m in low] + ([int(m) for m in cuts if int(m) > 1000] if FK == "f0" else []))
am_l = [low[str(m)]["am+all"]["limit_fb"] if m <= 1000 else cuts[str(m)]["am+all"]["limit_fb"] for m in am_m]
SYS = {"f0": "no systematics", "f0.05": "5% background uncertainty", "frun2": "Run-2-like background uncertainty"}[FK]
th = {int(m): rows[m]["sigma_th_fb"] for m in rows}

fig, ax = plt.subplots(figsize=(5.0, 3.4))
tm = np.array(sorted(th)); tv = np.array([th[m] for m in tm])
ax.plot(tm[tm >= 500], tv[tm >= 500], color="0.1", lw=1.4, label=r"$\sigma(\tilde g\tilde g)$ NNLO$_{\rm approx}$+NNLL, 13.6 TeV")
ax.plot(tm[tm <= 500], tv[tm <= 500], color="0.1", lw=1.2, ls="--", label=r"  below 500 GeV: LO $\times$ 2.07")
for n, s in enumerate(segs):
    lab = (lambda t: t if n == 0 else None)
    ax.fill_between(cm[s], cms["m1"][s] * SC, cms["p1"][s] * SC, color="#7aa6c9", alpha=0.25, lw=0, label=lab(r"CMS expected $\pm1\sigma$ $\times\sqrt{128/300}$"))
    ax.plot(cm[s], cms["exp"][s] * SC, color="#2b5d8a", lw=1.6, label=lab(r"CMS-EXO-21-004 expected $\times\sqrt{128/300}$"))
    ax.plot(cm[s], cms["exp"][s], color="#2b5d8a", lw=0.9, ls=":", label=lab(r"CMS expected, published 128 fb$^{-1}$"))
    ax.plot(cm[s], cms_th[s], color="0.45", lw=0.9, ls="-.", label=lab(r"$\sigma(\tilde g\tilde g)$ used by CMS, 13 TeV"))
ax.plot(am_m, am_l, color="0.35", lw=1.3, ls="--", label=r"this study: argmin + tuned cuts")
ax.plot(pm, pl, color="#9b2c2c", lw=1.6, label=f"this study: PairNet ({MODEL})")
ax.set_yscale("log"); ax.set_xlim(180, 2450)
ax.set_xlabel(r"$m_{\tilde g}$ [GeV]"); ax.set_ylabel(r"expected 95% CL limit on $\sigma$ [fb], 300 fb$^{-1}$")
ax.set_title(f"300 fb$^{{-1}}$; this study: 13.6 TeV, {SYS}, offline $H_T>450$ GeV below 1 TeV", fontsize=6.5, loc="left")
ax.legend(fontsize=6.2, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
spines_to_data_range(ax)
save_pdf(fig, f"{out}/cms300_from200_{MODEL}_{FK}.pdf"); fig.savefig(f"{out}/cms300_from200_{MODEL}_{FK}.png", dpi=170, bbox_inches="tight")
for m in (200, 300, 400, 500, 700, 1000, 1500, 2000):
    j = [i for i in range(len(cm)) if cm[i] == m]
    c = min(cms["exp"][j] * SC) if j else np.nan
    p = pl[pm == m][0] if (pm == m).any() else np.nan
    print(m, "CMS naive 300: %.3g fb" % c, "PairNet: %.3g fb" % p, "ratio CMS/PairNet %.2f" % (c / p))
