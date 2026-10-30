"""Limit comparison around 700 GeV: the 10% mass-window search (P categories) before and after the targeted MC and the
narrower window, with the CMS Run-2 observed limit, its naive 300 fb^-1 projection and the signal cross section.
usage: compare700.py style out window10.json cms_fig4c.yaml label=dir_or_glob[@dirs...] ...
  each label maps to one or more result directories (lowmass_eval_*_f0.1.json); per mass the best over files is taken."""
import glob, json, sys
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

style, out, wj, cy = sys.argv[1:5]; runs = [a.split("=", 1) for a in sys.argv[5:]]
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")
w = json.load(open(wj))["curves"]
d = yaml.safe_load(open(cy))
cm = np.array([x["value"] for x in d["independent_variables"][0]["values"]], float)
col = {v["header"]["name"]: np.array([x["value"] for x in v["values"]], float) for v in d["dependent_variables"]}
A = col["Acceptance"]; obs = col["Observed XSCN"] / A * 1e3; exp = col["Expected XSCN"] / A * 1e3
segs = np.split(np.arange(len(cm)), [i for i in range(1, len(cm)) if cm[i] == cm[i - 1]])
XLO, XHI = 450, 1100


def cats_best(dirs):
    out = {}
    for dr in dirs.split("@"):
        for f in glob.glob(f"{dr}/lowmass_eval_*_f*.json"):
            for m, x in json.load(open(f))["results"].items():
                v = min([x[k]["limit_fb"] for k in x if k.endswith("+cats") and isinstance(x[k], dict)] + [np.inf])
                if np.isfinite(v):
                    out[int(m)] = min(out.get(int(m), np.inf), v)
    return out


fig, ax = plt.subplots(figsize=(5.0, 3.4))
sig = w["PairNet, 10% on B"]; mm = np.array(sorted(int(m) for m in sig)); st = np.array([sig[str(m)]["sigma_th_fb"] for m in mm])
k = (mm >= XLO) & (mm <= XHI)
ax.plot(mm[k], st[k], color="0.1", lw=1.4, label=r"$\sigma(\tilde g\tilde g)$ NNLO$_{\rm approx}$+NNLL, 13.6 TeV")
for n, s in enumerate(segs):
    lab = (lambda t: t if n == 0 else None); ks = (cm[s] >= XLO - 60) & (cm[s] <= XHI + 60)
    ax.plot(cm[s][ks], obs[s][ks], color="#2b5d8a", lw=1.5, label=lab(r"CMS-EXO-21-004 observed, 128 fb$^{-1}$ (13 TeV)"))
    ax.plot(cm[s][ks], exp[s][ks] * np.sqrt(128 / 300), color="#2b5d8a", lw=0.9, ls="--", label=lab(r"CMS expected $\times\sqrt{128/300}$"))
kn = w.get("PairNet, B known exactly")
base = w["PairNet, 10% on B, P categories"]
bm = np.array([m for m in mm if XLO <= m <= XHI and str(m) in base]); ax.plot(bm, [base[str(m)]["limit_fb"] for m in bm], color="0.55", lw=1.3, marker="o", ms=2.5, label="PairNet, 10% on B: before")
STY = [("#c26a5a", "-", 1.4), ("#9b2c2c", "-", 2.2), ("#c26a5a", "--", 1.4), ("#9b2c2c", "--", 2.2)]
summary = {"before": {int(m): base[str(m)]["limit_fb"] for m in bm}}
for (lab, dirs), (c, ls, lw) in zip(runs, STY):
    r = cats_best(dirs); M = np.array(sorted(m for m in r if XLO <= m <= XHI))
    ax.plot(M, [r[m] for m in M], color=c, ls=ls, lw=lw, marker="o", ms=2.8, label=lab)
    summary[lab] = {int(m): float(r[m]) for m in M}
ax.set_yscale("log"); ax.set_xlim(XLO - 20, XHI + 20)
ax.set_xlabel(r"$m_{\tilde g}$ [GeV]"); ax.set_ylabel(r"95% CL limit on $\sigma$ [fb]")
ax.set_title(r"around 700 GeV: mass window, P categories; PairNet expected, 300 fb$^{-1}$, 13.6 TeV", fontsize=6.2, loc="left")
ax.legend(fontsize=5.7, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
spines_to_data_range(ax)
save_pdf(fig, f"{out}/compare700.pdf"); fig.savefig(f"{out}/compare700.png", dpi=170, bbox_inches="tight")
json.dump(summary, open(f"{out}/compare700.json", "w"), indent=1)
for k_, v in summary.items():
    print(k_, {m: round(x, 1) for m, x in v.items()})
