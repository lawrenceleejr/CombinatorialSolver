"""Expected reach versus integrated luminosity, with the selections tuned at 300 fb^-1 (cuts_study3 json): for each mass
and test half, signal efficiency and background cross section are fixed, and the limit is recomputed at luminosity L
(no systematics, and with a 5% background uncertainty). Reach against the 13.6 TeV NNLOapprox+NNLL cross section."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

style, js0, js5, nnll, out = sys.argv[1:6]
sys.path.insert(0, style); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
from limits import s_up  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")

th = json.load(open(nnll))["data"]
tm = np.array(sorted(int(float(k)) for k in th)); tx = np.array([th[str(m)]["xsec_pb"] * 1e3 for m in tm])
sig = lambda m: np.exp(np.interp(m, tm, np.log(tx)))
LUM = [60, 100, 140, 200, 300, 450, 700, 1000]
SEL = (("am+all", "argmin + tuned cuts", "0.3"), ("dma+all", r"max $\Delta$ ($A<0.1$) + tuned cuts", "#5f8a5f"), ("pn", "PairNet + P cut", "#9b2c2c"))


def reach(M, lim):
    r = np.log(lim) - np.log(sig(M))
    ex = np.where(r < 0)[0]
    if len(ex) == 0:
        return np.nan
    i = ex[-1]
    return float(M[i]) if i == len(M) - 1 else float(M[i] + (M[i + 1] - M[i]) * (-r[i]) / (r[i + 1] - r[i]))


summary = {}
fig, ax = plt.subplots(figsize=(4.6, 3.0))
for js, f, ls in ((js0, 0.0, "-"), (js5, 0.05, "--")):
    res = json.load(open(js))
    M = np.array(sorted(int(k) for k in res))
    for k, lab, c in SEL:
        rr = []
        for L in LUM:
            lim = []
            for m in M:
                x = res[str(m)][k]
                ls_ = [s_up(B * L, f) / (L * e) if e > 0 else np.inf for B, e in zip(x["xsB_fb"], x["effS"])]
                lim.append(np.exp(np.mean(np.log(ls_))))
            rr.append(reach(M, np.array(lim)))
        summary[f"{k}_f{f:g}"] = dict(zip(LUM, rr))
        ax.plot(LUM, np.array(rr) / 1000, color=c, ls=ls, lw=1.4, label=f"{lab}, " + ("no syst." if f == 0 else "5% on B"))
ax.axvline(300, color="0.75", lw=0.6)
ax.set_xscale("log"); ax.set_xticks(LUM); ax.set_xticklabels([str(x) for x in LUM])
ax.set_xlabel(r"integrated luminosity at 13.6 TeV [fb$^{-1}$]"); ax.set_ylabel("expected gluino mass reach [TeV]")
ax.axhline(3.0, color="0.6", lw=0.6, ls=":"); ax.annotate("signal samples end at 3.0 TeV", (LUM[0], 3.0), xytext=(0, 3), textcoords="offset points", fontsize=6.3, color="0.45")
ax.set_title(r"selections tuned at 300 fb$^{-1}$; reach against NNLO$_{\rm approx}$+NNLL $\sigma$", fontsize=6.5, loc="left")
ax.legend(fontsize=6.3, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
spines_to_data_range(ax)
save_pdf(fig, f"{out}/lumi_scan.pdf"); fig.savefig(f"{out}/lumi_scan.png", dpi=170, bbox_inches="tight")
json.dump(summary, open(f"{out}/lumi_scan.json", "w"), indent=1)
for k, d in summary.items():
    print(k, {L: (round(r) if r == r else None) for L, r in d.items()})
