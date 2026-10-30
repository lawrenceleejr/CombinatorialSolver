"""Slide-ready plots from cuts_study2_f*.json (one panel per file, legends outside the axes)."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker  # noqa: E402

style, js, cms_json, out, tag = sys.argv[1:6]
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")
res = json.load(open(js))
Ms = np.array(sorted(int(k) for k in res))
get = lambda k: np.array([res[str(m)][k]["limit_fb"] for m in Ms])
low = lambda k: np.array([res[str(m)][k]["min_nB_test"] < 5 for m in Ms])
NOTE = "open circles: fewer than 5 QCD MC events in the test half"


def draw(ax, k, lab, c, ls="-", lw=1.3, ref=None):
    y = get(k) / (get(ref) if ref else 1)
    ax.plot(Ms, y, color=c, ls=ls, lw=lw, label=lab)
    b = low(k)
    if b.any():
        ax.plot(Ms[b], y[b], ls="none", marker="o", mfc="white", mec=c, ms=3.2)


GROUPS = (("ht", "$H_T$", "0.15"), ("pt_low6", "lowest jet $p_T$ of the six", "#c9a227"), ("delta", r"$\Delta$ (weaker, stronger triplet)", "#5f8a5f"),
          ("d32", r"$D^2_{[3,2]}$ (worse, better triplet)", "#6a5f9b"), ("d2_63_32", r"$D^2_{[(6,3)+(3,2)]}$", "#4a87a8"))
for p, title, ylim in (("pn", "PairNet + P(signal) cut", (0.5, 2.0)), ("am", r"argmin $|m_1-m_2|$, $A<0.1$", None), ("dma", r"max $\Delta$ among $A<0.1$", None)):
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for g, lab, c in GROUPS:
        draw(ax, f"{p}+{g}", "+ " + lab, c, lw=1.0, ref=p)
    draw(ax, f"{p}+all", "+ all tuned together", "#9b2c2c", lw=1.8, ref=p)
    if p != "pn":
        draw(ax, f"{p}_cms", "+ fixed CMS-EXO-21-004 cuts", "0.45", ls="--", ref=p)
    ax.axhline(1, color="0.6", lw=0.6)
    ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim); ax.set_yticks([0.5, 0.7, 1, 1.4, 2]); ax.set_yticklabels(["0.5", "0.7", "1", "1.4", "2"]); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"$m_{\tilde g}$ [GeV]"); ax.set_ylabel(f"limit / {title} alone")
    ax.set_title(f"{title}: tuned on one half, tested on the other; {NOTE}", fontsize=6, loc="left")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
    spines_to_data_range(ax)
    save_pdf(fig, f"{out}/{tag}_gain_{p}.pdf"); fig.savefig(f"{out}/{tag}_gain_{p}.png", dpi=170, bbox_inches="tight")

cms = json.load(open(cms_json))
fig, ax = plt.subplots(figsize=(4.8, 3.2))
for k, lab, c, ls in (("am", r"argmin, $A<0.1$ only", "0.6", "--"), ("am+all", r"argmin + tuned $H_T$, $p_T$, $\Delta$, Dalitz cuts", "0.3", "-"),
                      ("dma+all", r"max $\Delta$ ($A<0.1$) + tuned cuts", "#5f8a5f", "-"), ("pn", "PairNet + P(signal) cut", "#9b2c2c", "-")):
    draw(ax, k, lab, c, ls, 1.4)
cm_m = np.array(sorted(int(x) for x in cms["ratio_cms_exp_over_ours"]))
ax.plot(cm_m, [cms["ratio_cms_exp_over_ours"][str(x)]["cms_exp_300_fb"] for x in cm_m], color="#2b5d8a", lw=1.3, marker="o", ms=2.5,
        label=r"CMS-EXO-21-004 expected $\times\sqrt{128/300}$")
ax.set_yscale("log"); ax.set_xlabel(r"$m_{\tilde g}$ [GeV]"); ax.set_ylabel(r"expected 95% CL limit on $\sigma$ [fb], 300 fb$^{-1}$")
fs = "no systematics" if tag.endswith("f0") else "5% background uncertainty"
ax.set_title(f"tuned on one half, tested on the other; {fs}; {NOTE}", fontsize=6, loc="left")
ax.legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
spines_to_data_range(ax)
save_pdf(fig, f"{out}/{tag}_limits.pdf"); fig.savefig(f"{out}/{tag}_limits.png", dpi=170, bbox_inches="tight")
for m in (500, 1000, 1500, 2000):
    r = res[str(m)]
    print(m, {k: round(r[k]["limit_fb"], 2) for k in ("am", "am+all", "dma+all", "pn", "pn+all")},
          "PairNet gain over best tuned classical %.2f" % (min(r["am+all"]["limit_fb"], r["dma+all"]["limit_fb"]) / r["pn"]["limit_fb"]))
