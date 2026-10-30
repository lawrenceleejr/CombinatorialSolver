"""Cutflow distributions for the recommended PairNet (test set): m_avg, A and logit P(signal) for a few signal
masses and weighted QCD, after (0) 6+ jets, (1) A < 0.1, (2) A < 0.1 and P(signal) above a threshold keeping
70% of all signal. Each sample is normalised to its own stage-0 total, so later stages show what survives.
One PDF + PNG per (variable, stage)."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

style, npz, out = sys.argv[1:4]
prefix = sys.argv[4] if len(sys.argv) > 4 else "cutflow"
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")

e = np.load(npz); s = e["is_signal"]; q = ~s; w = e["w"]
A = e["asym"]; x = e["mavg"]
HAS_P = "psig" in e.files
p = np.clip(e["psig"], 1e-7, 1 - 1e-7) if HAS_P else np.ones_like(A); lp = np.log(p / (1 - p))
thr = np.quantile(p[s], 0.30) if HAS_P else 0.0       # keeps 70% of all signal
stages = [("0_all", "6+ jets", np.ones_like(s, bool)), ("1_asym", "A < 0.1", A < 0.1), ("2_asym_p", "A < 0.1 and P(signal) cut", (A < 0.1) & (p >= thr))][: 3 if HAS_P else 2]
MASSES = (500, 1000, 1500, 2000)
COL = {500: "#c6dbef", 1000: "#6baed6", 1500: "#2171b5", 2000: "#08306b"}
samples = [(f"signal {M} GeV", s & (e["pmass"] == M), np.ones_like(w), COL[M]) for M in MASSES] + [("QCD (weighted)", q, w, "0.45")]
VARS = [("mavg", x, np.linspace(0, 3000, 61), r"$m_{\rm avg}$ of the chosen pairing [GeV]"),
        ("asym", A, np.linspace(0, 1, 51), r"mass asymmetry $A = |m_1-m_2|/(m_1+m_2)$"),
        ("logitp", lp, np.linspace(-14, 16, 61), r"logit $P$(signal)")][: 3 if HAS_P else 2]
flow = {}
for vname, v, bins, xl in VARS:
    ymin = 1e-6
    for sname, slab, cut in stages:
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        for lab, sel, wt, col in samples:
            tot = wt[sel].sum()
            h, _ = np.histogram(v[sel & cut], bins, weights=wt[sel & cut] / tot)
            h = h / np.diff(bins) * (bins[-1] - bins[0]) / (len(bins) - 1)  # fraction per bin
            kw = dict(color=col, lw=1.2) if not lab.startswith("QCD") else dict(color=col, lw=1.0)
            if lab.startswith("QCD"):
                ax.fill_between(0.5 * (bins[1:] + bins[:-1]), np.maximum(h, ymin / 10), ymin / 10, step="mid", color="0.85", lw=0)
            ax.step(0.5 * (bins[1:] + bins[:-1]), np.where(h > 0, h, np.nan), where="mid", **kw)
            flow.setdefault(lab, {})[sname] = float(wt[sel & cut].sum() / tot)
        if vname == "asym":
            ax.axvline(0.1, color="#9b2c2c", lw=0.8, ls="--")
        if vname == "logitp":
            ax.axvline(np.log(thr / (1 - thr)), color="#9b2c2c", lw=0.8, ls="--")
        ax.set_yscale("log"); ax.set_ylim(ymin, 1)
        ax.set_xlabel(xl); ax.set_ylabel("fraction of the sample per bin")
        ax.set_title(slab.replace("<", "$<$"), fontsize=8.5, loc="left")
        spines_to_data_range(ax)
        save_pdf(fig, f"{out}/{prefix}_{vname}_{sname}.pdf"); fig.savefig(f"{out}/{prefix}_{vname}_{sname}.png", dpi=150, bbox_inches="tight"); plt.close(fig)
# legend as its own small figure, so every panel stays uncluttered
fig, ax = plt.subplots(figsize=(4.6, 0.5)); ax.axis("off")
for i, (lab, _, _, col) in enumerate(samples):
    ax.plot([], [], color=col, lw=2, label=lab)
ax.legend(loc="center", ncol=5, frameon=False, fontsize=7.5, handlelength=1.5)
fig.savefig(f"{out}/cutflow_legend.png", dpi=150, bbox_inches="tight"); plt.close(fig)
json.dump({"p_threshold": float(thr), "flow": flow}, open(f"{out}/{prefix}.json", "w"), indent=1)
print("P threshold", thr)
for lab, d in flow.items():
    print(f"{lab:18s} " + "  ".join(f"{k}: {v:.4f}" for k, v in d.items()))
