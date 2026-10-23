"""Spurious signal of the 4- and 5-parameter dijet functions on the QCD MC, one panel.
usage: spurious_compare.py style dir"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

style, d0 = sys.argv[1:3]
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.axhspan(-0.5, 0.5, color="0.92", lw=0); ax.axhline(0, color="0.6", lw=0.6)
for t, col, dx, lab in (("_p4_o0", "0.55", -12, "4-parameter dijet function"), ("_p5_o0", "#9b2c2c", 12, r"5-parameter (adds $\log^3 x$)")):
    d = json.load(open(f"{d0}/spurious{t}.json")); ms = np.array(sorted(int(m) for m in d))
    y = np.array([d[str(m)]["ss_over_limit"] for m in ms]); e = np.array([d[str(m)]["err_over_limit"] for m in ms])
    ch = ((y / e) ** 2).sum()
    ax.errorbar(ms + dx, y, e, fmt="o", ms=2.8, color=col, lw=0.9, capsize=0, label=f"{lab}: $\\chi^2$ for zero = {ch:.0f}/{len(ms)}")
ax.set_xlabel(r"$m_{\tilde g}$ [GeV]"); ax.set_ylabel("fitted signal in QCD-only MC / expected limit")
ax.set_title("spurious signal on the QCD MC after the shape-fit-tuned selections (bars: MC statistics)", fontsize=6.0, loc="left")
ax.legend(fontsize=6.3, frameon=False, loc="lower right")
spines_to_data_range(ax)
save_pdf(fig, f"{d0}/spurious_compare.pdf"); fig.savefig(f"{d0}/spurious_compare.png", dpi=170, bbox_inches="tight")
