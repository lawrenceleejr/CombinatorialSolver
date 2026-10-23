"""Where the QCD m_avg spectrum peaks (in units of M) after the best selection, for the window-counting and the
shape-fit figures of merit, with and without the falling-spectrum requirement. Peaks from the spectra dumps.
usage: peak_fom.py style out label=dump_glob_pattern_with_{M} ..."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
style, out = sys.argv[1:3]; runs = [a.split("=", 1) for a in sys.argv[3:]]
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")
MS = [250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]
STY = [("0.55", "--"), ("0.55", "-"), ("#9b2c2c", "--"), ("#9b2c2c", "-")]
b = np.arange(0.4, 1.6001, 0.05); c = 0.5 * (b[1:] + b[:-1])
res = {}
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.axhspan(0.85, 1.1, color="#9b2c2c", alpha=0.07, lw=0); ax.text(255, 1.07, "signal window", fontsize=6.5, color="#9b2c2c", va="center")
for (lab, pat), (col, ls) in zip(runs, STY):
    pk = []
    for M in MS:
        try:
            a = np.load(pat.format(M=M), allow_pickle=True)
        except FileNotFoundError:
            pk.append(np.nan); continue
        y, _ = np.histogram(a["qcd_mavg"] / M, b, weights=a["qcd_w"]); y = np.convolve(y, np.ones(3) / 3, mode="same")
        pk.append(c[np.argmax(y)])
    res[lab] = dict(zip(MS, pk))
    ax.plot(MS, pk, ls=ls, color=col, lw=1.3, marker="o", ms=2.5, label=lab)
ax.set_xlabel(r"$m_{\tilde g}$ [GeV]"); ax.set_ylabel(r"QCD $m_{\rm avg}$ peak / $M$")
ax.set_title(r"offline $H_T>450$ GeV: optimising for a bump hunt keeps the QCD turnover low", fontsize=6.5, loc="left")
ax.legend(fontsize=6.2, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
spines_to_data_range(ax)
save_pdf(fig, f"{out}/peak_fom.pdf"); fig.savefig(f"{out}/peak_fom.png", dpi=170, bbox_inches="tight")
json.dump(res, open(f"{out}/peak_fom.json", "w"), indent=1, default=float)
for k, v in res.items(): print(k, {m: round(x, 2) for m, x in v.items()})
