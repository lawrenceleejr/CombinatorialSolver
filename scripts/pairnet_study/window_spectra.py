"""m_avg spectra after the 10% mass-window selection (enriched QCD merged), 300 fb^-1: QCD, and signal at its expected
95% CL limit stacked on top, with the window [0.85, 1.1] M. One single-panel file per mass.
usage: window_spectra.py style out working_points.json M:dump.npz ..."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

style, out, wpj = sys.argv[1:4]; specs = [a.split(":", 1) for a in sys.argv[4:]]
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")
wp = json.load(open(wpj)); L = 300.0
for ms, path in specs:
    M = int(ms); a = np.load(path, allow_pickle=True); r = wp[ms]
    b = np.linspace(0.5 * M, 1.6 * M, 45)
    q, _ = np.histogram(a["qcd_mavg"], b, weights=a["qcd_w"] * L)
    s, _ = np.histogram(a["sig_mavg"], b, weights=a["sig_w"])
    sw = a["sig_w"][(a["sig_mavg"] > 0.85 * M) & (a["sig_mavg"] < 1.1 * M)].sum()
    s = s * (r["eff"] * r["limit"] * L / sw)  # signal shape from the dump, window yield = efficiency x limit x L
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    ax.axvspan(0.85 * M, 1.1 * M, color="#9b2c2c", alpha=0.07, lw=0)
    ax.stairs(q + s, b, color="#9b2c2c", lw=1.3, label=f"QCD + signal at the expected limit ({r['limit']:.3g} fb)")
    ax.stairs(q, b, color="0.25", lw=1.3, label="QCD (enriched MC), 300 fb$^{-1}$")
    win = (0.5 * (b[1:] + b[:-1]) > 0.85 * M) & (0.5 * (b[1:] + b[:-1]) < 1.1 * M)
    ax.set_xlabel(r"$m_{\rm avg}$ [GeV]"); ax.set_ylabel(r"events / bin, 300 fb$^{-1}$")
    ax.set_ylim(0, max((q + s).max() * 1.25, 1))
    ax.set_title(f"{M} GeV: window S = {r['eff'] * r['limit'] * L:.3g}, B = {r['B']:.3g} (S/B = {r['eff'] * r['limit'] * L / r['B']:.2f})", fontsize=6.5, loc="left")
    ax.legend(fontsize=6.0, frameon=False, loc="upper right")
    spines_to_data_range(ax)
    save_pdf(fig, f"{out}/winspec_m{M}.pdf"); fig.savefig(f"{out}/winspec_m{M}.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    print(M, "B in window (dump) %.3g vs table %.3g" % (q[win].sum(), r["B"]))
