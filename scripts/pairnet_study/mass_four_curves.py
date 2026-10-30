"""m_avg of the chosen pairing for QCD and one high-mass signal, PairNet vs argmin|m1-m2|: four curves on one panel.
Each curve is normalised to its sample's total with 6+ jets, so the area is the fraction kept. One PDF + PNG per stage."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
style, pn_npz, cl_npz, out, M = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")
bins = np.linspace(0, 3200, 65); c = 0.5 * (bins[1:] + bins[:-1])
for stage, lab in (("all", "6+ jets"), ("asym", "A < 0.1")):
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ends = []
    for method, path, ls in (("PairNet", pn_npz, "-"), ("argmin $|m_1-m_2|$", cl_npz, "--")):
        e = np.load(path); s = e["is_signal"]; cut = (e["asym"] < 0.1) if stage == "asym" else np.ones_like(s, bool)
        for samp, sel, wt, col in ((f"signal {M} GeV", s & (e["pmass"] == M), np.ones(len(s)), "#1f4e8c"), ("QCD", ~s, e["w"], "0.25")):
            h, _ = np.histogram(e["mavg"][sel & cut], bins, weights=wt[sel & cut] / wt[sel].sum())
            ax.step(c, np.where(h > 0, h, np.nan), where="mid", color=col, ls=ls, lw=1.4 if ls == "-" else 1.2)
            kept = wt[sel & cut].sum() / wt[sel].sum()
            ends.append((samp, method, col, kept))
    ax.axvspan(0.85 * M, 1.1 * M, color="#9b2c2c", alpha=0.07, lw=0)
    ax.text(0.975 * M, 0.85, "window\n[0.85, 1.1] M", ha="center", va="top", fontsize=7, color="#9b2c2c")
    ax.set_yscale("log"); ax.set_ylim(1e-6, 1)
    ax.set_xlabel(r"$m_{\rm avg}$ of the chosen pairing [GeV]"); ax.set_ylabel("fraction of the sample per bin")
    ax.set_title(lab.replace("<", "$<$"), fontsize=8.5, loc="left")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=col, ls="-" if method == "PairNet" else "--", lw=1.4,
                      label=f"{samp}, {method}" + (f"\nkept {kept:.3f}" if stage == "asym" else ""))
               for samp, method, col, kept in ends]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=7.5, handlelength=2.2, labelspacing=1.0)
    spines_to_data_range(ax)
    save_pdf(fig, f"{out}/mass4_{stage}_m{M}.pdf"); fig.savefig(f"{out}/mass4_{stage}_m{M}.png", dpi=170, bbox_inches="tight"); plt.close(fig)
print("ok")
