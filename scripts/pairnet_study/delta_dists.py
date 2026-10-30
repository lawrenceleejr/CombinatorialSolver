"""Delta = (scalar pT sum of a triplet) - m_triplet, in units of M, for the PairNet pairing in the m_avg window with
A < 0.1: signal at M vs merged QCD (cross-section weighted), before and after a P(signal) cut keeping 50% of the
signal in the window. One file per (variable, mass): Delta of the weaker triplet (min) and of the stronger (max)."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

style, big, ht_npy, big_ht, out = sys.argv[1:6]
sys.path.insert(0, style); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
import qcdmerge  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")

d, _ = qcdmerge.load(big, ht_npy, big_ht, ["pn_m1", "pn_m2", "pn_delta1", "pn_delta2", "psig"])
v = np.load(big); s = v["is_signal"]
S = {k: v[k][s] for k in ("pn_m1", "pn_m2", "pn_delta1", "pn_delta2", "psig", "pmass")}
for M in (500, 700, 1000):
    def prep(x, sig):
        mavg = 0.5 * (x["pn_m1"] + x["pn_m2"]); A = np.abs(x["pn_m1"] - x["pn_m2"]) / np.maximum(x["pn_m1"] + x["pn_m2"], 1e-9)
        sel = (A < 0.1) & (mavg > 0.85 * M) & (mavg < 1.1 * M) & ((x["pmass"] == M) if sig else True)
        return sel, np.minimum(x["pn_delta1"], x["pn_delta2"]) / M, np.maximum(x["pn_delta1"], x["pn_delta2"]) / M
    ss, smin, smax = prep(S, True)
    qs, qmin, qmax = prep(d, False)
    pcut = np.quantile(S["psig"][ss], 0.5)
    for name, sv, qv, lab in (("min", smin, qmin, r"$\Delta$ of the weaker triplet / $M$"), ("max", smax, qmax, r"$\Delta$ of the stronger triplet / $M$")):
        fig, ax = plt.subplots(figsize=(4.4, 2.9))
        bins = np.linspace(-1.5, 1.5, 49)
        for sel_s, sel_q, ls, tag in ((ss, qs, "-", r"$A<0.1$, window"), (ss & (S["psig"] >= pcut), qs & (d["psig"] >= pcut), "--", "+ P cut (50% of signal)")):
            hs, _ = np.histogram(sv[sel_s], bins); hq, _ = np.histogram(qv[sel_q], bins, weights=d["wfb"][sel_q])
            ax.stairs(hs / max(hs.sum(), 1), bins, color="#1f4e8c", ls=ls, lw=1.3, label=f"signal {M} GeV, {tag}")
            ax.stairs(hq / max(hq.sum(), 1e-30), bins, color="0.3", ls=ls, lw=1.3, label=f"QCD, {tag} ({int(sel_q.sum())} MC ev.)")
        ax.set_yscale("log"); ax.set_ylim(1e-4, 0.5)
        ax.set_xlabel(lab); ax.set_ylabel("shape (fraction per bin)")
        ax.set_title(f"PairNet pairing, {M} GeV window", fontsize=7, loc="left")
        ax.legend(fontsize=6.2, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
        spines_to_data_range(ax)
        save_pdf(fig, f"{out}/delta_{name}_m{M}.pdf"); fig.savefig(f"{out}/delta_{name}_m{M}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)
    for cut in (0.0, 0.3, 0.5, 1.0):
        es = (smax[ss] > cut).mean(); eq = (d["wfb"][qs] * (qmax[qs] > cut)).sum() / d["wfb"][qs].sum()
        es2 = (smax[ss & (S["psig"] >= pcut)] > cut).mean(); qp = qs & (d["psig"] >= pcut)
        eq2 = (d["wfb"][qp] * (qmax[qp] > cut)).sum() / d["wfb"][qp].sum()
        print(f"M={M} Delta_max > {cut}M: signal {es:.3f} QCD {eq:.4f} (S/sqrtB gain {es/np.sqrt(eq):.2f}) | after P cut: signal {es2:.3f} QCD {eq2:.4f} (gain {es2/np.sqrt(max(eq2,1e-12)):.2f}, S/B gain {es2/max(eq2,1e-12):.1f})")
