"""HT distributions (scalar pT sum of all jets with pT > 30 GeV, |eta| < 3) on the r8B test set.

ht_inclusive.png       6+ jets: signal 500/1000/1500/2000 GeV and weighted QCD
ht_window_m{M}.png     after A < 0.1 and the m_avg window for M: signal and QCD for PairNet (and its P cut) and argmin
"""
import glob, json, sys
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

style, vars_npz, ht_all, xs_json, out = sys.argv[1:6]
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")

v = np.load(vars_npz); s = v["is_signal"]; q = ~s; w = v["w"]; pm = v["pmass"]
nsig = int(s.sum())
ht = np.empty(len(s), np.float32)
ht[:nsig] = np.load(ht_all)[:nsig]  # signal test files come first in both orderings
hq = []
for f in sorted(glob.glob("/scratch/llee/pairnet/data_r3/prod/qcd_c*9.h5")):
    with h5py.File(f) as h:
        pt = h["INPUTS/Source/pt"][:]; m = h["INPUTS/Source/MASK"][:].astype(bool)
    hq.append((pt * m).sum(1)[m.sum(1) >= 6])
ht[nsig:] = np.concatenate(hq)
np.save(f"{out}/ht_test_r8B.npy", ht)
BLUES = {500: "#9ecae1", 1000: "#4292c6", 1500: "#2171b5", 2000: "#08306b"}

fig, ax = plt.subplots(figsize=(4.6, 3.1))
bins = np.linspace(0, 5000, 61)
for M, c in BLUES.items():
    sel = s & (pm == M)
    h, _ = np.histogram(ht[sel], bins)
    ax.stairs(h / sel.sum(), bins, color=c, lw=1.3)
    i = np.argmax(h)
    ax.annotate(f"{M} GeV", (bins[i] + 60, h[i] / sel.sum()), fontsize=7, color=c)
h, _ = np.histogram(ht[q], bins, weights=w[q])
ax.stairs(h / w[q].sum(), bins, color="0.35", lw=1.0, fill=True, alpha=0.25)
ax.annotate("QCD (weighted)", (bins[3], h[2] / w[q].sum()), fontsize=7, color="0.3")
ax.set_yscale("log"); ax.set_ylim(1e-5, 1)
ax.set_xlabel(r"$H_T$ [GeV]"); ax.set_ylabel("fraction of the sample per bin")
ax.set_title("6+ jets, before any pairing", fontsize=7, loc="left")
spines_to_data_range(ax)
save_pdf(fig, f"{out}/ht_inclusive.pdf"); fig.savefig(f"{out}/ht_inclusive.png", dpi=170, bbox_inches="tight")

# P cut per mass from the xs study (tuned PairNet + A < 0.1 + P selection)
rows = json.load(open(xs_json))["rows"]
for M in (1000, 1500, 2000):
    fig, ax = plt.subplots(figsize=(4.8, 3.1))
    bins = np.linspace(0, 3.2 * M, 49)
    win = lambda p: (v[f"{p}_asym"] < 0.1) & (v[f"{p}_mavg"] > 0.85 * M) & (v[f"{p}_mavg"] < 1.1 * M)
    sg = s & (pm == M)
    eff = rows[str(M)]["pairnet_AP"]["p_cut_sig_eff"]
    pcut = np.quantile(v["psig"][sg & win("pn")], 1 - eff)
    curves = (("pn", win("pn"), "PairNet", "#9b2c2c", "-"),
              ("pnP", win("pn") & (v["psig"] >= pcut), "PairNet + P cut", "#9b2c2c", "--"),
              ("am", win("am"), r"argmin $|m_1-m_2|$", "0.45", "-"))
    bins = np.linspace(0, 3.2 * M, 25)
    hs, _ = np.histogram(ht[sg & win("pn")], bins)
    ax.stairs(hs / hs.sum(), bins, color="#1f4e8c", lw=1.6, label=f"signal {M} GeV, PairNet window")
    for _, sel, lab, c, ls in curves:
        hb, _ = np.histogram(ht[q & sel], bins, weights=w[q & sel])
        ax.stairs(hb / max(hb.sum(), 1e-12), bins, color=c, ls=ls, lw=1.2, label=f"QCD, {lab} ({int((q & sel).sum())} MC ev.)")
    ax.axvline(M, color="0.7", lw=0.6)
    ax.set_xlabel(r"$H_T$ [GeV]"); ax.set_ylabel("shape (fraction per bin)")
    ax.set_title(f"$H_T$ shapes in the {M} GeV $m_{{\\rm avg}}$ window, $A<0.1$", fontsize=7, loc="left")
    ax.legend(fontsize=6.3, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
    spines_to_data_range(ax)
    save_pdf(fig, f"{out}/ht_window_m{M}.pdf"); fig.savefig(f"{out}/ht_window_m{M}.png", dpi=170, bbox_inches="tight")
    for name, sel, *_ in curves:
        print(M, name, "signal HT q10/50/90", np.round(np.quantile(ht[sg & sel], [.1, .5, .9])),
              "QCD (unweighted) q10/50/90", np.round(np.quantile(ht[q & sel], [.1, .5, .9])), "nQCD", int((q & sel).sum()))
