"""Inputs for the PairNet vs main-branch report.

classical argmin|m1-m2| pairing of the leading 7 jets on the test set (the label main's phase 1 learns):
overall A<0.1 metrics and per-mass S/sqrt(B) with A<0.1 + m_avg window; PairNet r8B per-mass numbers from
its full-selection JSON; plots of the per-epoch history of main (validation hook) against r8B."""
import json, math, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

style, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")
sys.path.insert(0, "/scratch/llee/CombinatorialSolver")
from src.combinatorics import enumerate_assignments  # noqa: E402

# corrected argmin numbers (pairings use real jets only), from classical_metrics.py on the 10M-QCD test files
_c = json.load(open(f"{out}/classical.json"))
cl = _c["classical"]; cl_mass = {int(k): v for k, v in _c["classical_mass"].items()}
pn = json.load(open("/scratch/llee/pairnet/plots/selection/full_selection.json"))["r8B"]

# ---- per-mass sensitivity plot
Ms = sorted(int(k) for k in pn)
fig, ax = plt.subplots(figsize=(4.8, 3.2))
ax.plot(Ms, [pn[str(M)]["fom_best"] for M in Ms], color="#9b2c2c", lw=1.4, marker="o", ms=2.5)
ax.plot(Ms, [pn[str(M)]["fom_asym_only"] for M in Ms], color="#9b2c2c", lw=1.0, ls="--", marker="o", ms=2)
cm = [M for M in Ms if cl_mass.get(M, {}).get("fom")]
ax.plot(cm, [cl_mass[M]["fom"] for M in cm], color="0.35", lw=1.0, marker="o", ms=2)
for xs, ys, lab, col in ((Ms, [pn[str(M)]["fom_best"] for M in Ms], "PairNet: A<0.1 + P(signal) cut", "#9b2c2c"),
                         (Ms, [pn[str(M)]["fom_asym_only"] for M in Ms], "PairNet: A<0.1 only", "#9b2c2c"),
                         (cm, [cl_mass[M]["fom"] for M in cm], r"argmin $|m_1-m_2|$: A<0.1 only", "0.35")):
    ax.annotate(lab.replace("A<0.1", "$A<0.1$"), (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7, color=col)
ax.set_yscale("log"); ax.set_xlabel("signal mass hypothesis [GeV]"); ax.set_ylabel(r"$S/\sqrt{B}$ in the $m_{\rm avg}$ window (relative)")
spines_to_data_range(ax)
save_pdf(fig, f"{out}/fom_vs_mass.pdf"); fig.savefig(f"{out}/fom_vs_mass.png", dpi=160, bbox_inches="tight"); plt.close(fig)

# ---- per-epoch history: main (validation hook) vs PairNet r8B
hm = [json.loads(l) for l in open("/scratch/llee/pairnet/live/main.jsonl")]
hp = json.load(open("/scratch/llee/pairnet/sweep/out_r8B_small_12j_win5/r8B_small_12j_win5/metrics.json"))["history"]
p2 = next((r["epoch"] for r in hm if r.get("phase") == 2), None)
for key, ylab, fname in (("ratio", "signal in window at $A<0.1$ /\n" + r"$\sqrt{\rm QCD\ passing\ } A<0.1$", "ratio_vs_epoch"),
                         ("eff_on_reconstructable", "exact-pairing efficiency\n(reconstructable signal)", "eff_vs_epoch")):
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    f = (lambda r: r["sig_pass_asym_in_window"] / math.sqrt(r["bkg_pass_asym"])) if key == "ratio" else (lambda r: r[key])
    ax.plot([r["epoch"] for r in hp], [f(r) for r in hp], color="#9b2c2c", lw=1.3, marker="o", ms=2)
    ax.plot([r["epoch"] for r in hm], [f(r) for r in hm], color="0.2", lw=1.3, marker="o", ms=2)
    ax.axhline(cl["ratio"] if key == "ratio" else cl["eff_on_reconstructable"], color="0.55", lw=0.8, ls=":")
    ax.annotate("PairNet (r8B)", (hp[-1]["epoch"], f(hp[-1])), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7.5, color="#9b2c2c")
    ax.annotate("main-branch network", (hm[-1]["epoch"], f(hm[-1])), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7.5, color="0.2")
    ax.annotate(r"argmin $|m_1-m_2|$", (0, cl["ratio"] if key == "ratio" else cl["eff_on_reconstructable"]), xytext=(2, 3), textcoords="offset points", fontsize=7, color="0.45")
    if p2 is not None:
        ax.axvline(p2 - 0.5, color="0.75", lw=0.6)
        lo, hi = ax.get_ylim(); ax.text(p2 - 0.3, lo + 0.55 * (hi - lo), "main enters\nphase 2", fontsize=6.5, color="0.5", va="center")
    ax.set_xlabel("epoch"); ax.set_ylabel(ylab)
    spines_to_data_range(ax)
    save_pdf(fig, f"{out}/{fname}.pdf"); fig.savefig(f"{out}/{fname}.png", dpi=160, bbox_inches="tight"); plt.close(fig)
print("done")
