"""Cost of requiring a smoothly falling QCD m_avg spectrum under the signal window.

For each low-mass study variant (unconstrained; QCD peak <= 0.75 M; <= 0.6 M; DDT with <= 0.75 M) take the best
PairNet selection per mass and plot (1) the limit relative to the unconstrained one and (2) where the QCD m_avg
spectrum peaks after the chosen selection, in units of M, against the window [0.85, 1.1] M.
usage: falling_cost.py style out trig ftag label=dir [label=dir ...]   (first entry is the reference)
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

style, out, trig, ftag = sys.argv[1:5]
runs = [a.split("=", 1) for a in sys.argv[5:]]
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")
CONFS = ("P", "P+boost", "P+all", "DDT", "DDT+boost")
COL = ["0.15", "#9b2c2c", "#2b5d8a", "#5b8c5a", "#b07d2b"]


def dump_peak(dr, m):
    """QCD peak / M from the best-selection dump (runs without the shape constraint store no peak)."""
    p = f"{dr}/dump_{ftag}/spec_r14A_m{m}_trig{trig}_{ftag}.npz"
    if not os.path.exists(p):
        return np.nan
    a = np.load(p, allow_pickle=True); b = np.arange(0.4, 1.6001, 0.05)
    y, _ = np.histogram(a["qcd_mavg"] / m, b, weights=a["qcd_w"])
    y = np.convolve(y, np.ones(3) / 3, mode="same")
    return 0.5 * (b[1:] + b[:-1])[np.argmax(y)]


def best(d, dr):
    out = {}
    for m, x in d.items():
        c = [k for k in CONFS if k in x and np.isfinite(x[k]["limit_fb"])]
        if not c:
            out[int(m)] = (np.inf, None, np.nan)
            continue
        k = min(c, key=lambda k: x[k]["limit_fb"])
        pk = np.mean([h.get("qcd_peak_over_M", np.nan) for h in x[k].get("halves", [])] or [np.nan])
        out[int(m)] = (x[k]["limit_fb"], k, pk if pk == pk else dump_peak(dr, int(m)))
    return out


data = {}
for lab, dr in runs:
    p = f"{dr}/lowmass_eval_r14A_trig{trig}_{ftag}.json"
    if os.path.exists(p):
        data[lab] = best(json.load(open(p))["results"], dr)
ref = runs[0][0]
summary = {lab: {m: {"limit_fb": v[0], "sel": v[1], "qcd_peak_over_M": v[2], "ratio": v[0] / data[ref][m][0]} for m, v in d.items()} for lab, d in data.items()}
json.dump(summary, open(f"{out}/falling_cost_trig{trig}_{ftag}.json", "w"), indent=1, default=float)

fig, ax = plt.subplots(figsize=(4.6, 3.0))
for i, (lab, d) in enumerate(data.items()):
    if lab == ref:
        continue
    ms = np.array(sorted(d)); r = np.array([d[m][0] / data[ref][m][0] for m in ms])
    ok = np.isfinite(r)
    ax.plot(ms[ok], r[ok], "o-", ms=3, lw=1.2, color=COL[i], label=lab)
    for m in ms[~ok]:
        ax.annotate("not feasible", (m, 1.0), color=COL[i], fontsize=6, rotation=90, ha="center", va="bottom")
ax.axhline(1, color="0.6", lw=0.6)
ax.set_xlabel(r"$m_{\tilde g}$ [GeV]"); ax.set_ylabel("limit / unconstrained limit")
ax.set_title(f"r14A, offline $H_T>{trig}$ GeV, {ftag}: price of a falling QCD spectrum under the window", fontsize=6.5, loc="left")
ax.legend(fontsize=6.5, frameon=False)
spines_to_data_range(ax)
save_pdf(fig, f"{out}/falling_cost_trig{trig}_{ftag}.pdf"); fig.savefig(f"{out}/falling_cost_trig{trig}_{ftag}.png", dpi=170, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.6, 3.0))
ax.axhspan(0.85, 1.1, color="#9b2c2c", alpha=0.08, lw=0); ax.text(1010, 0.97, "window", color="#9b2c2c", fontsize=6.5, va="center")
for i, (lab, d) in enumerate(data.items()):
    ms = np.array(sorted(d)); pk = np.array([d[m][2] for m in ms])
    ax.plot(ms, pk, "o-", ms=3, lw=1.2, color=COL[i], label=lab)
ax.set_xlabel(r"$m_{\tilde g}$ [GeV]"); ax.set_ylabel(r"QCD $m_{\rm avg}$ peak / $M$ after the selection")
ax.set_title(f"r14A, offline $H_T>{trig}$ GeV, {ftag}: where the QCD spectrum turns over", fontsize=6.5, loc="left")
ax.legend(fontsize=6.5, frameon=False)
spines_to_data_range(ax)
save_pdf(fig, f"{out}/qcd_peak_trig{trig}_{ftag}.pdf"); fig.savefig(f"{out}/qcd_peak_trig{trig}_{ftag}.png", dpi=170, bbox_inches="tight"); plt.close(fig)
for lab, d in data.items():
    print(lab, {m: (round(v[0], 1), v[1], round(v[2], 2) if v[2] == v[2] else None) for m, v in sorted(d.items())})
