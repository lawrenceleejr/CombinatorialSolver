"""How much falling spectrum below the window is worth: profiled shape-fit sensitivity vs the start of the fit range.

Fixed selection per mass (the dumps of fisher_high.py, whose QCD m_avg peaks at 0.5-0.7 M); the background fit
(4-parameter dijet, all parameters free) starts at x0 * M and runs to 2 M. Plots sigma_mu(x0) / sigma_mu(background
known), i.e. the price of fitting the background, against x0. Single panel.
usage: lever_arm.py style dump_dir out masses
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

style, dd, out = sys.argv[1:4]; masses = [int(x) for x in sys.argv[4].split(",")]
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")
L = 300.0
FB = np.linspace(0.4, 2.0, 81); FC = 0.5 * (FB[1:] + FB[:-1])
X0 = np.arange(0.45, 0.851, 0.025)
COL = ["#9b2c2c", "#2b5d8a", "#5b8c5a", "#b07d2b", "0.3"]


def basis(x):
    return np.stack([np.ones_like(x), np.log1p(-x), np.log(x), np.log(x) ** 2], 1)


res = {}
fig, ax = plt.subplots(figsize=(4.6, 3.0))
for i, M in enumerate(masses):
    a = np.load(f"{dd}/spec_r14A_m{M}_trig450_fisher.npz", allow_pickle=True)
    sh, _ = np.histogram(a["sig_mavg"] / M, FB, weights=a["sig_w"] * L)
    qh, _ = np.histogram(a["qcd_mavg"] / M, FB, weights=a["qcd_w"] * L); q2, _ = np.histogram(a["qcd_mavg"] / M, FB, weights=(a["qcd_w"] * L) ** 2)
    pk = FC[int(np.argmax(np.convolve(qh, np.ones(5) / 5, mode="same")))]
    x = FC * M / 13600.0; X = basis(x)
    fit = (FC >= pk) & (qh > 0) & (q2 > 0)
    wt = qh[fit] / np.sqrt(q2[fit])
    th, *_ = np.linalg.lstsq(X[fit] * wt[:, None], np.log(qh[fit]) * wt, rcond=None)
    b = np.exp(X @ th)  # smooth background model from the peak upward, used for every fit range
    ys = []
    for x0 in X0:
        use = FC >= max(x0, pk)
        A = X[use] * np.sqrt(b[use])[:, None]; v = sh[use] / np.sqrt(b[use])
        r = v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
        known = 1 / np.sqrt((sh[FC >= pk] ** 2 / b[FC >= pk]).sum())
        ys.append((1 / np.sqrt(r @ r)) / known)
    ys = np.array(ys); ok = X0 >= pk - 1e-9
    ax.plot(X0[ok], ys[ok], "o-", ms=2.5, lw=1.2, color=COL[i % len(COL)])
    ax.annotate(f"{M} GeV", (X0[ok][-1], ys[ok][-1]), xytext=(4, 0), textcoords="offset points", fontsize=6.5, color=COL[i % len(COL)], va="center")
    res[M] = {"qcd_peak_over_M": float(pk), "x0": X0[ok].tolist(), "sigma_ratio": ys[ok].tolist()}
    print(M, "peak %.2f" % pk, " ".join(f"{x0:.3f}:{y:.2f}" for x0, y in zip(X0[ok], ys[ok])))
ax.axvspan(0.85, 0.87, color="#9b2c2c", alpha=0.15, lw=0); ax.text(0.845, ax.get_ylim()[1] * 0.97, "window starts", fontsize=6, ha="right", va="top", color="#9b2c2c")
ax.set_xlabel(r"start of the background fit / $M$ (fit to 2 $M$)")
ax.set_ylabel(r"$\sigma_\mu$(free dijet fit) / $\sigma_\mu$(bkg. known)")
ax.set_title("same selection, less falling spectrum below the window: the fit absorbs more of the signal", fontsize=6.3, loc="left")
spines_to_data_range(ax)
save_pdf(fig, f"{out}/lever_arm.pdf"); fig.savefig(f"{out}/lever_arm.png", dpi=170, bbox_inches="tight")
json.dump(res, open(f"{out}/lever_arm.json", "w"), indent=1)
