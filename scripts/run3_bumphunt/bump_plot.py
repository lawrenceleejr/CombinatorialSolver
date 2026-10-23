"""What the signal looks like on the smooth background in a fit-based search (Asimov, 300 fb^-1).

Background = 4-parameter dijet fit to the QCD MC after the chosen selection (from its peak to 2 M); pseudo-data =
background + mu * signal with mu set so that the injected signal has 5 sigma_mu (profiled shape-fit sensitivity).
The pseudo-data are then fitted background-only: the part of the signal the fit absorbs disappears, the rest is the
bump. Two single-panel files per mass: spectrum with the fit, and (data - fit) / sqrt(fit).
usage: bump_plot.py style dump_dir out trig masses
"""
import glob, json, sys
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

style, dd, out, trig = sys.argv[1:5]; masses = [int(x) for x in sys.argv[5].split(",")]
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")
L = 300.0


def basis(x):
    return np.stack([np.ones_like(x), np.log1p(-x), np.log(x), np.log(x) ** 2], 1)


def fit_bkg(y, x, w):
    """dijet function (log-linear basis) fitted to y with Poisson-like errors sqrt(w); returns the curve."""
    X = basis(x)
    th0, *_ = np.linalg.lstsq(X * (y / np.sqrt(w))[:, None], np.log(y) * y / np.sqrt(w), rcond=None)
    r = least_squares(lambda t: (np.exp(X @ t) - y) / np.sqrt(w), th0, method="lm", max_nfev=20000)
    return np.exp(X @ r.x)


summary = {}
for M in masses:
    fs = glob.glob(f"{dd}/spec_r14A_m{M}_trig{trig}_*.npz")
    if not fs:
        continue
    a = np.load(fs[0], allow_pickle=True)
    b = np.linspace(0.4 * M, 2.0 * M, 49); c = 0.5 * (b[1:] + b[:-1])
    q, _ = np.histogram(a["qcd_mavg"], b, weights=a["qcd_w"] * L); q2, _ = np.histogram(a["qcd_mavg"], b, weights=(a["qcd_w"] * L) ** 2)
    s, _ = np.histogram(a["sig_mavg"], b, weights=a["sig_w"] * L)  # events per fb of signal cross section
    i0 = int(np.argmax(np.convolve(q, np.ones(3) / 3, mode="same")))
    ok = (np.arange(len(q)) >= i0) & (q > 0) & (q2 > 0)
    x = c[ok] / 13600.0
    bk = fit_bkg(q[ok], x, q2[ok])  # smooth background model from the MC
    A = basis(x) * np.sqrt(bk)[:, None]; v = s[ok] / np.sqrt(bk)
    r = v - A @ np.linalg.lstsq(A, v, rcond=None)[0]
    sig_mu = 1 / np.sqrt(r @ r)  # fb
    mu = 5 * sig_mu
    data = bk + mu * s[ok]
    fit0 = fit_bkg(data, x, data)
    pull = (data - fit0) / np.sqrt(fit0)
    summary[M] = {"sigma_mu_fb": float(sig_mu), "limit_fb": float(1.96 * sig_mu), "injected_fb": float(mu),
                  "signal_events": float(mu * s.sum()), "max_pull": float(pull.max()),
                  "absorbed_fraction": float(1 - np.sum(np.clip(data - fit0, 0, None)) / (mu * s[ok].sum()))}
    lab = f"{M} GeV, offline $H_T>{trig}$ GeV, {str(a['sel'])}"
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.axvspan(0.85 * M, 1.1 * M, color="#9b2c2c", alpha=0.07, lw=0)
    ax.stairs(q, b, color="0.55", lw=0.8, label="QCD MC")
    ax.plot(c[ok], data, "o", ms=2.2, color="0.1", label=f"pseudo-data: background + {mu:.3g} fb signal")
    ax.plot(c[ok], fit0, color="#2b5d8a", lw=1.0, label="background-only dijet fit")
    ax.stairs(mu * s, b, color="#9b2c2c", lw=1.2, label="injected signal")
    ax.set_yscale("log"); ax.set_xlabel(r"$m_{\rm avg}$ [GeV]"); ax.set_ylabel(r"events / bin, 300 fb$^{-1}$")
    ax.set_title(lab + r": signal at 5$\sigma_\mu$ of the shape fit", fontsize=6.3, loc="left")
    ax.legend(fontsize=6.0, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
    spines_to_data_range(ax)
    save_pdf(fig, f"{out}/bump_m{M}_trig{trig}.pdf"); fig.savefig(f"{out}/bump_m{M}_trig{trig}.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    ax.axvspan(0.85 * M, 1.1 * M, color="#9b2c2c", alpha=0.07, lw=0)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.bar(c[ok], pull, width=b[1] - b[0], color="#9b2c2c", alpha=0.75, lw=0)
    ax.set_xlabel(r"$m_{\rm avg}$ [GeV]"); ax.set_ylabel(r"(data $-$ fit) / $\sqrt{\rm fit}$")
    ax.set_title(lab + f": what survives the background fit ({100 * (1 - summary[M]['absorbed_fraction']):.0f}% of the injected signal)", fontsize=6.0, loc="left")
    spines_to_data_range(ax)
    save_pdf(fig, f"{out}/pull_m{M}_trig{trig}.pdf"); fig.savefig(f"{out}/pull_m{M}_trig{trig}.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    print(M, {k: round(v, 3) for k, v in summary[M].items()})
json.dump(summary, open(f"{out}/bump_trig{trig}.json", "w"), indent=1)
