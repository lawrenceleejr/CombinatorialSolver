"""Spurious-signal check of the 4-parameter dijet function on the QCD MC after each shape-fit-tuned selection.

Signal + background fit to the QCD-only m_avg spectrum (MC weights and their errors, 300 fb^-1 normalisation), from the
QCD peak to 2 M; the fitted signal cross section (fb) is compared with the expected limit. With the MC errors much
larger than 300 fb^-1 data errors this tests the function only to MC precision. Single-panel plot + json.
usage: spurious.py style out trig dump_dir limits_json masses [high_dump_dir high_json high_masses]
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

style, out, trig, dd, lj = sys.argv[1:6]; masses = [int(x) for x in sys.argv[6].split(",")]
extra = sys.argv[7:10] if len(sys.argv) > 9 else None
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")
L = 300.0; FB = np.linspace(0.4, 2.0, 49); FC = 0.5 * (FB[1:] + FB[:-1])


NPAR = int(__import__("os").environ.get("NPAR", "4")); OFFSET = int(__import__("os").environ.get("FIT_OFFSET", "0"))
TAG = __import__("os").environ.get("TAG", "")


def basis(x):
    cols = [np.ones_like(x), np.log1p(-x), np.log(x), np.log(x) ** 2, np.log(x) ** 3][:NPAR]
    return np.stack(cols, 1)


def lim_low(x):
    return min(x[k]["limit_fb"] for k in ("P", "P+boost", "P+all") if k in x)


def run(M, path, limit):
    a = np.load(path, allow_pickle=True)
    q, _ = np.histogram(a["qcd_mavg"] / M, FB, weights=a["qcd_w"] * L); q2, _ = np.histogram(a["qcd_mavg"] / M, FB, weights=(a["qcd_w"] * L) ** 2)
    s, _ = np.histogram(a["sig_mavg"] / M, FB, weights=a["sig_w"] * L)  # events per fb
    i0 = int(np.argmax(np.convolve(q, np.ones(3) / 3, mode="same"))); ok = (np.arange(48) >= i0 + OFFSET) & (q > 0) & (q2 > 0)
    X = basis(FC[ok] * M / 13600.0); e = np.sqrt(q2[ok])
    th0, *_ = np.linalg.lstsq(X * (q[ok] / e)[:, None], np.log(q[ok]) * q[ok] / e, rcond=None)
    r = least_squares(lambda p: (np.exp(X @ p) - q[ok]) / e, th0, method="lm", max_nfev=20000)  # background-only fit
    b = np.exp(X @ r.x)
    # signal strength linearised around the background-only fit, background parameters profiled (MC errors)
    A = (X * b[:, None]) / e[:, None]; v = s[ok] / e; d = (q[ok] - b) / e
    proj = lambda y: y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    pv = proj(v); mu = float(pv @ proj(d) / (pv @ pv)); err = float(1 / np.sqrt(pv @ pv))
    return {"mu_ss_fb": mu, "mu_ss_err_fb": err, "expected_limit_fb": limit, "ss_over_limit": mu / limit, "err_over_limit": err / limit,
            "chi2": float((r.fun ** 2).sum()), "ndf": int(ok.sum() - NPAR), "sigma_mu_fit_fb": None}


res = {}
lows = json.load(open(lj))["results"]
for M in masses:
    res[M] = run(M, f"{dd}/spec_r14A_m{M}_trig{trig}_f0.npz", lim_low(lows[str(M)]))
if extra:
    hd, hj, hm = extra; hr = json.load(open(hj))["results"]
    for M in [int(x) for x in hm.split(",")]:
        res[M] = run(M, f"{hd}/spec_r14A_m{M}_trig450_fisher.npz", hr[str(M)]["limit_fb"])
fig, ax = plt.subplots(figsize=(4.6, 3.0))
ms = sorted(res); y = np.array([res[m]["ss_over_limit"] for m in ms]); ye = np.array([res[m]["err_over_limit"] for m in ms])
ax.axhspan(-0.5, 0.5, color="0.9", lw=0); ax.axhline(0, color="0.6", lw=0.6)
ax.errorbar(ms, y, ye, fmt="o", ms=3, color="#9b2c2c", lw=1, capsize=0)
ax.set_xlabel(r"$m_{\tilde g}$ [GeV]"); ax.set_ylabel("fitted signal in QCD-only MC / expected limit")
ax.set_title(f"spurious signal of the {NPAR}-parameter dijet fit (start: QCD peak + {OFFSET} bins), shape-fit-tuned selections (error bars: QCD MC statistics)", fontsize=6.0, loc="left")
spines_to_data_range(ax)
save_pdf(fig, f"{out}/spurious{TAG}.pdf"); fig.savefig(f"{out}/spurious{TAG}.png", dpi=170, bbox_inches="tight")
json.dump(res, open(f"{out}/spurious{TAG}.json", "w"), indent=1)
for m in ms:
    print(m, "S_ss = %.3g +- %.3g fb, limit %.3g fb -> %.2f +- %.2f of the limit, chi2/ndf %.0f/%d" % (res[m]["mu_ss_fb"], res[m]["mu_ss_err_fb"], res[m]["expected_limit_fb"], res[m]["ss_over_limit"], res[m]["err_over_limit"], res[m]["chi2"], res[m]["ndf"]))
