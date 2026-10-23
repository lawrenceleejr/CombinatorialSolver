"""m_avg spectra behind each expected-limit point: QCD (merged held-out MC) and signal after the chosen selection
(A < 0.1, P(signal), boost cuts, offline H_T threshold; no window), 300 fb^-1, with a 4-parameter dijet-function fit
to the QCD spectrum and the counting window [0.85, 1.1] M. One file per (mass, H_T threshold).
usage: spectra_fit.py style dump_dir out model masses trigs ftag  (dump_dir holds one systematics setting)
"""
import glob, json, sys
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

style, dd, out, model = sys.argv[1:5]
masses = [int(x) for x in sys.argv[5].split(",")]; trigs = [float(x) for x in sys.argv[6].split(",")]
ftag = sys.argv[7] if len(sys.argv) > 7 else "f0"
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")
L = 300.0
th = json.load(open("/scratch/llee/pairnet/xs_theory/pp13600_gluino_NNLO+NNLL.json"))["data"]


def sigma(M):
    if M >= 500:
        return th[str(M)]["xsec_pb"] * 1e3
    fs = sorted(glob.glob(f"/scratch/llee/pairnet/prod_lowmass/siglm_m{M}_incl_c*.h5"))[:3]
    return 2.07 * np.mean([h5py.File(f).attrs["sigma_mb"] for f in fs]) * 1e12


def dijet(p, x):
    return np.exp(p[0]) * (1 - x) ** p[1] / x ** (p[2] + p[3] * np.log(x))


summary = {}
for M in masses:
    for t in trigs:
        fs = sorted(glob.glob(f"{dd}/spec_{model}_m{M}_trig{t:g}_*.npz"))  # one systematics setting per dump directory
        if not fs:
            continue
        a = np.load(fs[0], allow_pickle=True)
        bins = np.linspace(0.4 * M, 2.0 * M, 49); ctr = 0.5 * (bins[1:] + bins[:-1])
        qb, _ = np.histogram(a["qcd_mavg"], bins, weights=a["qcd_w"] * L)
        qe2, _ = np.histogram(a["qcd_mavg"], bins, weights=(a["qcd_w"] * L) ** 2)
        qn, _ = np.histogram(a["qcd_mavg"], bins)
        sb, _ = np.histogram(a["sig_mavg"], bins, weights=a["sig_w"] * sigma(M) * L)
        x = ctr / 13600.0
        i0 = int(np.argmax(qb))  # fit the falling part only, from the spectrum peak upward
        ok = (qn >= 3) & (np.arange(len(qb)) >= i0)
        on_tail = 0.85 * M >= bins[i0]
        res = None
        if ok.sum() >= 8:
            f = lambda p: (dijet(p, x[ok]) - qb[ok]) / np.sqrt(qe2[ok])
            res = least_squares(f, [np.log(max(qb[ok][0], 1e-9) * x[ok][0] ** 5), 10.0, 5.0, 0.0], method="lm", max_nfev=20000)
        win = (a["qcd_mavg"] > 0.85 * M) & (a["qcd_mavg"] < 1.1 * M)
        B = (a["qcd_w"][win] * L).sum(); S = (a["sig_w"][(a["sig_mavg"] > 0.85 * M) & (a["sig_mavg"] < 1.1 * M)] * sigma(M) * L).sum()
        fig, ax = plt.subplots(figsize=(4.6, 3.0))
        ax.axvspan(0.85 * M, 1.1 * M, color="#9b2c2c", alpha=0.07, lw=0)
        ax.stairs(qb, bins, color="0.3", lw=1.2, label=f"QCD MC ({int(qn.sum())} events)")
        if res is not None:
            chi2 = float((res.fun ** 2).sum()); ndf = int(ok.sum() - 4)
            ax.plot(ctr[ok], dijet(res.x, x[ok]), color="#2b5d8a", lw=1.0, ls="--", label=f"4-par. dijet fit from the peak, $\\chi^2$/ndf {chi2:.0f}/{ndf}")
            ax.axvline(bins[i0], color="#2b5d8a", lw=0.5, ls=":")
        else:
            chi2, ndf = None, None
        ax.stairs(sb, bins, color="#9b2c2c", lw=1.4, label=f"signal {M} GeV ($\\sigma\\times$300 fb$^{{-1}}$)")
        ax.set_yscale("log"); ax.set_xlabel(r"$m_{\rm avg}$ [GeV]"); ax.set_ylabel(r"events / bin, 300 fb$^{-1}$")
        cuts = json.loads(str(a["cuts"]))
        nm = {"psig": "P", "ht": "$H_T$", "delta_max": r"$\Delta_{\max}$", "delta_min": r"$\Delta_{\min}$", "pt_low6": r"$p_{T,6}$", "d32_max": r"$D^2_{[3,2]}$", "d2_63_32": r"$D^2_{[6,3]}$", "asym": "A", "u": "u"}
        cs = ", ".join(f"{nm[k]}$>${v:.3g}" if k == "psig" else f"{nm[k]}$>${v:.2g}$M$" if k in ("ht", "delta_max", "delta_min", "pt_low6") and v > -1e8 else f"{nm[k]}$<${v:.2g}" if k in ("d32_max", "d2_63_32", "asym", "u") and v < 1e8 else "" for k, v in cuts.items())
        cs = ", ".join(c for c in cs.split(", ") if c)
        ax.set_title(f"{model}, offline $H_T>{t:g}$ GeV: {str(a['sel'])} ({cs}); window S = {S:.3g}, B = {B:.3g}" + ("" if on_tail else "; window below the spectrum peak"), fontsize=5.6, loc="left")
        ax.legend(fontsize=6.2, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1))
        spines_to_data_range(ax)
        name = f"spec_{model}_m{M}_trig{t:g}_{ftag}"
        save_pdf(fig, f"{out}/{name}.pdf"); fig.savefig(f"{out}/{name}.png", dpi=170, bbox_inches="tight"); plt.close(fig)
        summary[name] = {"S": float(S), "B": float(B), "S_over_sqrtB": float(S / np.sqrt(max(B, 1e-9))), "chi2": chi2, "ndf": ndf, "selection": str(a["sel"]), "cuts": cuts, "fit_from_GeV": float(bins[i0]), "window_on_falling_tail": bool(on_tail)}
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary[name].items() if k != "cuts"}, cs)
json.dump(summary, open(f"{out}/spectra_{model}_{ftag}.json", "w"), indent=1)
