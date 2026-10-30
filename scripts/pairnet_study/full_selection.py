"""Full analysis selection from a run's test-set eval.npz:
m_avg window [0.85, 1.1] M, A < 0.1, and a per-mass P(signal) threshold chosen to maximise S/sqrt(B)
(requiring >= 20 surviving QCD MC events). Also the QCD efficiency of A < 0.1 (+ a fixed P cut) versus
m_avg, to check that the selection does not sculpt the background."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

style, out = sys.argv[1], sys.argv[2]; runs = sys.argv[3:]  # name=path/to/eval.npz
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")

MASSES = list(range(500, 2401, 100))
res = {}
fig, ax = plt.subplots(figsize=(4.8, 3.1))
cols = ["0.1", "#b2182b", "#2166ac", "#1b7837", "#762a83"]
for i, spec in enumerate(runs):
    name, path = spec.split("=", 1)
    e = np.load(path); s = e["is_signal"]; q = ~s; w = e["w"]; W = w[q].sum()
    A = e["asym"] < 0.1; x = e["mavg"]; p = e["psig"]
    rows = {}
    for M in MASSES:
        sm = s & (e["pmass"] == M)
        if not sm.any():
            continue
        win = (x > 0.85 * M) & (x < 1.1 * M)
        S0 = (A & win)[sm].mean(); B0 = (w * (A & win))[q].sum() / W
        best = (S0 / np.sqrt(B0), 1.0, S0, B0, int((A & win & q).sum()))
        for eff in np.linspace(0.3, 0.98, 35):
            t = np.quantile(p[sm], 1 - eff)
            sel = A & win & (p >= t)
            nb = int(sel[q].sum())
            if nb < 20:
                continue
            S = sel[sm].mean(); B = (w * sel)[q].sum() / W
            if S / np.sqrt(B) > best[0]:
                best = (S / np.sqrt(B), eff, S, B, nb)
        rows[M] = {"fom_asym_only": S0 / np.sqrt(B0), "fom_best": best[0], "p_cut_sig_eff": best[1],
                   "S": best[2], "B": best[3], "n_qcd": best[4]}
    res[name] = rows
    g = np.exp(np.mean([np.log(r["fom_best"]) for r in rows.values()]))
    g0 = np.exp(np.mean([np.log(r["fom_asym_only"]) for r in rows.values()]))
    print(f"{name:12s} geometric-mean S/sqrtB over {len(rows)} masses: A<0.1 only {g0:.2f}, with per-mass P cut {g:.2f}")
    print("   " + " ".join(f"{M}:{r['fom_best']:.1f}({r['p_cut_sig_eff']:.2f})" for M, r in rows.items() if M % 500 == 0))
    # sculpting: QCD efficiency of A<0.1 and of A<0.1 + P cut at 70% overall signal efficiency, vs m_avg
    bins = np.linspace(300, 2600, 24); c = 0.5 * (bins[1:] + bins[:-1])
    tot, _ = np.histogram(x[q], bins, weights=w[q])
    t70 = np.quantile(p[s], 0.3)
    for sel, ls, lab in ((A, "-", r"$A<0.1$"), (A & (p >= t70), "--", r"$A<0.1$ + P(signal) cut")):
        h, _ = np.histogram(x[q & sel], bins, weights=w[q & sel])
        ax.plot(c, np.where(tot > 0, h / np.maximum(tot, 1e-30), np.nan), color=cols[i % 5], ls=ls, lw=1.1,
                label=f"{name}: {lab}")
ax.set_yscale("log"); ax.set_xlabel(r"$m_{\rm avg}$ of the chosen pairing [GeV]"); ax.set_ylabel("QCD efficiency (weighted)")
ax.legend(frameon=False, fontsize=6.5)
spines_to_data_range(ax)
save_pdf(fig, f"{out}/qcd_eff_vs_mavg.pdf"); fig.savefig(f"{out}/qcd_eff_vs_mavg.png", dpi=150, bbox_inches="tight")
json.dump(res, open(f"{out}/full_selection.json", "w"), indent=1, default=float)
