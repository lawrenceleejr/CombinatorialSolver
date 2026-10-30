"""m_est vs m_avg of the chosen pairing for r5D (test set): signal resolution per mass, signal in window,
and the QCD distributions (weighted) after A < 0.1, to look for sculpting by the learned estimate."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
style, npz, out = sys.argv[1:4]
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")
e = np.load(npz); s = e["is_signal"]; q = ~s; A = e["asym"] < 0.1

def hw(x):
    x = np.sort(x); k = int(0.68 * len(x)); i = int(np.argmin(x[k:] - x[:-k])); return (x[i + k] - x[i]) / (x[i + k] + x[i])
print("mass  | 68% half-width m_avg  m_est | in window [0.85,1.1]M & A<0.1: m_avg  m_est")
for M in (500, 1000, 1500, 2000):
    sel = s & (e["pmass"] == M)
    ra, re = e["mavg"][sel] / M, e["m_est"][sel] / M
    wa = (A[sel] & (ra > 0.85) & (ra < 1.1)).mean(); we = (A[sel] & (re > 0.85) & (re < 1.1)).mean()
    print(f"{M:5d} |            {hw(ra):.3f}  {hw(re):.3f} |                              {wa:.3f}  {we:.3f}")
ra, re = e["mavg"][s] / e["pmass"][s], e["m_est"][s] / e["pmass"][s]
print(f"all   |            {hw(ra):.3f}  {hw(re):.3f} |                              {(A[s] & (ra > 0.85) & (ra < 1.1)).mean():.3f}  {(A[s] & (re > 0.85) & (re < 1.1)).mean():.3f}")
print("median m_est/m_avg on signal", np.median(re / ra), " on QCD", np.median(e["m_est"][q] / e["mavg"][q]))

bins = np.linspace(200, 3000, 57)
for name, var in (("m_avg", e["mavg"]), ("m_est", e["m_est"])):
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for lab, sel, kw in (("QCD, all", q, dict(color="0.6", lw=1.0)), ("QCD, A < 0.1", q & A, dict(color="0.1", lw=1.3))):
        h, ed = np.histogram(var[sel], bins=bins, weights=e["w"][sel], density=True)
        ax.step(0.5 * (ed[1:] + ed[:-1]), h, where="mid", **kw)
        i = int(np.argmax(h)); ax.annotate(lab, (0.5 * (ed[i] + ed[i + 1]), h[i]), xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=kw["color"])
    for M in (500, 1000, 1500, 2000):
        ax.axvline(M, color="#b2182b", lw=0.4, alpha=0.5)
    ax.set_yscale("log"); ax.set_xlabel(f"{name} [GeV]"); ax.set_ylabel("weighted QCD density")
    ax.text(0.98, 0.95, "red lines: signal mass points", transform=ax.transAxes, ha="right", va="top", fontsize=7, color="#b2182b")
    spines_to_data_range(ax)
    save_pdf(fig, f"{out}/qcd_{name}.pdf"); fig.savefig(f"{out}/qcd_{name}.png", dpi=150, bbox_inches="tight")
