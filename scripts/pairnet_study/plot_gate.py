"""Gate output for QCD the model trained on vs QCD it never saw, epoch 2 vs epoch 39 (one PDF per epoch)."""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
style, npz, outdir = sys.argv[1:4]
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")
d = np.load(npz)
bins = np.linspace(-8, 12, 61)
logit = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
for ep in ("ep2", "ep39"):
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for key, lab, kw in (("qcd_seen", "QCD, trained on", dict(color="0.15", lw=1.2)),
                         ("qcd_unseen", "QCD, unseen", dict(color="#b2182b", lw=1.2)),
                         ("sig_test", "signal (test)", dict(color="0.6", lw=1.0, ls="--"))):
        x = logit(d[f"{ep}/{key}/psig"])
        h, e = np.histogram(x, bins=bins, density=True)
        c = 0.5 * (e[1:] + e[:-1])
        ax.step(c, h, where="mid", **kw)
        i = int(np.argmax(h))
        ax.annotate(lab, (c[i], h[i]), xytext=(3, 3), textcoords="offset points", fontsize=7.5, color=kw["color"])
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.set_xlabel("gate output, logit P(signal)")
    ax.set_ylabel("density")
    ax.set_title(f"epoch {ep[2:]}", fontsize=9, loc="left")
    spines_to_data_range(ax)
    save_pdf(fig, f"{outdir}/gate_seen_vs_unseen_{ep}.pdf")
    fig.savefig(f"{outdir}/gate_seen_vs_unseen_{ep}.png", dpi=150, bbox_inches="tight")
print("ok")
