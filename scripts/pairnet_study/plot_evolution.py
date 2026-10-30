"""Per-epoch evolution plots from metrics.json history, one PDF per panel."""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

style, metrics, outdir = sys.argv[1:4]
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")

h = json.load(open(metrics))["history"]
ep = [r["epoch"] for r in h]
SEL = 2  # epoch kept by --select combined
INK, MUTED, ACCENT = "0.15", "0.6", "#b2182b"


def series(key):
    if key == "score":
        return [r["eff_on_all_signal"] + 1 - r["bkg_pass_asym"] for r in h]
    if key.startswith("s@"):
        return [r["sig_in_window_at_bkg"][key[2:]] for r in h]
    return [r[key] for r in h]


def line_plot(name, key, ylabel, fmt="{:.3f}", band=None, mark=True):
    y = series(key)
    fig, ax = plt.subplots(figsize=(4.5, 3))
    if band:
        ax.axhspan(*band, color="0.92", lw=0, zorder=0)
    ax.plot(ep, y, color=INK, lw=1.0, marker="o", ms=2.2, zorder=2)
    if mark:
        ax.plot([SEL], [y[SEL]], "o", ms=5, mfc="none", mec=ACCENT, mew=1.2, zorder=3)
        ax.annotate(f"epoch {SEL} (selected)", (SEL, y[SEL]), xytext=(6, 4), textcoords="offset points",
                    fontsize=7.5, color=ACCENT)
    ax.annotate(fmt.format(y[-1]), (ep[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=7.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    spines_to_data_range(ax)
    save_pdf(fig, f"{outdir}/{name}.pdf")
    fig.savefig(f"{outdir}/{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


line_plot("evo_selection_score", "score", "selection score\n(sig eff + 1 $-$ QCD pass)")
line_plot("evo_qcd_pass_asym", "bkg_pass_asym", "QCD passing asym $<$ 0.1\n(weighted fraction, validation)")
line_plot("evo_sig_pass_asym", "sig_pass_asym", "signal passing asym $<$ 0.1\n(validation)")
line_plot("evo_eff_all_signal", "eff_on_all_signal", "exact pairing, all signal\n(validation)")
line_plot("evo_auc", "auc", "gate P(signal) AUC\n(validation)", fmt="{:.4f}")
line_plot("evo_rej50", "bkg_rej_at_50", "QCD rejection at 50%\nsignal efficiency", fmt="{:.0f}")
line_plot("evo_sig_at_5pct_qcd", "s@0.05", "signal in mass window\nat 5% QCD pass (asym scan)")

# Across epochs: QCD pass rate vs signal pass rate (the operating point moves along a line)
fig, ax = plt.subplots(figsize=(4, 3.2))
xb, ys = series("bkg_pass_asym"), series("sig_pass_asym")
sc = ax.scatter(xb, ys, c=ep, cmap="Greys", vmin=-8, vmax=39, s=14, edgecolors=INK, linewidths=0.4)
ax.plot([xb[SEL]], [ys[SEL]], "o", ms=7, mfc="none", mec=ACCENT, mew=1.2)
ax.annotate("epoch 2", (xb[SEL], ys[SEL]), xytext=(6, -10), textcoords="offset points", fontsize=7.5, color=ACCENT)
ax.annotate("epoch 39", (xb[-1], ys[-1]), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
ax.set_xlabel("QCD passing asym $<$ 0.1")
ax.set_ylabel("signal passing asym $<$ 0.1")
spines_to_data_range(ax)
save_pdf(fig, f"{outdir}/evo_operating_point.pdf")
fig.savefig(f"{outdir}/evo_operating_point.png", dpi=150, bbox_inches="tight")
print("done")
