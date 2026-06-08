"""
Plotting for the bump-hunt cut analysis: ROC scans, combined working-point
clouds, cutflow bars, and the average-mass bump-hunt histogram.

These functions are shared by the standalone tool (``src.analysis``) and the
training-time ROC animation (``src.train``) so a single definition drives both.
All functions take an existing matplotlib ``Axes`` where practical, so they can
be composed into multi-panel figures or animation frames.
"""

from __future__ import annotations

import numpy as np

from .cut_analysis import (
    OBS_LABELS,
    Observables,
    combined_mask,
    cutflow,
    roc_curves,
    working_point_cloud,
)
from .plot_style import HIST_COLORS, edge, init_style, rgba, style_axis

# Distinct colours for the per-variable ROC curves.
_ROC_PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def draw_roc_curves(ax, sig: Observables, bkg: Observables, cuts: dict,
                    variables: list[str] | None = None, n: int = 120) -> None:
    """Per-variable ROC curves (signal efficiency vs background rejection)."""
    rocs = roc_curves(sig, bkg, variables=variables, cuts=cuts, n=n)
    for i, (v, r) in enumerate(rocs.items()):
        color = _ROC_PALETTE[i % len(_ROC_PALETTE)]
        ax.plot(r["sig_eff"], r["bkg_rej"], color=color, linewidth=1.8,
                label=f"{v}  (AUC {r['auc']:.2f})")
    ax.plot([0, 1], [1, 0], color="#BBBBBB", linewidth=1.0, linestyle=":", label="random")
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background rejection  $1-\\varepsilon_{\\mathrm{bkg}}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("ROC: single-cut scans", loc="left")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    style_axis(ax)


def draw_working_point_cloud(ax, sig: Observables, bkg: Observables, cuts: dict,
                             n_per_axis: int = 6) -> None:
    """Scatter of many combined cut working points with the Pareto front and the
    nominal working point highlighted."""
    wp = working_point_cloud(sig, bkg, cuts=cuts, n_per_axis=n_per_axis)
    ax.scatter(wp["sig_eff"], wp["bkg_rej"], s=6, c="#BFD9EC",
               edgecolors="none", alpha=0.6, label="cut combinations")
    pf = wp["pareto"]
    order = np.argsort(wp["sig_eff"][pf])
    ax.plot(wp["sig_eff"][pf][order], wp["bkg_rej"][pf][order],
            color=HIST_COLORS["accent"], linewidth=1.8, marker="o",
            markersize=3, markeredgecolor="white", markeredgewidth=0.5,
            label="Pareto front")

    # Nominal working point (the DEFAULT_CUTS thresholds).
    s_mask = combined_mask(sig, cuts)
    b_mask = combined_mask(bkg, cuts)
    s_eff = sig.weights[s_mask].sum() / sig.weights.sum()
    b_rej = 1.0 - bkg.weights[b_mask].sum() / bkg.weights.sum()
    ax.scatter([s_eff], [b_rej], s=90, marker="*", color=HIST_COLORS["qcd"],
               edgecolors="black", linewidths=0.6, zorder=5,
               label=f"nominal ({s_eff:.2f}, {b_rej:.2f})")
    ax.set_xlabel("Signal efficiency")
    ax.set_ylabel("Background rejection")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Combined working points", loc="left")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    style_axis(ax)


def draw_mass_bump(ax, sig: Observables, bkg: Observables, cuts: dict,
                   n_bins: int = 40, mass_range: tuple[float, float] | None = None,
                   logy: bool = False) -> None:
    """Average-mass bump-hunt histogram: signal (filled) over QCD (outline),
    showing the spectrum BEFORE (faint) and AFTER (bold) the combined cuts."""
    s_all, b_all = sig.avg_mass, bkg.avg_mass
    s_w, b_w = sig.weights, bkg.weights
    if mass_range is None:
        allm = np.concatenate([s_all, b_all])
        mass_range = (float(np.percentile(allm, 0.5)), float(np.percentile(allm, 99.5)))
    edges = np.linspace(*mass_range, n_bins + 1)

    s_sel = combined_mask(sig, cuts)
    b_sel = combined_mask(bkg, cuts)

    def _h(vals, w, m=None):
        return np.histogram(vals if m is None else vals[m],
                            bins=edges, weights=w if m is None else w[m])[0]

    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    # Before cuts: faint reference.
    ax.bar(centers, _h(b_all, b_w), width=width, align="center", linewidth=0,
           facecolor=rgba(HIST_COLORS["qcd"], 0.12))
    ax.bar(centers, _h(s_all, s_w), width=width, align="center", linewidth=0,
           facecolor=rgba(HIST_COLORS["signal"], 0.12))

    # After cuts: bold.
    ax.bar(centers, _h(b_all, b_w, b_sel), width=width, align="center", linewidth=0,
           facecolor=rgba(HIST_COLORS["qcd"], 0.55), label="QCD (after cuts)")
    ax.stairs(_h(b_all, b_w, b_sel), edges, color=edge(HIST_COLORS["qcd"]), linewidth=1.2)
    ax.bar(centers, _h(s_all, s_w, s_sel), width=width, align="center", linewidth=0,
           facecolor=rgba(HIST_COLORS["signal"], 0.7), label="Signal (after cuts)")
    ax.stairs(_h(s_all, s_w, s_sel), edges, color=edge(HIST_COLORS["signal"]), linewidth=1.2)

    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(OBS_LABELS["avg_mass"])
    ax.set_ylabel("Events (weighted)")
    ax.set_title("Average-mass bump hunt (faint = before cuts)", loc="left")
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    style_axis(ax, grid_axis="y")


def draw_cutflow(ax, rows: list[dict]) -> None:
    """Signal- and background-fraction-remaining bar chart across cut stages."""
    names = [r["cut"] for r in rows]
    s_frac = [r["sig_frac"] for r in rows]
    b_frac = [r["bkg_frac"] for r in rows]
    y = np.arange(len(rows))
    ax.barh(y - 0.2, s_frac, height=0.38, color=HIST_COLORS["signal"], label="signal frac")
    ax.barh(y + 0.2, b_frac, height=0.38, color=HIST_COLORS["qcd"], label="bkg frac")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Fraction remaining")
    ax.set_xlim(0, 1.02)
    ax.set_title("Cutflow (fraction remaining)", loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    style_axis(ax, grid_axis="x")


def make_summary_figure(sig: Observables, bkg: Observables, cuts: dict,
                        out_path, title: str | None = None,
                        n_per_axis: int = 6, logy: bool = False):
    """Compose the four-panel analysis summary (ROC, working points, cutflow,
    bump hunt) and save to ``out_path``.  Returns the path, or ``None`` if
    matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        init_style(plt)
    except ImportError:
        print("  Warning: matplotlib not available; skipping summary figure.")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    draw_roc_curves(axes[0, 0], sig, bkg, cuts)
    draw_working_point_cloud(axes[0, 1], sig, bkg, cuts, n_per_axis=n_per_axis)
    draw_cutflow(axes[1, 0], cutflow(sig, bkg, cuts))
    draw_mass_bump(axes[1, 1], sig, bkg, cuts, logy=logy)
    if title:
        fig.suptitle(title, fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(str(out_path))
    plt.close(fig)
    return out_path
