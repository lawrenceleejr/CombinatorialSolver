"""
Shared plot styling: the Okabe–Ito palette and the Tufte-flavoured serif style
used across the project's figures.  Factored out so the standalone analysis
plots (``src.cut_plots``) match the training animations without importing the
heavy ``src.train`` module.

``src.train`` keeps its own private copies of these helpers for backward
compatibility; this module is the shared home for new plotting code.
"""

from __future__ import annotations

from pathlib import Path

# Okabe–Ito colourblind-safe palette (https://jfly.uni-koeln.de/color/).
HIST_COLORS = {
    "signal_correct": "#009E73",  # bluish green — correct interpretation
    "signal_wrong":   "#CFCFCF",  # light grey — wrong interpretation
    "signal":         "#009E73",
    "qcd":            "#D55E00",  # vermillion — QCD background
    "mean":           "#2A2A2A",
    "accent":         "#0072B2",  # blue — ROC / highlight
    "accent2":        "#CC79A7",  # reddish purple — secondary
}

_FONTS_REGISTERED = False


def ensure_serif_fonts() -> None:
    """Register the OFL EB Garamond fonts vendored in ``assets/fonts/`` so plots
    use them without a system install or network.  Best-effort and idempotent."""
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return
    _FONTS_REGISTERED = True
    try:
        import matplotlib.font_manager as fm
        have = {f.name for f in fm.fontManager.ttflist}
        if {"EB Garamond", "Garamond"} & have:
            return
        font_dir = Path(__file__).resolve().parent.parent / "assets" / "fonts"
        bundled = sorted(font_dir.rglob("*.otf")) + sorted(font_dir.rglob("*.ttf"))
        for p in bundled:
            try:
                fm.fontManager.addfont(str(p))
            except Exception:
                pass
    except Exception:
        pass


def init_style(plt) -> None:
    """Apply the global serif / Tufte rcParams.  Idempotent."""
    ensure_serif_fonts()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["EB Garamond", "Garamond", "Adobe Garamond Pro",
                       "Times New Roman", "Times", "Nimbus Roman No9 L",
                       "Liberation Serif", "DejaVu Serif"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "mathtext.fontset": "cm",
        "axes.titlepad": 14,
        "axes.unicode_minus": False,
    })


def style_axis(ax, grid_axis: str = "both") -> None:
    """Drop top/right spines, lighten the rest, faint gridlines behind data."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#777777")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors="#555555", length=3, width=0.8)
    ax.grid(True, axis=grid_axis, color="#E9E9E9", linewidth=0.6)
    ax.set_axisbelow(True)


def rgba(color, alpha: float):
    from matplotlib.colors import to_rgba
    return to_rgba(color, alpha)


def edge(color, factor: float = 0.55):
    from matplotlib.colors import to_rgba
    r, g, b = to_rgba(color)[:3]
    return (r * factor, g * factor, b * factor)
