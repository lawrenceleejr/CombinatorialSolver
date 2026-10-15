"""Shared Tufte style for report figures: EB Garamond (vendored in assets/fonts)."""
import os

import matplotlib
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def use():
    import matplotlib.pyplot as plt

    plt.style.use(os.path.join(ROOT, "assets", "tufte.mplstyle"))
    fdir = os.path.join(ROOT, "assets", "fonts", "ebgaramond")
    for f in os.listdir(fdir):
        if f.endswith(".otf"):
            font_manager.fontManager.addfont(os.path.join(fdir, f))
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["EB Garamond 12", "EB Garamond", "cmr10"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "EB Garamond 12",
        "mathtext.it": "EB Garamond 12:italic",
        "mathtext.bf": "EB Garamond 12:bold",
        "font.size": 11,
        "axes.labelsize": 11,
        "savefig.dpi": 220,
    })
    return plt
