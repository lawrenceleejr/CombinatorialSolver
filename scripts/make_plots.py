"""
Report figures (Tufte style: one panel per file, range frames, direct labels).

  python scripts/make_plots.py --runs gated_n7=runs/gated_n7/eval.npz \
      legacy=runs/legacy_asis/eval.npz classical=runs/classical7/eval.npz --out plots/report

Each eval.npz is either train_pairnet.py output (best/target) or
eval_legacy.py output (correct).  Writes PDF + SVG for every figure.
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
import sys  # noqa: E402

sys.path.insert(0, HERE)
import plotstyle  # noqa: E402

plt = plotstyle.use()

INK = "#1f1f1f"
MUTED = "#8c8c8c"
ACCENT = "#c44e52"
BLUE = "#4c72b0"
PALETTE = [ACCENT, BLUE, MUTED, "#55a868", "#8172b3", "#ccb974"]


def range_frame(ax, x=None, y=None):
    if x is not None:
        ax.spines["bottom"].set_bounds(min(x), max(x))
    if y is not None:
        ax.spines["left"].set_bounds(min(y), max(y))


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def load(path):
    ev = dict(np.load(path))
    if "correct" not in ev:
        ev["correct"] = (ev["best"] == ev["target"]) & (ev["target"] >= 0)
    ev["is_signal"] = ev["is_signal"].astype(bool)
    return ev


def eff_vs_mass(runs, out):
    fig, ax = plt.subplots(figsize=(4.2, 3))
    allm, ally = [], []
    for i, (name, ev) in enumerate(runs.items()):
        s = ev["is_signal"]
        ms = np.unique(ev["pmass"][s])
        y = [ev["correct"][s & (ev["pmass"] == m)].mean() for m in ms]
        c = PALETTE[i % len(PALETTE)]
        ax.plot(ms, y, "-o", color=c, ms=3, lw=1)
        ax.text(ms[-1] * 1.02, y[-1], name, color=c, va="center", fontsize=8)
        allm += list(ms)
        ally += y
    ax.set_xlabel("gluino mass [GeV]")
    ax.set_ylabel("true interpretation chosen\n(fraction of all signal events)")
    ax.set_xticks(sorted(set(allm)))
    ax.set_xlim(min(allm) * 0.9, max(allm) * 1.25)
    top = np.ceil(max(ally) * 50) / 50
    ax.set_ylim(0, top * 1.05)
    ax.set_yticks(np.arange(0, top + 1e-9, 0.02))
    range_frame(ax, x=allm, y=[0, top])
    save(fig, out, "eff_vs_mass")


def preselection_curve(runs, out, window=(0.85, 1.1)):
    """Vary the asymmetry cut A: signal kept in mass window vs QCD kept."""
    fig, ax = plt.subplots(figsize=(4.2, 3))
    cuts = np.linspace(0.01, 0.5, 50)
    xs, ys = [], []
    for i, (name, ev) in enumerate(runs.items()):
        s = ev["is_signal"]
        if s.all():
            continue
        rel = ev["mavg"] / np.maximum(ev["pmass"], 1)
        inwin = (rel > window[0]) & (rel < window[1])
        w = ev["w"].astype(float)
        sig = [((ev["asym"] < a) & inwin)[s].mean() for a in cuts]
        bkg = [(w * (ev["asym"] < a))[~s].sum() / w[~s].sum() for a in cuts]
        c = PALETTE[i % len(PALETTE)]
        ax.plot(bkg, sig, color=c, lw=1.2)
        j = int(np.argmin(np.abs(cuts - 0.1)))
        ax.plot(bkg[j], sig[j], "o", color=c, ms=3)
        ax.text(bkg[-1] * 1.05, sig[-1], name, color=c, va="center", fontsize=8)
        xs += bkg
        ys += sig
    ax.set_xscale("log")
    ax.set_xlabel("QCD passing asymmetry cut (weighted fraction)")
    ax.set_ylabel("signal passing cut with\n$m_{avg}$ in [0.85, 1.1] M")
    ax.text(0.02, 0.98, r"dots: $A<0.1$", transform=ax.transAxes, fontsize=7, color=MUTED, va="top")
    ax.set_ylim(0, max(ys) * 1.08)
    range_frame(ax, x=[min(xs), max(xs)], y=[0, max(ys)])
    save(fig, out, "preselection_curve")


def mavg_hist(runs, out, cut=0.1):
    for name, ev in runs.items():
        s = ev["is_signal"]
        pas = ev["asym"] < cut
        fig, ax = plt.subplots(figsize=(4.2, 3))
        bins = np.linspace(0, 2500, 51)
        ctr = 0.5 * (bins[1:] + bins[:-1])
        ymax = 0
        for j, m in enumerate(np.unique(ev["pmass"][s])):
            sel = s & (ev["pmass"] == m) & pas
            h, _ = np.histogram(ev["mavg"][sel], bins)
            h = h / max(s[ev["pmass"] == m].sum(), 1)
            ax.step(ctr, h, where="mid", color=BLUE, lw=0.8, alpha=0.4 + 0.1 * j)
            ymax = max(ymax, h.max())
        if (~s).any():
            w = ev["w"].astype(float)
            h, _ = np.histogram(ev["mavg"][~s & pas], bins, weights=w[~s & pas])
            h = h / w[~s].sum()
            ax.step(ctr, h, where="mid", color=ACCENT, lw=1.2)
            k = int(np.argmax(h))
            ax.text(ctr[k] + 60, h[k], "QCD", color=ACCENT, fontsize=8)
            ymax = max(ymax, h.max())
        ax.text(0.98, 0.95, "signal, one curve per mass", color=BLUE, fontsize=8,
                transform=ax.transAxes, ha="right")
        ax.set_xlabel(r"$m_{avg}$ of chosen interpretation, $A<%.2g$ [GeV]" % cut)
        ax.set_ylabel("fraction of all events per 50 GeV")
        range_frame(ax, x=[0, 2500], y=[0, ymax])
        slug = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
        save(fig, out, f"mavg_{slug}")


def matching_breakdown(path, out):
    r = json.load(open(path))
    ms = sorted(int(k) for k in r)
    fig, ax = plt.subplots(figsize=(4.2, 3))
    keys = [("all_matched", "all six matched", INK),
            ("first_reason_merged", "merged", ACCENT),
            ("first_reason_soft", "soft (FSR)", BLUE),
            ("first_reason_forward", "forward", MUTED)]
    for k, lab, c in keys:
        y = [r[str(m)][k] for m in ms]
        ax.plot(ms, y, "-o", color=c, ms=3, lw=1)
        ax.text(ms[-1] * 1.02, y[-1], lab, color=c, va="center", fontsize=8)
    ax.set_xlabel("gluino mass [GeV]")
    ax.set_ylabel("fraction of signal events")
    ax.set_xticks(ms)
    ax.set_xlim(ms[0] * 0.9, ms[-1] * 1.3)
    ax.set_ylim(0, 0.75)
    range_frame(ax, x=ms, y=[0, 0.7])
    save(fig, out, "matching_breakdown")

    fig, ax = plt.subplots(figsize=(4.2, 3))
    for n, c in zip((6, 7, 8, 10), (MUTED, ACCENT, BLUE, INK)):
        y = [r[str(m)][f"reconstructable_top{n}"] for m in ms]
        ax.plot(ms, y, "-o", color=c, ms=3, lw=1)
        ax.text(ms[-1] * 1.02, y[-1], f"leading {n} jets", color=c, va="center", fontsize=8)
    ax.set_xlabel("gluino mass [GeV]")
    ax.set_ylabel("true 3+3 interpretation available")
    ax.set_xticks(ms)
    ax.set_xlim(ms[0] * 0.9, ms[-1] * 1.35)
    ax.set_ylim(0, 0.7)
    range_frame(ax, x=ms, y=[0, 0.7])
    save(fig, out, "truth_ceiling")


def massvar_fom(vars_path, out, asym_cut=0.1):
    """S/sqrt(B) in the shortest 68% signal window, relative to m_avg alone."""
    sys.path.insert(0, HERE)
    from mass_variables_fom import shortest

    d = np.load(vars_path)
    s = d["is_signal"].astype(bool)
    pre = d["asym"] < asym_cut
    w = d["w"]
    wq = w[~s].sum()
    floor = 3 * w[~s].mean() / wq
    ms = np.unique(d["pmass"][s])

    def fom(v, extra=None):
        out_ = []
        for m in ms:
            sm = s & (d["pmass"] == m)
            base = pre.copy()
            if extra == "ht90":
                base &= d["HT"] >= np.quantile(d["HT"][sm & pre], 0.10)
            lo, hi = shortest(d[v][sm & base])
            sel = base & (d[v] >= lo) & (d[v] <= hi)
            S = (sm & sel).sum() / sm.sum()
            B = (w * (~s & sel)).sum() / wq
            out_.append(S / np.sqrt(max(B, floor)))
        return np.array(out_)

    ref = fom("m3")
    curves = [("m3", None, r"$m_{avg}$ of chosen 3+3", INK),
              ("dressed0.8", None, r"dressed, $\Delta R<0.8$", BLUE),
              ("thrust", None, "thrust hemispheres", MUTED),
              ("HT", None, r"$H_T$ window", "#55a868"),
              ("m3", "ht90", r"$m_{avg}$ window + loose $H_T$ floor", ACCENT)]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    sel = ms < 1900  # 2 TeV point is limited by QCD MC statistics
    ys = []
    for v, ex, lab, c in curves:
        y = fom(v, ex) / ref
        ax.plot(ms[sel], y[sel], "-o", color=c, ms=3, lw=1.1)
        ax.text(ms[sel][-1] * 1.03, y[sel][-1], lab, color=c, va="center", fontsize=8)
        ys += list(y[sel])
    ax.axhline(1, color=MUTED, lw=0.5, ls=(0, (2, 2)))
    ax.set_xlabel("gluino mass [GeV]")
    ax.set_ylabel(r"$S/\sqrt{B}$ relative to $m_{avg}$ alone")
    ax.set_xticks(ms[sel])
    ax.set_xlim(ms[0] * 0.9, ms[sel][-1] * 1.9)
    range_frame(ax, x=list(ms[sel]), y=[min(ys), max(ys)])
    save(fig, out, "massvar_fom")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--matching", default="results/matching.json")
    ap.add_argument("--out", default="plots/report")
    ap.add_argument("--massvars", default=None)
    a = ap.parse_args()
    runs = {}
    for r in a.runs:
        k, p = r.split("=")
        runs[k] = load(p)
    if os.path.exists(a.matching):
        matching_breakdown(a.matching, a.out)
    if a.massvars:
        massvar_fom(a.massvars, a.out)
    if runs:
        eff_vs_mass(runs, a.out)
        preselection_curve(runs, a.out)
        mavg_hist(runs, a.out)


if __name__ == "__main__":
    main()
