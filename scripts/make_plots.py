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


def spread_labels(ax, items, x, min_gap):
    """Direct labels at the line ends, nudged apart so they never overlap.
    items: list of (y, text, color)."""
    items = sorted(items)
    ys = [y for y, _, _ in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    for (y0, t, c), y in zip(items, ys):
        ax.text(x, y, t, color=c, va="center", fontsize=8)


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
    allm, ally, ends = [], [], []
    for i, (name, ev) in enumerate(runs.items()):
        s = ev["is_signal"]
        ms = np.unique(ev["pmass"][s])
        y = [ev["correct"][s & (ev["pmass"] == m)].mean() for m in ms]
        c = PALETTE[i % len(PALETTE)]
        ax.plot(ms, y, "-o", color=c, ms=3, lw=1)
        ends.append((y[-1], name, c))
        allm += list(ms)
        ally += y
    spread_labels(ax, ends, max(allm) * 1.02, max(ally) * 0.05)
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
        for logy in (False, True):
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
                ax.text(ctr[k] + 60, h[k] * (1.3 if logy else 1.0), "QCD", color=ACCENT, fontsize=8)
                ymax = max(ymax, h.max())
            ax.text(0.98, 0.95, "signal, one curve per mass", color=BLUE, fontsize=8,
                    transform=ax.transAxes, ha="right")
            ax.set_xlabel(r"$m_{avg}$ of chosen interpretation, $A<%.2g$ [GeV]" % cut)
            ax.set_ylabel("fraction of all events per 50 GeV")
            if logy:
                ax.set_yscale("log")
                ax.set_ylim(1e-5, ymax * 2)
                ax.spines["bottom"].set_bounds(0, 2500)
            else:
                range_frame(ax, x=[0, 2500], y=[0, ymax])
            slug = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_")
            save(fig, out, f"mavg_{slug}" + ("_log" if logy else ""))


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


def strategy_fom(path, out, methods=None, name="strategy_fom"):
    """S/sqrt(B) (best cut and window) vs mass for search strategies, log scale."""
    r = json.load(open(path))
    default = [("argmin |Δm|, asym cut", "argmin |Δm| + asym cut", MUTED),
               ("current network, asym cut", "current network + asym cut", "#b9b3a3"),
               ("expert BDT, argmin pairing", "expert-feature BDT", BLUE),
               ("gated PairNet, gate cut", "gated PairNet, gate cut", ACCENT),
               ("gated PairNet, gate cut, TRUTH pairing (bound)", "same, true pairing (bound)", INK)]
    methods = methods or default
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    ends, ally, allm = [], [], []
    for key, lab, c in methods:
        if key not in r:
            continue
        ms = sorted(int(k) for k in r[key] if k.isdigit())
        y = [r[key][str(m)]["fom"] for m in ms]
        ls = (0, (2, 2)) if "bound" in lab else "-"
        ax.plot(ms, y, marker="o", color=c, ms=3, lw=1.1, ls=ls)
        ends.append((np.log10(y[-1]), lab, c))
        ally += y
        allm += ms
    ax.set_yscale("log")
    # direct labels in log space, nudged apart
    items = sorted(ends)
    ys = [v for v, _, _ in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + 0.09)
    for (_, t, c), yv in zip(items, ys):
        ax.text(max(allm) * 1.03, 10**yv, t, color=c, va="center", fontsize=7.5)
    ax.set_xlabel("gluino mass [GeV]")
    ax.set_ylabel(r"$S/\sqrt{B}$, relative units (best cut and window)")
    ax.set_xticks([500, 1000, 1500, 2000])
    ax.set_xlim(min(allm) * 0.9, max(allm) * 1.75)
    ax.spines["bottom"].set_bounds(min(allm), max(allm))
    save(fig, out, name)


ROBUST_TITLES = {"signal, chain500": "signal: g̃ → q̃q, q̃ → qq",
                 "signal, as150": "signal: shower αs +10%", "signal, as123": "signal: shower αs −10%",
                 "QCD, as150": "QCD: shower αs +10%", "QCD, as123": "QCD: shower αs −10%",
                 "signal, jes-3": "signal: jet energy −3%", "QCD, jes-3": "QCD: jet energy −3%",
                 "signal, jes-fwd": "signal: −5% at |η| > 1.5", "QCD, jes-fwd": "QCD: −5% at |η| > 1.5",
                 "signal, jer+5": "signal: +5% jet smearing", "QCD, jer+5": "QCD: +5% jet smearing"}
DETECTOR_VARS = ("jes-3", "jes-fwd", "jer+5")


def robustness_plot(path, out, wp="0.01", name="robustness", cols=None):
    """Small multiples on one shared scale: selection efficiency in a varied
    sample relative to nominal, at a working point fixed on nominal QCD."""
    r = json.load(open(path))["samples"]
    cols = [k for k in (cols or r) if "nominal" not in k and k in r]
    titles = ROBUST_TITLES
    methods = [m for m in r[cols[0]]["methods"] if not m.startswith("argmin")]
    methods += [m for m in r[cols[0]]["methods"] if m.startswith("argmin")]
    y = np.arange(len(methods))[::-1]
    data = {}
    for c in cols:
        nom = r["signal, nominal" if c.startswith("signal") else "QCD, nominal"]
        rat, err = [], []
        for m in methods:
            v, n = r[c]["methods"][m][wp], nom["methods"][m][wp]
            rr = v["eff"] / n["eff"] if n["eff"] > 0 else np.nan
            rat.append(rr)
            err.append(rr * np.hypot(v["err"] / max(v["eff"], 1e-12), n["err"] / max(n["eff"], 1e-12)))
        data[c] = (np.array(rat), np.array(err), r[c]["acceptance_6j"] / nom["acceptance_6j"])
    lo = min(0.9, min(np.nanmin(a - e) for a, e, _ in data.values()))
    hi = max(1.1, max(np.nanmax(a + e) for a, e, _ in data.values()))
    fig, axes = plt.subplots(1, len(cols), figsize=(1.6 * len(cols) + 1.9, 0.27 * len(methods) + 1.0),
                             sharey=True, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, cols):
        rat, err, acc = data[c]
        ax.axvline(1.0, color="#d9d9d9", lw=0.8, zorder=0)
        for yi, ri, ei, m in zip(y, rat, err, methods):
            ci = MUTED if m.startswith("argmin") else ACCENT
            ax.plot([ri - ei, ri + ei], [yi, yi], color=ci, lw=0.8)
            ax.plot(ri, yi, "o", color=ci, ms=3.2)
        ax.set_title(f"{titles.get(c, c)}\n≥ 6-jet acceptance ×{acc:.2f}", fontsize=8, loc="left",
                     color=INK, linespacing=1.4)
        ax.spines["bottom"].set_bounds(lo, hi)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=7.5)
    axes[0].set_xlim(lo - 0.03, hi + 0.03)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(methods, fontsize=8)
    axes[0].set_ylim(-0.6, len(methods) - 0.4)
    fig.supxlabel(f"selection efficiency in the varied sample / nominal, at the {float(wp):.1%} "
                  "nominal-QCD working point (±1σ MC statistics)", fontsize=8.5)
    fig.tight_layout(w_pad=0.8)
    save(fig, out, name)


def truth_definition_plot(h5path, mass, out, name="truth_definition"):
    """Mass of the three jets matched to each gluino's decay quarks, with the
    quarks taken before or after their own FSR (both stored by generate_pythia)."""
    import h5py
    sys.path.insert(0, os.path.join(HERE, ".."))
    from src.pairdata import truth_sets_from_quarks
    with h5py.File(h5path, "r") as f:
        src = f["INPUTS/Source"]
        pt, eta, phi, m, mask = (src[k][:] for k in ("pt", "eta", "phi", "mass", "MASK"))
        mask = mask.astype(bool)
        q = {lab: f[key][:] for lab, key in (("before its own FSR", "truth/quarks_before_fsr"),
                                              ("after its own FSR (used so far)", "truth/quarks_after_fsr"))}
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    E = np.sqrt(px**2 + py**2 + pz**2 + m**2)
    fig, ax = plt.subplots(figsize=(4.4, 2.9))
    bins = np.linspace(0.4, 1.2, 65)
    ends = []
    for (lab, qq), col in zip(q.items(), (ACCENT, MUTED)):
        tj = truth_sets_from_quarks(qq, eta, phi, mask)
        vals = []
        for i in range(len(tj)):
            a, b = tj[i, :3], tj[i, 3:]
            if (tj[i] < 0).any() or len(set(a)) < 3 or len(set(b)) < 3 or set(a) & set(b):
                continue
            for h in (a, b):
                s = [E[i, h].sum(), px[i, h].sum(), py[i, h].sum(), pz[i, h].sum()]
                vals.append(np.sqrt(max(s[0]**2 - s[1]**2 - s[2]**2 - s[3]**2, 0)) / mass)
        v = np.array(vals)
        hist, _ = np.histogram(v, bins)
        hist = hist / hist.sum() / np.diff(bins)
        ax.step(bins[:-1], hist, where="post", color=col, lw=1.1)
        lo, hi = np.quantile(v, [0.16, 0.84])
        ends.append((hist.max(), f"{lab}\n16–84% within {lo:.2f}–{hi:.2f} M", col))
    top = max(e[0] for e in ends)
    for k, (_, t, c) in enumerate(sorted(ends, reverse=True)):
        ax.text(0.42, top * (1.0 - 0.24 * k), t, color=c, fontsize=7.5, va="top", linespacing=1.3)
    ax.set_xlabel(f"mass of the three truth-matched jets / M   (M = {mass:.0f} GeV)")
    ax.set_ylabel("density")
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_bounds(0.4, 1.2)
    save(fig, out, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robustness", default=None, help="robustness.json")
    ap.add_argument("--truth-def", default=None, help="h5 file with both truth-quark definitions")
    ap.add_argument("--truth-def-mass", type=float, default=1000.0)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--matching", default="results/matching.json")
    ap.add_argument("--out", default="plots/report")
    ap.add_argument("--massvars", default=None)
    ap.add_argument("--fom", default=None, help="search_study fom.json")
    a = ap.parse_args()
    runs = {}
    for r in a.runs:
        k, p = r.split("=")
        runs[k] = load(p)
    if os.path.exists(a.matching):
        matching_breakdown(a.matching, a.out)
    if a.massvars:
        massvar_fom(a.massvars, a.out)
    if a.fom:
        strategy_fom(a.fom, a.out)
    if a.robustness:
        keys = list(json.load(open(a.robustness))["samples"])
        det = [k for k in keys if k.split(", ")[1] in DETECTOR_VARS]
        robustness_plot(a.robustness, a.out, name="robustness_physics", cols=[k for k in keys if k not in det])
        if det:
            robustness_plot(a.robustness, a.out, name="robustness_detector",
                            cols=sorted(det, key=lambda k: (not k.startswith("signal"),
                                                            DETECTOR_VARS.index(k.split(", ")[1]))))
    if a.truth_def:
        truth_definition_plot(a.truth_def, a.truth_def_mass, a.out)
    if runs:
        eff_vs_mass(runs, a.out)
        preselection_curve(runs, a.out)
        mavg_hist(runs, a.out)


if __name__ == "__main__":
    main()
