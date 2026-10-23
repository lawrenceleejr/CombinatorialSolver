"""Anatomy of the selected signal m_avg distribution: which events make the narrow peak and which the broad pedestal.

Truth categories per event (held-out signal files c16-19, inclusive sample), using the generator jet-quark matching:
  correct   both truth triplets fully matched to reconstructed jets and PairNet's chosen triplet masses equal theirs
  wrong     both truth triplets fully matched, PairNet chose a different split
  unmatched at least one of the six quarks has no matched jet (out of acceptance, merged, or lost to FSR)
Selection: A < 0.1, valid, offline H_T > trig, P(signal) >= the shape-fit-tuned threshold (and its H_T/M cut).
One single-panel file per mass (stacked m_avg by category) plus a summary json.
usage: peak_anatomy.py style out trig fisher_json masses
"""
import json, sys
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

style, out, trig, fj = sys.argv[1:5]; trig = float(trig); masses = [int(x) for x in sys.argv[5].split(",")]
sys.path.insert(0, style)
from tufte_helpers import spines_to_data_range, save_pdf  # noqa: E402
plt.style.use(f"{style}/tufte.mplstyle")
P = "/scratch/llee/pairnet/prod_lowmass"; E = "/ospool/uc-shared/project/futurecolliders/llee/pairnet/eval_r14"
FJ = json.load(open(fj))["results"]


def p4(pt, eta, phi, m):
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    return np.stack([np.sqrt(px ** 2 + py ** 2 + pz ** 2 + m ** 2), px, py, pz], -1)


def mass3(J, idx):
    v = np.take_along_axis(J, idx[:, :, None], 1).sum(1)
    return np.sqrt(np.maximum(v[:, 0] ** 2 - (v[:, 1:] ** 2).sum(1), 0))


summary = {}
for M in masses:
    x = FJ[str(M)]; conf = min([k for k in ("P", "P+boost", "P+all") if k in x], key=lambda k: x[k]["limit_fb"])
    cuts = x[conf]["halves"][0]["cuts"]
    cols = {"correct": [], "wrong": [], "unmatched": []}
    for c in (16, 17, 18, 19):
        h = h5py.File(f"{P}/siglm_m{M}_incl_c{c}.h5"); a = np.load(f"{E}/siglm_m{M}_incl_c{c}.npz")
        S = h["INPUTS/Source"]
        J = p4(S["pt"][:], S["eta"][:], S["phi"][:], S["mass"][:])
        t1 = np.stack([h[f"TARGETS/g1/j{k}"][:] for k in (1, 2, 3)], 1); t2 = np.stack([h[f"TARGETS/g2/j{k}"][:] for k in (1, 2, 3)], 1)
        full = (t1 >= 0).all(1) & (t2 >= 0).all(1)
        tm1, tm2 = mass3(J, np.maximum(t1, 0)), mass3(J, np.maximum(t2, 0))
        m1, m2 = a["r14A_m1"], a["r14A_m2"]
        same = (np.isclose(m1, tm1, rtol=1e-3) & np.isclose(m2, tm2, rtol=1e-3)) | (np.isclose(m1, tm2, rtol=1e-3) & np.isclose(m2, tm1, rtol=1e-3))
        mavg = 0.5 * (m1 + m2); A = np.abs(m1 - m2) / np.maximum(m1 + m2, 1e-9)
        sel = (A < 0.1) & a["r14A_valid"].astype(bool) & (a["ht"] > trig) & (a["r14A_psig"] >= cuts.get("psig", -1)) & (a["ht"] / M > cuts.get("ht", 0))
        for k, msk in (("correct", full & same), ("wrong", full & ~same), ("unmatched", ~full)):
            cols[k].append(mavg[sel & msk] / M)
    cols = {k: np.concatenate(v) for k, v in cols.items()}
    n = {k: len(v) for k, v in cols.items()}; win = {k: int(((v > 0.85) & (v < 1.1)).sum()) for k, v in cols.items()}
    summary[M] = {"selection": conf, "cuts": cuts, "n": n, "in_window": win,
                  "correct_fraction_all": n["correct"] / max(sum(n.values()), 1), "correct_fraction_window": win["correct"] / max(sum(win.values()), 1)}
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    b = np.linspace(0.4, 2.0, 65)
    ax.hist([cols["correct"], cols["wrong"], cols["unmatched"]], b, stacked=True, histtype="stepfilled",
            color=["#9b2c2c", "#d9a38f", "0.8"], label=[f"correct pairing ({n['correct']})", f"all quarks matched, wrong split ({n['wrong']})", f"a quark has no jet ({n['unmatched']})"])
    ax.axvspan(0.85, 1.1, color="0.2", alpha=0.06, lw=0)
    ax.set_xlabel(r"$m_{\rm avg}$ / $M$"); ax.set_ylabel("signal MC events / bin")
    ax.set_title(f"{M} GeV, $H_T>{trig:g}$ GeV, {conf} (shape-fit tuned): {100 * summary[M]['correct_fraction_window']:.0f}% of the window is correctly paired", fontsize=6.2, loc="left")
    ax.legend(fontsize=6.2, frameon=False, loc="upper right")
    spines_to_data_range(ax)
    save_pdf(fig, f"{out}/anatomy_m{M}.pdf"); fig.savefig(f"{out}/anatomy_m{M}.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    print(M, conf, n, "window", win, "correct frac all %.2f window %.2f" % (summary[M]["correct_fraction_all"], summary[M]["correct_fraction_window"]))
json.dump(summary, open(f"{out}/anatomy.json", "w"), indent=1)
