#!/usr/bin/env python
"""Harvest the F5 audit trainings: identical evaluation of every finished run, seed ensembles,
comparison table (TABLE.md, harvest.json) and plots (plots/*.pdf, tufte style, one panel per PDF).

  python harvest.py                      # every finished run under runs/ + the r14A reference
  python harvest.py --only r14A          # reference only (self-test)

Working point (identical for every run), on the held-out test list:
  a QCD efficiency t is the weighted fraction of ALL test QCD events with A < 0.1 and gate score
  >= thr(t); the signal efficiency at mass M is the fraction of all test signal events of that
  mass with A < 0.1, m_avg in [0.85, 1.1] M and gate score >= thr(t).
  (The training-time metric psig_at_bkg normalises t to the QCD passing A < 0.1 instead; that
  definition moves with the pairing, this one does not.)
Errors: Poisson bootstrap over events, the same replica weights for every run, so the error of the
difference to the reference (d_ref) accounts for the shared test set.  n_eff is the effective
number of surviving weighted QCD events, (sum w)^2 / sum w^2: below ~30 the point is one or two
heavy events and the quoted error is not reliable.
Tail (never trained on; QCD pre-selected by the r14A filter, so only the AND with that filter is
unbiased): weighted fraction of the tail events that the run also keeps (A < 0.1, score >= thr(t)).
For r14A itself this is the fraction it keeps of its own tail; a run below that removes events
r14A let through, but events r14A rejected and the run accepts are invisible here.
"""
import argparse
import glob
import json
import os
import sys
from itertools import combinations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R14A = "/scratch/llee/pairnet/sweep/out_r14A_small_12j_win5_lowmass_boost/r14A_small_12j_win5_lowmass_boost"
MASSES = (400, 700, 1000, 1500, 2000)
TS = (0.05, 0.02, 0.005, 0.001, 0.0002)
ASYM_CUT, WLO, WHI = 0.1, 0.85, 1.1
VARIANTS = ("mup", "mdn", "msm", "m0")


def logit(p):
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def load_run(d):
    """score (gate logit), mavg, asym of a run directory; None if it has not finished."""
    f = os.path.join(d, "eval.npz")
    if not os.path.exists(f):
        return None
    e = np.load(f)
    r = {"score": e["gate_logit"].astype(np.float64) if "gate_logit" in e.files else logit(e["psig"]),
         "mavg": e["mavg"], "asym": e["asym"], "best": e["best"], "dir": d}
    return r


def interp_masses(test, best, n_assign=7):
    """m_avg and asymmetry of interpretation `best` (index into the 70 (3,3) splits of the leading
    7 jets), recomputed with numpy from the test arrays (for ensemble-chosen pairings)."""
    interps = [(a, b) for a in combinations(range(n_assign), 3)
               for b in combinations([j for j in range(n_assign) if j not in a], 3) if a < b]
    ia = np.array([a for a, _ in interps])[best]  # (N, 3)
    ib = np.array([b for _, b in interps])[best]
    pt, eta, phi, m = (test[k][:, :n_assign].astype(np.float64) for k in ("pt", "eta", "phi", "m"))
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    en = np.sqrt(px**2 + py**2 + pz**2 + m**2)
    rows = np.arange(len(best))[:, None]

    def gm(ix):
        s = [v[rows, ix].sum(1) for v in (en, px, py, pz)]
        return np.sqrt(np.maximum(s[0]**2 - s[1]**2 - s[2]**2 - s[3]**2, 0.0))
    m1, m2 = gm(ia), gm(ib)
    return 0.5 * (m1 + m2), np.abs(m1 - m2) / np.maximum(m1 + m2, 1e-9)


class Evaluator:
    def __init__(self, test, nboot=60, seed=7):
        self.sig = test["is_signal"].astype(bool)
        self.w = test["w"].astype(np.float64)
        self.pm = test["pmass"]
        rng = np.random.default_rng(seed)
        self.iq = np.flatnonzero(~self.sig)
        self.wq = self.w[self.iq]
        self.bq = rng.poisson(1.0, (nboot, len(self.iq))).astype(np.int8)  # shared replicas
        self.im = {M: np.flatnonzero(self.sig & (np.abs(self.pm - M) < 1)) for M in MASSES}
        self.bs = {M: rng.poisson(1.0, (nboot, len(ix))).astype(np.int8) for M, ix in self.im.items()}
        self.nboot = nboot

    def thresholds(self, score, asym, wrep=None):
        """thr(t) and n_eff: weighted fraction t of all QCD has A < 0.1 and score >= thr."""
        w = self.wq if wrep is None else self.wq * wrep
        pas = asym[self.iq] < ASYM_CUT
        s, ww = score[self.iq][pas], w[pas]
        o = np.argsort(-s)
        s, ww = s[o], ww[o]
        c = np.cumsum(ww) / w.sum()
        c2 = np.cumsum(ww**2)
        thr, neff = [], []
        for t in TS:
            k = int(np.searchsorted(c, t, side="right")) - 1  # last event with cumulative <= t
            if k < 0:
                thr.append(np.inf), neff.append(0.0)
            else:
                thr.append(s[k]), neff.append((c[k] * w.sum())**2 / c2[k])
        return np.array(thr), np.array(neff)

    def run(self, r):
        """point estimates and bootstrap replicas of eff[mass, t]."""
        score, mavg, asym = r["score"], r["mavg"], r["asym"]
        thr, neff = self.thresholds(score, asym)
        eff = np.zeros((len(MASSES), len(TS)))
        rep = np.zeros((self.nboot, len(MASSES), len(TS)))
        thr_b = np.array([self.thresholds(score, asym, self.bq[b])[0] for b in range(self.nboot)])
        for i, M in enumerate(MASSES):
            ix = self.im[M]
            ok = (asym[ix] < ASYM_CUT) & (mavg[ix] > WLO * M) & (mavg[ix] < WHI * M)
            sc = np.where(ok, score[ix], -np.inf)
            eff[i] = [(sc >= t).mean() for t in thr]
            bw = self.bs[M].astype(np.float64)
            for j in range(len(TS)):
                rep[:, i, j] = (bw * (sc[None, :] >= thr_b[:, j:j + 1])).sum(1) / bw.sum(1)
        pasq = asym[self.iq] < ASYM_CUT
        return {"thr": thr, "neff": neff, "eff": eff, "rep": rep,
                "bkg_a01": float(self.wq[pasq].sum() / self.wq.sum()),
                "sig_win_a01": [float(((asym[self.im[M]] < ASYM_CUT) & (mavg[self.im[M]] > WLO * M)
                                       & (mavg[self.im[M]] < WHI * M)).mean()) for M in MASSES]}

    def at_fixed_thr(self, thr, score, mavg, asym):
        """signal efficiency per mass and actual QCD efficiency at given thresholds (mass variants)."""
        eff = np.zeros((len(MASSES), len(TS)))
        for i, M in enumerate(MASSES):
            ix = self.im[M]
            ok = (asym[ix] < ASYM_CUT) & (mavg[ix] > WLO * M) & (mavg[ix] < WHI * M)
            eff[i] = [(ok & (score[ix] >= t)).mean() for t in thr]
        pq = asym[self.iq] < ASYM_CUT
        q = np.array([self.wq[pq & (score[self.iq] >= t)].sum() / self.wq.sum() for t in thr])
        return eff, q


def tail_kept(thr, score, asym, w, src):
    out = {}
    for k, name in ((0, "vlmT"), (1, "htT")):
        m = src == k
        out[name] = [float(w[m & (asym < ASYM_CUT) & (score >= t)].sum() / w[m].sum()) for t in thr]
    return out


def plot_history(name, d, pdir):
    f = os.path.join(d, "metrics.json")
    if not os.path.exists(f):
        return None
    h = json.load(open(f))["history"]
    if not h or "val_loss" not in h[-1]:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sty = glob.glob("/home/llee/.claude/skills/synced/*/tufte-plots/assets/tufte.mplstyle")
    if sty:
        plt.style.use(sty[0])
    ep = [e["epoch"] + 1 for e in h]
    vl, tl = [e["val_loss"] for e in h], [e["train_eval_loss"] for e in h]
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(ep, vl, "-o", ms=3, color="k"), ax.plot(ep, tl, "-o", ms=3, color="0.6")
    ax.annotate("validation", (ep[-1], vl[-1]), xytext=(4, 2), textcoords="offset points", fontsize=8)
    ax.annotate("train subset", (ep[-1], tl[-1]), xytext=(4, -8), textcoords="offset points", fontsize=8, color="0.4")
    ax.set_xlabel("epoch"), ax.set_ylabel("nominal loss"), ax.set_title(f"{name}: overtraining check", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(os.path.join(pdir, f"overtrain_{name}.pdf"), bbox_inches="tight"), plt.close(fig)
    key = "sig_at_psig_bkg" if "sig_at_psig_bkg" in h[-1] else "sig_in_window_at_bkg"
    tk = "0.005" if key == "sig_at_psig_bkg" else "0.02"
    fig, ax = plt.subplots(figsize=(4.5, 3))
    v, t = [e[key][tk] for e in h], [e["train_sub"][key][tk] for e in h]
    ax.plot(ep, v, "-o", ms=3, color="k"), ax.plot(ep, t, "-o", ms=3, color="0.6")
    ax.annotate("validation", (ep[-1], v[-1]), xytext=(4, 2), textcoords="offset points", fontsize=8)
    ax.annotate("train subset", (ep[-1], t[-1]), xytext=(4, -8), textcoords="offset points", fontsize=8, color="0.4")
    what = ("gate cut, %s of A<0.1 QCD" if key == "sig_at_psig_bkg" else "asymmetry scan, %s of QCD") % f"{100 * float(tk):g}%"
    ax.set_xlabel("epoch"), ax.set_ylabel("signal efficiency in window"), ax.set_title(f"{name}: {what}", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(os.path.join(pdir, f"eff_vs_epoch_{name}.pdf"), bbox_inches="tight"), plt.close(fig)
    return {"n_ep": len(h), "val_loss": vl[-1], "train_eval_loss": tl[-1], "gap": vl[-1] - tl[-1],
            "val_loss_min_ep": int(np.argmin(vl)) + 1, "epoch_time": float(np.median([e["time"] for e in h]))}


def plot_compare(res, pdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sty = glob.glob("/home/llee/.claude/skills/synced/*/tufte-plots/assets/tufte.mplstyle")
    if sty:
        plt.style.use(sty[0])
    names = list(res)
    for i, M in enumerate(MASSES):
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for n in names:
            r = res[n]
            kw = dict(color="k", lw=1.4, zorder=5) if n == "r14A" else dict(lw=0.8, alpha=0.8)
            ax.errorbar(100 * np.array(TS), r["eff"][i], yerr=r["rep"][:, i].std(0), marker="o", ms=2.5, **kw)
            ax.annotate(n, (100 * TS[0], r["eff"][i][0]), xytext=(4, 0), textcoords="offset points", fontsize=6)
        ax.set_xscale("log"), ax.set_xlabel("QCD efficiency [%]"), ax.set_ylabel("signal efficiency in window")
        ax.set_title(f"M = {M} GeV, test set", fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.savefig(os.path.join(pdir, f"sig_vs_qcd_M{M}.pdf"), bbox_inches="tight"), plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default=os.path.join(HERE, "runs"))
    ap.add_argument("--test-npz", default=os.path.join(HERE, "data/cache/test_2da66480a069.npz"))
    ap.add_argument("--tail-ref", default=os.path.join(HERE, "data/tail_eval_ref.npz"))
    ap.add_argument("--tail-npz", default=os.path.join(HERE, "data/f5/tail_eval.npz"))
    ap.add_argument("--ref", default=R14A, help="reference run directory (r14A); '' for none")
    ap.add_argument("--run", action="append", default=[], help="extra name=dir")
    ap.add_argument("--ens", action="append", default=[], help="name=member1,member2,... (default: r14A+f5_s1..3)")
    ap.add_argument("--only", nargs="*", help="restrict to these run names")
    ap.add_argument("--nboot", type=int, default=60)
    ap.add_argument("--out", default=HERE)
    a = ap.parse_args()
    pdir = os.path.join(a.out, "plots")
    os.makedirs(pdir, exist_ok=True)

    test = dict(np.load(a.test_npz))
    ev = Evaluator(test, a.nboot)
    dirs = {}
    if a.ref:
        dirs["r14A"] = a.ref
    for d in sorted(glob.glob(os.path.join(a.runs_dir, "out_*", "*", ""))):
        dirs[os.path.basename(d.rstrip("/"))] = d.rstrip("/")
    for s in a.run:
        n, d = s.split("=", 1)
        dirs[n] = d
    runs = {}
    for n, d in dirs.items():
        if a.only and n not in a.only:
            continue
        r = load_run(d)
        if r is None:
            print(f"[skip] {n}: no eval.npz yet")
            continue
        if len(r["score"]) != len(ev.sig):
            print(f"[skip] {n}: {len(r['score'])} events, test list has {len(ev.sig)}")
            continue
        runs[n] = r

    # seed ensembles: average gate logit; pairing from the first member, or from the averaged logp
    ens = dict(s.split("=", 1) for s in a.ens) or {"ens4": "r14A,f5_s1,f5_s2,f5_s3", "ens3": "f5_s1,f5_s2,f5_s3"}
    for en, mem in ens.items():
        mem = [m for m in mem.split(",") if m in runs]
        if len(mem) < 2:
            continue
        sc = np.mean([runs[m]["score"] for m in mem], 0)
        runs[f"{en}_gate[{len(mem)}]"] = {"score": sc, "mavg": runs[mem[0]]["mavg"], "asym": runs[mem[0]]["asym"],
                                          "members": mem, "dir": None}
        lp = [m for m in mem if os.path.exists(os.path.join(runs[m]["dir"], "eval_logits.npz"))]
        if len(lp) >= 2:
            acc = None
            for m in lp:
                x = np.load(os.path.join(runs[m]["dir"], "eval_logits.npz"))["logp"].astype(np.float32)
                acc = x if acc is None else acc + x
            best = acc.argmax(1)
            mavg, asym = interp_masses(test, best)
            m0 = runs[lp[0]]
            ma0, as0 = interp_masses(test, m0["best"])  # closure: numpy masses vs the member's own
            print(f"[{en}] pairing closure on {lp[0]}: max |dm_avg|/m_avg {np.nanmax(np.abs(ma0 - m0['mavg']) / np.maximum(m0['mavg'], 1)):.2e}, "
                  f"max |dA| {np.nanmax(np.abs(as0 - m0['asym'])):.2e}; ensemble pairing differs from {lp[0]} in {(best != m0['best']).mean():.3f} of events")
            runs[f"{en}_gate+pair[{len(lp)}]"] = {"score": np.mean([runs[m]["score"] for m in lp], 0), "mavg": mavg,
                                                  "asym": asym, "members": lp, "dir": None}

    res = {}
    for n, r in runs.items():
        res[n] = ev.run(r)
        print(f"[done] {n}")
    ref = res.get("r14A")

    # tail and mass variants
    tr = np.load(a.tail_ref) if os.path.exists(a.tail_ref) else None
    tw = np.load(a.tail_npz)["w"].astype(np.float64) if tr is not None and os.path.exists(a.tail_npz) else None
    for n, r in runs.items():
        x = res[n]
        if tw is not None:
            if n == "r14A":
                m1, m2 = tr["r14A_m1"].astype(np.float64), tr["r14A_m2"].astype(np.float64)
                x["tail"] = tail_kept(x["thr"], logit(tr["r14A_psig"]), np.abs(m1 - m2) / (m1 + m2), tw, tr["src"])
            elif r.get("members"):
                es = [np.load(os.path.join(runs[m]["dir"], "eval_tail.npz")) for m in r["members"]
                      if runs[m]["dir"] and os.path.exists(os.path.join(runs[m]["dir"], "eval_tail.npz"))]
                if len(es) == len(r["members"]) and "gate+pair" not in n:
                    x["tail"] = tail_kept(x["thr"], np.mean([e["gate_logit"] for e in es], 0), es[0]["asym"], tw, tr["src"])
            elif os.path.exists(os.path.join(r["dir"], "eval_tail.npz")):
                e = np.load(os.path.join(r["dir"], "eval_tail.npz"))
                x["tail"] = tail_kept(x["thr"], e["gate_logit"].astype(np.float64), e["asym"], tw, tr["src"])
        if r["dir"]:
            x["var"] = {}
            for v in VARIANTS:
                f = os.path.join(r["dir"], f"eval_test_{v}.npz")
                if os.path.exists(f):
                    e = np.load(f)
                    eff, q = ev.at_fixed_thr(x["thr"], e["gate_logit"].astype(np.float64), e["mavg"], e["asym"])
                    x["var"][v] = {"eff": eff.tolist(), "qcd": q.tolist()}
            x["hist"] = plot_history(n, r["dir"], pdir)

    # table
    L = ["# F5 comparison (test list; signal efficiency in window, A<0.1, [0.85,1.1]M, x100; +- Poisson bootstrap)", "",
         "QCD efficiency = weighted fraction of all test QCD with A<0.1 and gate >= thr.  d_ref = run - r14A with the",
         "paired bootstrap error (shared test set).  n_eff = effective surviving QCD events.", ""]
    for j, t in enumerate(TS):
        L += [f"## QCD efficiency {100 * t:g}%", "",
              "| run | n_eff | " + " | ".join(f"M{M}" for M in MASSES) + " | mean | d_ref(mean) |",
              "|---|---|" + "---|" * (len(MASSES) + 2)]
        for n, x in res.items():
            e, s = 100 * x["eff"][:, j], 100 * x["rep"][:, :, j].std(0)
            mean, ms = e.mean(), 100 * x["rep"][:, :, j].mean(1).std()
            dref = ""
            if ref is not None and n != "r14A":
                d = 100 * (x["rep"][:, :, j] - ref["rep"][:, :, j]).mean(1)
                dref = f"{100 * (x['eff'][:, j] - ref['eff'][:, j]).mean():+.2f} +- {d.std():.2f}"
            L.append(f"| {n} | {x['neff'][j]:.0f} | " + " | ".join(f"{a_:.2f} +- {b_:.2f}" for a_, b_ in zip(e, s))
                     + f" | {mean:.2f} +- {ms:.2f} | {dref} |")
        L.append("")
    L += ["## Pairing only (no gate cut)", "", "| run | QCD A<0.1 | " + " | ".join(f"sig win M{M}" for M in MASSES) + " |",
          "|---|---|" + "---|" * len(MASSES)]
    for n, x in res.items():
        L.append(f"| {n} | {x['bkg_a01']:.4f} | " + " | ".join(f"{v:.4f}" for v in x["sig_win_a01"]) + " |")
    L += ["", "## Never-trained tail (r14A-selected QCD): weighted fraction also kept by the run, x100", "",
          "| run | sample | " + " | ".join(f"t={100 * t:g}%" for t in TS) + " |", "|---|---|" + "---|" * len(TS)]
    for n, x in res.items():
        for k, v in x.get("tail", {}).items():
            L.append(f"| {n} | {k} | " + " | ".join(f"{100 * u:.2f}" for u in v) + " |")
    L += ["", "## Jet-mass variants at the nominal thresholds: mean signal efficiency x100 / actual QCD efficiency in %", "",
          "| run | variant | " + " | ".join(f"t={100 * t:g}%" for t in TS[:4]) + " |", "|---|---|" + "---|" * 4]
    for n, x in res.items():
        for v, d in x.get("var", {}).items():
            e, q = np.array(d["eff"]).mean(0), d["qcd"]
            L.append(f"| {n} | {v} | " + " | ".join(f"{100 * e[j]:.2f} / {100 * q[j]:.3f}" for j in range(4)) + " |")
    L += ["", "## Training curves (nominal loss; gap = validation - train subset)", "",
          "| run | epochs | val loss | train-subset loss | gap | epoch of min val loss | s/epoch |", "|---|---|---|---|---|---|---|"]
    for n, x in res.items():
        h = x.get("hist")
        if h:
            L.append(f"| {n} | {h['n_ep']} | {h['val_loss']:.4f} | {h['train_eval_loss']:.4f} | {h['gap']:+.4f} | {h['val_loss_min_ep']} | {h['epoch_time']:.0f} |")
    open(os.path.join(a.out, "TABLE.md"), "w").write("\n".join(L) + "\n")
    plot_compare(res, pdir)
    js = {n: {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in x.items() if k != "rep"}
          | {"err": x["rep"].std(0).tolist()} for n, x in res.items()}
    json.dump({"masses": MASSES, "ts": TS, "runs": js}, open(os.path.join(a.out, "harvest.json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
