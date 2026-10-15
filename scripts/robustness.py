"""
Modelling robustness of the event selection.

Every method's threshold is fixed on the NOMINAL held-out QCD at a weighted
efficiency of 1%, 0.3% and 0.1% (as an analysis would fix its working point
in simulation). The same thresholds are then applied to samples generated with
different physics:

  sig_m1000_chain500 : g~ -> ~u_R u~, ~u_R -> d~ s~ (m~q = 500 GeV) instead of
                       the three-body g~ -> uds the networks were trained on
  *_as150 / *_as123  : shower alpha_s (ISR and FSR) 0.150 / 0.123 instead of
                       Monash 0.1365, i.e. roughly +-10%: more / less radiation

Reported per sample: the >= 6-jet acceptance, the selection efficiency within
the >= 6-jet region, the exact-pairing rate and the mass peak of the selected
signal. Nominal QCD is the held-out qcd_s8-s10, s15, s16 (>= 6 jets,
pTHat > 200 GeV, the settings of the varied QCD); with ~12k varied QCD events
only the 1% working point is statistically meaningful for QCD.

  python scripts/robustness.py --extra "7-jet + gate sees pairings + mass head=runs/g7_gi_mh/model.pt"
"""

import argparse
import glob
import json
import os
import pickle
import sys

import h5py
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from search_study import mavg_asym, pairnet_outputs, shortest  # noqa: E402
from src.features import classical_pairing, event_shape_features, pairing_features, stack  # noqa: E402
from src.pairdata import load_events  # noqa: E402

QCD_EFFS = (0.01, 0.003, 0.001)


def acceptance(path, is_signal):
    """>= 6-jet acceptance per generated event (weighted for QCD), from the file attributes."""
    with h5py.File(path, "r") as f:
        n_tried = float(f.attrs["n_tried"])
        nj = f["INPUTS/Source/MASK"][:].sum(1)
        w = f["EventVars/normweight"][:]
    sel = nj >= 6
    return (sel.sum() if is_signal else w[sel].sum()) / n_tried


def scores(d, a, nets):
    """{method: (score, g1, g2)}; g1/g2 are the chosen jet triplets (None: no pairing)."""
    out = {}
    g1, g2 = classical_pairing(d)
    _, as_cl = mavg_asym(d, g1, g2)
    out["argmin |Δm|, A < 0.1"] = ((as_cl < 0.1).astype(float), g1, g2)
    if os.path.exists(a.bdt):
        bd = pickle.load(open(a.bdt, "rb"))
        X, keys = stack([event_shape_features(d), pairing_features(d, g1, g2)])
        keep = [i for i, k in enumerate(keys) if k not in bd["drop"]]
        out["expert BDT"] = (bd["model"].predict_proba(X[:, keep])[:, 1], g1, g2)
    for name, path in nets.items():
        (sg1, sg2), _, gate, _ = pairnet_outputs(path, d)
        out[name] = (gate, sg1, sg2)
    if os.path.exists(a.stack):
        from train_stack import stack_features
        from search_study import G1, G2
        from src.pairnet import load_checkpoint
        st = pickle.load(open(a.stack, "rb"))
        net_st, _ = load_checkpoint(st["pairnet"])
        Xst, _, best = stack_features(net_st, d)
        out["stacked (expert + PairNet)"] = (st["model"].predict_proba(Xst)[:, 1], G1[best], G2[best])
    return out


def truth_ok(d, g1, g2):
    """Chosen 3+3 pairing equals the truth (all six quarks matched to distinct jets)."""
    tj = d["tjets"]
    ok = (tj >= 0).all(1)
    t1, t2 = np.sort(tj[:, :3], 1), np.sort(tj[:, 3:], 1)
    c1, c2 = np.sort(g1, 1), np.sort(g2, 1)
    same = ((c1 == t1).all(1) & (c2 == t2).all(1)) | ((c1 == t2).all(1) & (c2 == t1).all(1))
    return ok & same


def load(paths, is_signal):
    d = load_events(paths, 20, 3, is_signal, None, min_jets=6)
    # truth jet index per quark in the pT-sorted list (nearest jet within 0.4; -1 if none)
    if is_signal:
        from src.pairdata import truth_sets_from_quarks
        tq = []
        files = []
        for x in paths:  # same order as load_events
            files += sorted(glob.glob(x)) or [x]
        for p in files:
            with h5py.File(p, "r") as f:
                src = f["INPUTS/Source"]
                pt, eta, phi, mask = src["pt"][:], src["eta"][:], src["phi"][:], src["MASK"][:].astype(bool)
                order = np.argsort(np.where(mask, -pt, np.inf), 1, kind="stable")
                tk = lambda x: np.take_along_axis(x, order, 1)
                t = truth_sets_from_quarks(f["truth/quarks"][:], tk(eta), tk(phi), tk(mask))
                tq.append(t[mask.sum(1) >= 6])
        tj = np.concatenate(tq)
        # quarks sharing a jet with another quark cannot be paired correctly
        dup = np.array([len(set(r[r >= 0].tolist())) != (r >= 0).sum() for r in tj])
        tj[dup] = -1
        d["tjets"] = tj
    return d


def jet_variation(d, kind, seed=0):
    """Detector-level variations of loaded (pT-sorted) events, no regeneration:
      jes-3    all jets -3% (pT and mass); P(signal) is invariant by construction
               (the network divides by HT) except for jets lost below 30 GeV
      jes-fwd  -5% for |eta| > 1.5 only, which breaks that invariance
      jer+5    an extra 5% Gaussian pT smearing per jet
    Re-applies pT > 30 GeV, re-sorts, keeps >= 6-jet events; returns
    (events, weighted fraction of events kept)."""
    rng = np.random.default_rng(seed)
    pt, m, eta, phi, mask = (d[k].copy() for k in ("pt", "m", "eta", "phi", "mask"))
    f = {"jes-3": lambda: np.full(pt.shape, 0.97),
         "jes-fwd": lambda: np.where(np.abs(eta) > 1.5, 0.95, 1.0),
         "jer+5": lambda: np.clip(1 + 0.05 * rng.standard_normal(pt.shape), 0.5, 1.5)}[kind]()
    pt, m = pt * f, m * f
    mask &= pt > 30.0
    order = np.argsort(np.where(mask, -pt, np.inf), 1, kind="stable")
    tk = lambda x: np.take_along_axis(x, order, 1)
    out = dict(d)
    ms = tk(mask)
    for k, v in (("pt", pt), ("m", m), ("eta", eta), ("phi", phi)):
        out[k] = np.where(ms, tk(v), 0.0).astype(d[k].dtype)
    out["mask"] = ms
    out["njall"] = mask.sum(1)
    if "tjets" in d:
        inv = np.argsort(order, 1)  # old position -> new position
        tj = np.maximum(d["tjets"], 0)
        still = (d["tjets"] >= 0) & np.take_along_axis(mask, tj, 1)
        out["tjets"] = np.where(still, np.take_along_axis(inv, tj, 1), -1)
    keep = out["njall"] >= 6
    n = len(keep)
    return ({k: (v[keep] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in out.items()},
            float((d["w"] * keep).sum() / d["w"].sum()))


def weff(sel, w):
    s, tot = (w * sel).sum(), w.sum()
    return s / tot, np.sqrt((w**2 * sel).sum()) / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nom-sig", default="data/pythia/sig_m1000_s2.h5")
    ap.add_argument("--nom-qcd", nargs="+", default=[f"data/pythia/qcd_s{i}.h5" for i in (8, 9, 10, 15, 16)])
    ap.add_argument("--thr-qcd", nargs="+", default=None,
                    help="QCD that fixes the working points (default: --nom-qcd)")
    ap.add_argument("--var-sig", nargs="+", default=["data/robust/sig_m1000_chain500.h5",
                                                     "data/robust/sig_m1000_as150.h5",
                                                     "data/robust/sig_m1000_as123.h5"])
    ap.add_argument("--var-qcd", nargs="+", default=["as150=data/robust/qcd_as150_*.h5",
                                                     "as123=data/robust/qcd_as123_*.h5"])
    ap.add_argument("--jet-vars", nargs="*", default=["jes-3", "jes-fwd", "jer+5"],
                    help="detector-level variations applied to the nominal samples")
    ap.add_argument("--gated", default="runs/gated_n7/model.pt")
    ap.add_argument("--v2", default="runs/v2_full/model.pt")
    ap.add_argument("--n6", default="runs/gated_n6/model.pt")
    ap.add_argument("--extra", nargs="*", default=[], help="name=path to more PairNet checkpoints")
    ap.add_argument("--bdt", default="runs/bdt/bdt.pkl")
    ap.add_argument("--stack", default="runs/stack/stack.pkl")
    ap.add_argument("--out", default="results/robustness")
    a = ap.parse_args()
    torch.set_num_threads(4)

    nets = {"gated PairNet, 7 jets": a.gated, "v2 (12-jet context)": a.v2, "gated PairNet, 6 jets": a.n6}
    nets.update(dict(x.split("=", 1) for x in a.extra))
    nets = {k: v for k, v in nets.items() if os.path.exists(v)}

    # working points on nominal QCD
    dq = load(a.thr_qcd or a.nom_qcd, False)
    sq = scores(dq, a, nets)
    thr = {}
    for m, (s, _, _) in sq.items():
        if m.startswith("argmin"):
            thr[m] = {e: 0.5 for e in QCD_EFFS}  # fixed preselection, no working point
            continue
        o = np.argsort(-s)
        cw = np.cumsum(dq["w"][o]) / dq["w"].sum()
        thr[m] = {e: float(s[o][min(np.searchsorted(cw, e), len(cw) - 1)]) for e in QCD_EFFS}

    samples = {"signal, nominal": (a.nom_sig, True, None), "QCD, nominal": (a.nom_qcd, False, None)}
    for p in a.var_sig:
        if os.path.exists(p):
            samples["signal, " + os.path.basename(p)[len("sig_m1000_"):-3]] = (p, True, None)
    for spec in a.var_qcd:
        name, pat = spec.split("=", 1)
        fs = sorted(glob.glob(pat))
        if fs:
            samples["QCD, " + name] = (fs, False, None)
    for kind in a.jet_vars:
        samples["signal, " + kind] = (a.nom_sig, True, kind)
        samples["QCD, " + kind] = (a.nom_qcd, False, kind)

    res = {"thresholds": thr, "samples": {}}
    cache = {}
    for sname, (paths, is_sig, kind) in samples.items():
        plist = paths if isinstance(paths, list) else [paths]
        key = (tuple(plist), is_sig)
        if key not in cache:
            cache[key] = (load(plist, is_sig), np.mean([acceptance(p, is_sig) for p in plist]))
        d, acc = cache[key]
        if kind:
            d, kept = jet_variation(d, kind)
            acc = acc * kept
        nj = d["njall"]
        r = {"n_events": int(len(nj)), "acceptance_6j": float(acc),
             "mean_njets": float(np.average(nj, weights=d["w"])),
             "frac_7plus": float(np.average(nj >= 7, weights=d["w"])), "methods": {}}
        if is_sig:
            r["frac_all6_matched_distinct"] = float((d["tjets"] >= 0).all(1).mean())
        for m, (s, g1, g2) in scores(d, a, nets).items():
            mm = {}
            mavg = mavg_asym(d, g1, g2)[0]
            for e in QCD_EFFS:
                sel = s >= thr[m][e] if not m.startswith("argmin") else s > 0.5
                eff, err = weff(sel, d["w"])
                x = {"eff": float(eff), "err": float(err)}
                if is_sig:
                    x["pairing_correct"] = float(truth_ok(d, g1, g2)[sel].mean()) if sel.any() else 0.0
                    if sel.sum() > 50:
                        lo, hi = shortest(mavg[sel] / 1000.0, 0.68)
                        x["mavg_median_over_M"] = float(np.median(mavg[sel]) / 1000.0)
                        x["mavg_halfwidth68"] = float((hi - lo) / 2)
                mm[str(e)] = x
            r["methods"][m] = mm
        res["samples"][sname] = r
        print(f"{sname:28s} N {len(nj):6d}  acc(>=6j) {acc:.4g}  <nj> {r['mean_njets']:.2f}", flush=True)

    # ratios to nominal
    print("\nselection efficiency relative to nominal at the 1% / 0.3% / 0.1% QCD working points")
    for sname, r in res["samples"].items():
        if "nominal" in sname:
            continue
        nom = res["samples"]["signal, nominal" if sname.startswith("signal") else "QCD, nominal"]
        print(f"  {sname}   (>=6-jet acceptance x{r['acceptance_6j'] / nom['acceptance_6j']:.3f})")
        for m in r["methods"]:
            cells = []
            for e in QCD_EFFS:
                v, n = r["methods"][m][str(e)], nom["methods"][m][str(e)]
                if n["eff"] <= 0:
                    cells.append("   n/a      ")
                    continue
                rat = v["eff"] / n["eff"]
                rerr = rat * np.sqrt((v["err"] / max(v["eff"], 1e-12)) ** 2 + (n["err"] / n["eff"]) ** 2)
                cells.append(f"{rat:5.2f}±{rerr:4.2f}")
                r["methods"][m][str(e)]["ratio"] = float(rat)
                r["methods"][m][str(e)]["ratio_err"] = float(rerr)
            extra = ""
            if sname.startswith("signal"):
                v, n = r["methods"][m]["0.01"], nom["methods"][m]["0.01"]
                if "mavg_median_over_M" in v and "mavg_median_over_M" in n:
                    extra = (f"  peak {n['mavg_median_over_M']:.3f}->{v['mavg_median_over_M']:.3f}"
                             f"  width {n['mavg_halfwidth68']:.3f}->{v['mavg_halfwidth68']:.3f}"
                             f"  pairing {n['pairing_correct']:.3f}->{v['pairing_correct']:.3f}")
            print(f"    {m:44s} " + "  ".join(cells) + extra)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "robustness.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
