"""
Which strategy finds the signal best?  Every method is reduced to two numbers
per event, an event score (higher = more signal-like) and a mass m_avg from its
chosen 3+3 pairing, and all are scored the same way on the same held-out
events:

  FOM(M) = max over (score cut, mass window) of  S / sqrt(B)
    S = fraction of all signal events at mass M selected and inside the window
    B = weighted fraction of all QCD (>= 6 jets) selected and inside the window
    score cut: keeps 100, 90, ..., 10% of the signal at M
    window   : shortest interval holding 50/68/80/90% of the selected signal
    B floor  : three typical QCD events from the signal's HT range

  python scripts/search_study.py --out results/search_study
"""

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from src.features import (classical_pairing, event_shape_features, four_vectors, inv_mass,  # noqa: E402
                          pairing_features, stack)
from src.pairdata import concat, load_events  # noqa: E402
from src.pairnet import enumerate_interpretations, load_checkpoint  # noqa: E402

INTERP7 = enumerate_interpretations(7, ((3, 3),))
G1 = np.array([a for a, _ in INTERP7])
G2 = np.array([b for _, b in INTERP7])


def mavg_asym(d, g1, g2):
    p = four_vectors(d)
    ar = np.arange(len(g1))[:, None]
    m1, m2 = inv_mass(p[ar, g1].sum(1)), inv_mass(p[ar, g2].sum(1))
    return 0.5 * (m1 + m2), np.abs(m1 - m2) / np.maximum(m1 + m2, 1e-9)


@torch.no_grad()
def pairnet_outputs(path, d, bs=2048):
    net, _ = load_checkpoint(path)
    n = net.n_jets
    t = {k: torch.from_numpy(np.ascontiguousarray(d[k][:, :n])) for k in ("pt", "eta", "phi", "m", "mask")}
    t = {k: (v.float() if v.dtype != torch.bool else v) for k, v in t.items()}
    sig_best, mix_best, gate = [], [], []
    for i in range(0, len(t["pt"]), bs):
        o = net(*(t[k][i: i + bs] for k in ("pt", "eta", "phi", "m", "mask")))
        A = net.n_assign
        sig_best.append(o["assign_logits"].argmax(1))
        mix_best.append(o["logits"][:, :A].argmax(1))
        gate.append(torch.sigmoid(o["gate_logit"]) if "gate_logit" in o else 1 - o["logits"].softmax(1)[:, -1])
    return torch.cat(sig_best).numpy(), torch.cat(mix_best).numpy(), torch.cat(gate).numpy()


def shortest(v, frac):
    v = np.sort(v)
    k = max(int(np.ceil(frac * len(v))), 2)
    w = v[k - 1:] - v[: len(v) - k + 1]
    i = int(np.argmin(w))
    return v[i], v[i + k - 1]


def fom_table(score, mavg, is_sig, pmass, w, ht):
    out = {}
    s_all = is_sig
    wq, mq, sq, hq = w[~s_all], mavg[~s_all], score[~s_all], ht[~s_all]
    wtot = wq.sum()
    for M in np.unique(pmass[s_all]):
        sm = s_all & (pmass == M)
        ss, ms = score[sm], mavg[sm]
        lo, hi = np.quantile(ht[sm], [0.1, 0.9])
        reg = (hq > lo) & (hq < hi)
        floor = 3 * (wq[reg].mean() if reg.any() else wq.mean()) / wtot
        best = {"fom": 0.0}
        for keep in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
            thr = -np.inf if keep >= 1 else np.quantile(ss, 1 - keep)
            sel_s = ss >= thr
            if sel_s.sum() < 30:
                continue
            sel_q = sq >= thr
            for q in (0.5, 0.68, 0.8, 0.9):
                a, b = shortest(ms[sel_s], q)
                S = (sel_s & (ms >= a) & (ms <= b)).sum() / sm.sum()
                B = (wq * (sel_q & (mq >= a) & (mq <= b))).sum() / wtot
                fom = S / np.sqrt(max(B, floor))
                if fom > best["fom"]:
                    best = {"fom": float(fom), "keep": keep, "window_frac": q, "S": float(S),
                            "B": float(B), "B_floored": bool(B < floor), "window": [float(a), float(b)],
                            "n_qcd": int((sel_q & (mq >= a) & (mq <= b)).sum())}
        out[int(M)] = best
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sig", default="data/pythia/sig_*_s2.h5")
    ap.add_argument("--bkg", nargs="+", default=None)
    ap.add_argument("--gated", default="runs/gated_n7/model.pt")
    ap.add_argument("--extra", nargs="*", default=[], help="name=path to more PairNet checkpoints")
    ap.add_argument("--legacy", default="runs/legacy_asis/model.pt")
    ap.add_argument("--bdt", default="runs/bdt/bdt.pkl")
    ap.add_argument("--out", default="results/search_study")
    a = ap.parse_args()
    torch.set_num_threads(4)
    bkg_files = a.bkg or sorted(set(glob.glob("data/pythia/qcd_s8.h5") + glob.glob("data/pythia/qcd_s9.h5")
                                    + glob.glob("data/pythia/qcd_s1[0-9].h5")))
    print("QCD eval files:", [os.path.basename(f) for f in bkg_files])
    sig = load_events(a.sig, 20, 3, True, None, min_jets=6)
    bkg = load_events(bkg_files, 20, 3, False, None, min_jets=6)
    d = concat(sig, bkg)
    # truth pairing within the leading 7 jets (index into INTERP7), for the upper bound
    t7 = concat(load_events(a.sig, 7, 3, True, None, min_jets=6),
                load_events(bkg_files, 7, 3, False, None, min_jets=6))["target"]
    s, pm, w = d["is_signal"], d["pmass"], d["w"]
    ht = (d["pt"] * d["mask"]).sum(1)
    print(f"eval: {s.sum()} signal, {(~s).sum()} QCD (weighted eff. N "
          f"{w[~s].sum() ** 2 / (w[~s] ** 2).sum():.0f})")

    methods = {}
    g1, g2 = classical_pairing(d)
    m_cl, as_cl = mavg_asym(d, g1, g2)
    methods["argmin |Δm|, asym cut"] = (-as_cl, m_cl)

    if os.path.exists(a.legacy):
        sys.path.insert(0, HERE)
        from eval_legacy import legacy_logits
        from src.combinatorics import enumerate_assignments
        lg, _ = legacy_logits(a.legacy, d, 7)
        asg = enumerate_assignments(7)
        b = lg.argmax(1).numpy()
        lg1 = np.array([asg[i][1] for i in b])
        lg2 = np.array([asg[i][2] for i in b])
        m_lg, as_lg = mavg_asym(d, lg1, lg2)
        methods["current network, asym cut"] = (-as_lg, m_lg)

    if os.path.exists(a.bdt):
        bd = pickle.load(open(a.bdt, "rb"))
        X, keys = stack([event_shape_features(d), pairing_features(d, g1, g2)])
        keep = [i for i, k in enumerate(keys) if k not in bd["drop"]]
        p_bdt = bd["model"].predict_proba(X[:, keep])[:, 1]
        methods["expert BDT, argmin pairing"] = (p_bdt, m_cl)

    ckpts = {"gated PairNet": a.gated}
    ckpts.update(dict(x.split("=", 1) for x in a.extra))
    for name, path in ckpts.items():
        sb, mb, gate = pairnet_outputs(path, d)
        m_mix, as_mix = mavg_asym(d, G1[mb], G2[mb])
        m_sig, _ = mavg_asym(d, G1[sb], G2[sb])
        methods[f"{name}, asym cut"] = (-as_mix, m_mix)
        methods[f"{name}, gate cut"] = (gate, m_sig)
        if name == "gated PairNet":
            if "expert BDT, argmin pairing" in methods:
                methods["expert BDT, PairNet pairing"] = (p_bdt, m_sig)
            tb = np.where(s & (t7 >= 0), t7, sb)
            m_tr, _ = mavg_asym(d, G1[tb], G2[tb])
            methods["gated PairNet, gate cut, TRUTH pairing (bound)"] = (gate, m_tr)

    from sklearn.metrics import roc_auc_score
    res = {}
    for name, (score, mavg) in methods.items():
        r = fom_table(score, mavg, s, pm, w, ht)
        r["auc"] = float(roc_auc_score(s, score, sample_weight=np.where(s, 1.0 / s.sum(), w / w[~s].sum())))
        res[name] = r
        print(f"{name:52s} AUC {r['auc']:.4f} | FOM " +
              " ".join(f"{m}:{r[m]['fom']:7.1f}{'*' if r[m].get('B_floored') else ' '}"
                       for m in sorted(k for k in r if k != "auc")), flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "fom.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "per_event.npz"), is_signal=s, pmass=pm, w=w, ht=ht,
                        **{f"score::{k}": v[0] for k, v in methods.items()},
                        **{f"mavg::{k}": v[1] for k, v in methods.items()})
    print("(* = background at the floor: QCD MC statistics limit that point)")


if __name__ == "__main__":
    main()
