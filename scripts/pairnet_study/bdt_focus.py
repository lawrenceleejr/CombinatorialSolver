"""Mass-window search at one mass: replace the rectangular cuts on top of PairNet by a gradient-boosted classifier.

Events: A < 0.1, valid pairing, offline H_T > 450 GeV, m_avg in a window around M (exported by lowmass_study4.py EXPORT).
Two-fold: train on half h, choose the score threshold (and optional score categories) on half h, evaluate on the
other half; limit = 10% on B Asimov counting (categories: 10% each, uncorrelated), geometric mean of the two folds.
Floors as in the cut tuning: >= 20 effective QCD MC events and >= 20 QCD MC events in the window, >= 30 signal MC events.
usage: bdt_focus.py export_dir M [lo hi]   (window edges in units of M, default 0.85 1.1)
"""
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
sys.path.insert(0, "/tmp/claude-12421/-scratch-llee/a35d08e7-9d87-5923-8644-0d53c5b0f767/scratchpad")
from limits import s_up, mu_up_multi  # noqa: E402

ex, M = sys.argv[1], int(sys.argv[2]); LO, HI = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (0.85, 1.1)
L, F = 300.0, 0.10
MIN_NEFF, MIN_NB, MIN_NS = 20, 20, 30
neff = lambda w: w.sum() ** 2 / max((w ** 2).sum(), 1e-300)


def feats(d):
    ma = d["pn_mavg"]
    p = np.clip(d["psig"], 0, 1 - 1e-7)
    am_ma = 0.5 * (d["am_m1"] + d["am_m2"])
    return np.stack([-np.log10(1 - p), d["ht"] / M, d["pn_delta_max"] / M, d["pn_delta_min"] / M, d["pn_d32_max"],
                     np.minimum(d["pn_d32_1"], d["pn_d32_2"]), d["pn_d2_63_32"], d["pn_pt_low6"] / M, d["pn_asym"], ma / M,
                     am_ma / np.maximum(ma, 1), d["am_asym"] if "am_asym" in d else np.abs(d["am_m1"] - d["am_m2"]) / np.maximum(d["am_m1"] + d["am_m2"], 1e-9),
                     d["pt6"] / M], 1)


NAMES = ["-log10(1-P)", "HT/M", "dmax/M", "dmin/M", "d32max", "d32min", "d2_63_32", "pt6low/M", "A", "mavg/M", "am_mavg/mavg", "am_A", "pt6lead/M"]
s = dict(np.load(f"{ex}/sig_m{M}.npz")); q = dict(np.load(f"{ex}/qcd_m{M}.npz"))
ws, wq = s["wfb"], q["wfb"]  # signal: fraction of produced events per MC event; QCD: fb per MC event
wins = (s["pn_mavg"] > LO * M) & (s["pn_mavg"] < HI * M); winq = (q["pn_mavg"] > LO * M) & (q["pn_mavg"] < HI * M)
Xs, Xq = feats(s)[wins], feats(q)[winq]; ws, wq = ws[wins], wq[winq]; hs, hq = s["half"][wins], q["half"][winq]
Ps, Pq = s["psig"][wins], q["psig"][winq]
print(f"M={M} window [{LO}, {HI}] M: signal MC {wins.sum()} (eff {ws.sum():.4f}), QCD MC {winq.sum()} (neff {neff(wq):.0f}, {wq.sum() * L:.3g} events)", flush=True)


def lim_at(ss, sw, qs, qw, thr):  # single category above thr; signal efficiency x2, QCD x2 (half sample)
    ks, kq = ss >= thr, qs >= thr
    if kq.sum() < MIN_NB or neff(qw[kq]) < MIN_NEFF or ks.sum() < MIN_NS:
        return np.inf, None
    S = 2 * sw[ks].sum() * L; B = 2 * qw[kq].sum() * L
    return s_up(B, F) / S, (S, B, int(kq.sum()), neff(qw[kq]))


def cats_at(ss, sw, qs, qw, edges):
    S, B = [], []
    e = list(edges) + [np.inf]
    for lo, hi in zip(e[:-1], e[1:]):
        ks, kq = (ss >= lo) & (ss < hi), (qs >= lo) & (qs < hi)
        if kq.sum() < MIN_NB or neff(qw[kq]) < MIN_NEFF:
            return np.inf
        S.append(2 * sw[ks].sum() * L); B.append(2 * qw[kq].sum() * L)
    return mu_up_multi(S, B, F)


res = {"P": [], "BDT": [], "BDT_cats": []}
for h in (0, 1):
    tr_s, tr_q = hs == h, hq == h; te_s, te_q = ~tr_s, ~tr_q
    # train: QCD weights normalised to the signal total, clipped against single huge-weight events
    wq_tr = wq[tr_q] / wq[tr_q].sum(); wq_tr = np.minimum(wq_tr, np.quantile(wq_tr, 0.999)); wq_tr /= wq_tr.sum()
    ws_tr = ws[tr_s] / ws[tr_s].sum()
    X = np.concatenate([Xs[tr_s], Xq[tr_q]]); y = np.concatenate([np.ones(tr_s.sum()), np.zeros(tr_q.sum())]); w = np.concatenate([ws_tr, wq_tr])
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    clf.fit(X, y, sample_weight=w)
    sc_s_tr, sc_q_tr = clf.predict_proba(Xs[tr_s])[:, 1], clf.predict_proba(Xq[tr_q])[:, 1]
    sc_s_te, sc_q_te = clf.predict_proba(Xs[te_s])[:, 1], clf.predict_proba(Xq[te_q])[:, 1]
    # thresholds from the training half's signal-score quantiles
    grid = np.quantile(sc_s_tr, 1 - np.geomspace(0.6, 0.003, 60))
    for name, (a_s, a_q, b_s, b_q, g) in {"P": (Ps[tr_s], Pq[tr_q], Ps[te_s], Pq[te_q], np.quantile(Ps[tr_s], 1 - np.geomspace(0.6, 0.003, 60))),
                                          "BDT": (sc_s_tr, sc_q_tr, sc_s_te, sc_q_te, grid)}.items():
        best = min(((lim_at(a_s, ws[tr_s], a_q, wq[tr_q], t)[0], t) for t in g), key=lambda x: x[0])
        l, info = lim_at(b_s, ws[te_s], b_q, wq[te_q], best[1])
        res[name].append(l)
        print(f"  fold {h} {name}: thr {best[1]:.5g} train-half limit {best[0]:.3g} fb -> other half {l:.3g} fb  (S per fb, B, nB, neffB) {info}", flush=True)
        if name == "BDT":
            edges = [best[1]]
            for _ in range(2):
                cand = [(cats_at(a_s, ws[tr_s], a_q, wq[tr_q], [t] + edges), t) for t in g if t < edges[0]]
                cur = cats_at(a_s, ws[tr_s], a_q, wq[tr_q], edges)
                if not cand or min(cand)[0] >= cur * 0.98:
                    break
                edges = [min(cand)[1]] + edges
            lc = cats_at(b_s, ws[te_s], b_q, wq[te_q], edges)
            res["BDT_cats"].append(lc)
            print(f"  fold {h} BDT categories {[round(e, 5) for e in edges]} -> other half {lc:.3g} fb", flush=True)
    imp = getattr(clf, "feature_importances_", None)
gm = lambda x: float(np.exp(np.mean(np.log(x))))
print(f"M={M}: P cut only {gm(res['P']):.3g} fb | BDT {gm(res['BDT']):.3g} fb | BDT + categories {gm(res['BDT_cats']):.3g} fb", flush=True)
