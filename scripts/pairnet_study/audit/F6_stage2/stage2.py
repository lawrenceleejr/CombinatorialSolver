#!/usr/bin/env python
"""Item 1: pooled, mass-parametrised second stage. Every selected candidate (r14A argmax pairing, A < 0.1,
H_T > 450 GeV) is described in units of its own m_avg, plus log m_avg; the models learn signal (all masses, inside
their windows) against the physically weighted QCD, per log-m_avg bin normalised to the same total.
Nesting: fold h is split in two sub-folds; a model trained on one scores the other (cross-fitted scores used to tune
thresholds on fold h); the evaluation fold 1-h is scored by the average of the two models, neither of which saw it.
Models: dr2 (2-D binned density ratio in logit P, log H_T/m_avg), lin2 (logit P + a log(H_T/m_avg), a per mass on the
tuning fold), gbdt_core / gbdt_ext (HistGradientBoosting), mlp_core (small MLP on a weight-resampled sample).
usage: stage2.py out.json [masses] [pmin]"""
import json, os, sys, time
import numpy as np
import f6lib as F

out = sys.argv[1]
MASSES = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 and sys.argv[2] != "all" else F.LOW_MASSES + F.HIGH_MASSES
PMIN = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2  # the second stage is defined for P >= PMIN only
MODELS = os.environ.get("MODELS", "dr2,gbdt_core,gbdt_ext,mlp_core").replace("+", ",").split(",")
t0 = time.time()
Q = F.load_qcd()
SIG = {M: F.load_sig(M) for M in F.LOW_MASSES + F.HIGH_MASSES}
SIG = {M: s for M, s in SIG.items() if s is not None}
print(f"loaded {time.time() - t0:.0f}s; signal masses {list(SIG)}", flush=True)

CORE = ["lp", "lht", "dmin", "dmax", "d32_max", "d2_63_32", "pt_low6", "asym", "mest", "unused", "n_jets", "dy", "pt6sys", "tpt_max", "tpt_min", "lm"]
EXT = CORE + ["m6", "pt_isr", "pt8", "eta_max6", "mj_sum6", "pt1", "pt2", "dphi", "y6", "ent", "margin", "n_sym", "p_sym", "p_near", "agree", "mspread3", "mspread5",
              "sl_margin", "sl_ent", "sl_dm", "d32_min"]


def feats(d, k):
    m = np.maximum(d["pn_mavg"][k], 1e-3); g = lambda c: d[c][k].astype(np.float32)
    p = np.clip(d["psig"][k].astype(np.float64), 1e-7, 1 - 1e-7)
    x = {"lp": np.log(p / (1 - p)), "lht": np.log(np.maximum(g("ht") / m, 1e-3)), "dmin": g("pn_delta_min") / m, "dmax": g("pn_delta_max") / m,
         "d32_max": g("pn_d32_max"), "d2_63_32": g("pn_d2_63_32"), "pt_low6": g("pn_pt_low6") / m, "asym": g("pn_asym"), "mest": g("m_est") / m,
         "unused": g("pt_unused") / m, "n_jets": g("n_jets"), "dy": g("dy"), "pt6sys": g("pt6sys") / m, "tpt_max": g("tpt_max") / m, "tpt_min": g("tpt_min") / m,
         "lm": np.log(m), "m6": g("m6") / m, "pt_isr": g("pt_isr") / m, "pt8": g("pt8") / m, "eta_max6": g("eta_max6"), "mj_sum6": g("mj_sum6") / m,
         "pt1": g("pt1") / m, "pt2": g("pt2") / m, "dphi": g("dphi"), "y6": g("y6"), "ent": g("ent"), "margin": np.minimum(g("margin"), 30), "n_sym": g("n_sym"),
         "p_sym": g("p_sym"), "p_near": g("p_near"), "agree": g("agree"), "mspread3": g("mspread3"), "mspread5": g("mspread5"),
         "sl_margin": np.minimum(g("sl_margin"), 30), "sl_ent": g("sl_ent"), "sl_dm": np.abs(g("sl_ma_sig") / m - 1), "d32_min": np.minimum(g("pn_d32_1"), g("pn_d32_2"))}
    return np.column_stack([np.nan_to_num(x[c], nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32) for c in EXT])


# ---- pooled candidate set: rows = QCD candidates then signal candidates (all masses)
wq = np.where(Q["pn_mavg"] < 1150, Q["wL"], Q["wH"])
kq = np.where(Q["pn_valid"].astype(bool) & (Q["pn_asym"] < 0.1) & (Q["ht"] > F.TRIG) & (wq > 0) & (Q["pn_mavg"] > 120) & (Q["pn_mavg"] < 3900) & (Q["psig"] >= PMIN))[0]
ks = {M: np.where(s["pn_valid"].astype(bool) & (s["pn_asym"] < 0.1) & (s["ht"] > F.TRIG) & (s["psig"] >= PMIN) & (s["pn_mavg"] > 0.4 * M) & (s["pn_mavg"] < 1.6 * M))[0] for M, s in SIG.items()}
X = np.concatenate([feats(Q, kq)] + [feats(SIG[M], ks[M]) for M in SIG])
y = np.concatenate([np.zeros(len(kq), np.int8)] + [np.ones(len(ks[M]), np.int8) for M in SIG])
half = np.concatenate([Q["half"][kq]] + [SIG[M]["half"][ks[M]] for M in SIG])
sub = np.concatenate([Q["sub"][kq]] + [SIG[M]["sub"][ks[M]] for M in SIG])
mav = np.concatenate([Q["pn_mavg"][kq]] + [SIG[M]["pn_mavg"][ks[M]] for M in SIG])
inwin = np.concatenate([np.ones(len(kq), bool)] + [(SIG[M]["pn_mavg"][ks[M]] > 0.85 * M) & (SIG[M]["pn_mavg"][ks[M]] < 1.1 * M) for M in SIG])
wraw = np.concatenate([wq[kq]] + [SIG[M]["wfb"][ks[M]] / SIG[M]["wfb"][ks[M]].sum() for M in SIG])
off = np.cumsum([0, len(kq)] + [len(ks[M]) for M in SIG]); OFFS = dict(zip(SIG, off[1:-1]))
print(f"pooled rows: QCD {len(kq)}, signal {int(y.sum())} (in window {int((inwin & (y == 1)).sum())}); P >= {PMIN}", flush=True)
BW = np.log(1.1 / 0.85) / 2
mb = np.floor(np.log(mav / 100.0) / BW).astype(int)


def train_weights(rows):
    """Per log-m bin: QCD total = signal total = 1 (bins without both are dropped); signal only inside its window."""
    w = np.zeros(len(rows))
    yy, bb, ww, iw = y[rows], mb[rows], wraw[rows], inwin[rows]
    for b in np.unique(bb):
        q = (bb == b) & (yy == 0); s = (bb == b) & (yy == 1) & iw
        if q.sum() < 50 or s.sum() < 50:
            continue
        w[q] = ww[q] / ww[q].sum(); w[s] = ww[s] / ww[s].sum()
    return w


IC = [EXT.index(c) for c in CORE]


def fit_predict(model, tr, te_list):
    w = train_weights(tr); k = w > 0
    Xtr, ytr, wtr = X[tr][k], y[tr][k], w[k] * k.sum() / w[k].sum()
    if model == "dr2":
        ex, ey = np.linspace(-2, 10, 13), np.linspace(0.5, 2.7, 12)
        cx = lambda a: np.clip(a[:, 0], ex[0] + 1e-6, ex[-1] - 1e-6); cy = lambda a: np.clip(a[:, 1], ey[0] + 1e-6, ey[-1] - 1e-6)
        H = [np.histogram2d(cx(Xtr[ytr == c]), cy(Xtr[ytr == c]), [ex, ey], weights=wtr[ytr == c])[0] for c in (0, 1)]
        sm = lambda h: sum(np.roll(np.roll(np.pad(h, 1, mode="edge"), i, 0), j, 1)[1:-1, 1:-1] * wt for i, j, wt in
                           ((0, 0, 4), (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1))) / 8
        Hq, Hs = sm(H[0]) / H[0].sum(), sm(H[1]) / H[1].sum()
        R = np.log((Hs + 1e-5) / (Hq + 1e-5))
        return [R[np.digitize(cx(X[te]), ex) - 1, np.digitize(cy(X[te]), ey) - 1].astype(np.float32) for te in te_list]
    if model.startswith("gbdt"):
        from sklearn.ensemble import HistGradientBoostingClassifier as H
        cols = IC if model == "gbdt_core" else list(range(len(EXT)))
        m = H(max_iter=250, learning_rate=0.06, max_leaf_nodes=8, min_samples_leaf=300, l2_regularization=20.0, max_bins=63, early_stopping=False, random_state=1)
        m.fit(Xtr[:, cols], ytr, sample_weight=wtr)
        return [m.decision_function(X[te][:, cols]).astype(np.float32) for te in te_list]
    if model == "mlp_core":
        from sklearn.neural_network import MLPClassifier
        rng = np.random.default_rng(3); N = 150000
        idx = np.concatenate([rng.choice(np.where(ytr == c)[0], N, p=wtr[ytr == c] / wtr[ytr == c].sum()) for c in (0, 1)])
        mu, sd = Xtr[:, IC].mean(0), Xtr[:, IC].std(0) + 1e-6
        z = lambda a: np.clip((a[:, IC] - mu) / sd, -8, 8)
        m = MLPClassifier((32, 32), alpha=3e-2, batch_size=1024, learning_rate_init=2e-3, max_iter=40, early_stopping=True, validation_fraction=0.15, n_iter_no_change=5, random_state=1)
        m.fit(z(Xtr[idx]), ytr[idx])
        pr = lambda a: np.clip(m.predict_proba(z(a))[:, 1].astype(np.float64), 1e-7, 1 - 1e-7)
        return [np.log(pr(X[te]) / (1 - pr(X[te]))).astype(np.float32) for te in te_list]
    raise ValueError(model)


# scores[model][h]: score used when tuning on fold h (cross-fitted inside fold h, average of the two models on fold 1-h)
scores = {}
for model in MODELS:
    scores[model] = {}
    for h in (0, 1):
        sc = np.zeros(len(y), np.float32)
        a, b, e = np.where((half == h) & (sub == 0))[0], np.where((half == h) & (sub == 1))[0], np.where(half != h)[0]
        pb, pe1 = fit_predict(model, a, [b, e]); pa, pe2 = fit_predict(model, b, [a, e])
        sc[b] = pb; sc[a] = pa; sc[e] = 0.5 * (pe1 + pe2)
        scores[model][h] = sc
    print(f"trained {model} {time.time() - t0:.0f}s", flush=True)
if os.environ.get("SAVE_SCORES"):
    np.savez_compressed(os.environ["SAVE_SCORES"], kq=kq, **{f"ks_{M}": v for M, v in ks.items()}, **{f"{m}_{h}": scores[m][h] for m in scores for h in (0, 1)})
ALIN = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
res = {}
for M in MASSES:
    if M not in SIG:
        continue
    S = SIG[M]; w = F.weights(Q, M)
    Bt, St = F.make_tab(Q, M, w), F.make_tab(S, M, S["wfb"])
    posq = np.full(len(Q["w"]), -1); posq[kq] = np.arange(len(kq))
    poss = np.full(len(S["w"]), -1); poss[ks[M]] = OFFS[M] + np.arange(len(ks[M]))
    pb, ps = posq[Bt.eid], poss[St.eid]

    def col(arr):  # pooled-row array -> table columns (-inf where the candidate is not in the pooled set, e.g. P < PMIN)
        return np.where(pb >= 0, arr[np.maximum(pb, 0)], -np.inf).astype(np.float32), np.where(ps >= 0, arr[np.maximum(ps, 0)], -np.inf).astype(np.float32)
    res[M] = {}
    for f in [float(x) for x in os.environ.get("FS", "0.10,0.05").split(",")]:
        folds = [{}, {}]
        for name, free in F.CONF.items():
            s, sc = F.run_conf(St, Bt, free, f, cats=3, smin=0.8)
            for i in (0, 1):
                folds[i][name] = s[i]; folds[i][name + "+cats"] = sc[i]
        for model in MODELS:
            for h in (0, 1):
                cb, cs = col(scores[model][h]); Bt.add("s2", cb, ">="); St.add("s2", cs, ">=")
                for name, free in (("", ["s2"]), ("+all", ["s2"] + [v for v in F.CONF["P+all"] if v != "psig"]), ("+P+all", ["s2"] + F.CONF["P+all"])):
                    cuts, best = F.tune(St, Bt, free, h, f)
                    s = F.Sel(St, Bt, cuts, 1 - h); s.feasible = bool(np.isfinite(best)) and F.shape_ok(St, Bt, cuts, h)
                    folds[h][model + name] = s
                    if name == "":
                        if s.feasible:
                            bounds = F.tune_cats(St, Bt, cuts, "s2", h, f, 3)
                            sc = F.Sel(St, Bt, cuts, 1 - h, "s2", bounds)
                        else:
                            sc = F.Sel(St, Bt, cuts, 1 - h); sc.feasible = False
                        folds[h][model + "+cats"] = sc
        # lin2: logit P + a log(H_T / m_avg), a chosen on the tuning fold
        lpq, lhq = X[:, 0], X[:, 1]
        for h in (0, 1):
            best = None
            for a in ALIN:
                cb, cs = col(lpq + a * lhq); Bt.add("s2", cb, ">="); St.add("s2", cs, ">=")
                cuts, lim = F.tune(St, Bt, ["s2"], h, f)
                if best is None or lim < best[0]:
                    best = (lim, a, cuts)
            lim, a, cuts = best
            cb, cs = col(lpq + a * lhq); Bt.add("s2", cb, ">="); St.add("s2", cs, ">=")
            s = F.Sel(St, Bt, cuts, 1 - h); s.feasible = bool(np.isfinite(lim)) and F.shape_ok(St, Bt, cuts, h); s.cuts = {**cuts, "a": a}
            folds[h]["lin2"] = s
        r = F.summarize(folds, f, len(S["w"]), len(Q["w"]), "P+all")
        rP = F.summarize([{k: v for k, v in folds[i].items() if k in ("P", "dr2", "lin2")} for i in (0, 1)], f, len(S["w"]), len(Q["w"]), "P")
        for k in rP:
            r[k]["ratio_vs_P"] = rP[k]["ratio"]; r[k]["ratio_vs_P_err"] = rP[k]["ratio_err"]
        res[M][f"f{f:g}"] = {"sel": r}
        print(f"M={M} f={f:g}: P {r['P']['limit_fb']:.4g}  P+all {r['P+all']['limit_fb']:.4g} (neff {r['P+all']['neffB']:.0f})  +cats {r['P+all+cats']['ratio']:.2f}+-{r['P+all+cats']['ratio_err']:.2f} | "
              + "  ".join(f"{k} {v['ratio']:.2f}+-{v['ratio_err']:.2f}({v['neffB']:.0f})" for k, v in r.items() if not k.startswith("P")) + f"  {time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(out, "w"), indent=1, default=float)
