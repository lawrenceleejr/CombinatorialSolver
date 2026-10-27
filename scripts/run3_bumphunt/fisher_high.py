"""High-mass (500-2400 GeV) expected limits with the shape-fit figure of merit, same events as compare_models.py.

Per mass: P(signal) cut tuned on one half to minimise the expected limit of a binned m_avg fit (signal strength + free
4-parameter dijet background, stat-only Fisher information at 300 fb^-1, limit = 1.96 sigma_mu), with the
falling-background requirement (QCD m_avg peak <= PEAK_MAX * M, signal peak in [0.85, 1.1] M); evaluated on the other
half. The window-counting limit (no syst.) of the same cut is stored for comparison. Dumps the spectra of the chosen cut.
usage: fisher_high.py ht_all.npy xs_study.json full_big_vars.npz big.npz ht.npz out
"""
import glob, json, os, sys
import h5py
import numpy as np

ht_all, xs_json, full_big, big, htn, out = sys.argv[1:7]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from limits import s_up, mu_up_multi  # noqa: E402
L = 300.0
PEAK_MAX = float(os.environ.get("PEAK_MAX", "0.75"))
filt = {int(k): x for k, x in json.load(open(xs_json))["filter_eff"].items()}
P = "/scratch/llee/pairnet/data_r3/prod"
fac = {f: h5py.File(f).attrs["sigma_mb"] / h5py.File(f).attrs["weight_sum"] * 1e12 for f in sorted(glob.glob(f"{P}/qcd_c*.h5"))}
pnf = [f for f in fac if f[-4] in "89"]
k_old = np.mean([fac[f] for f in pnf]) / len(pnf)
full = np.load(full_big)
HT_SUB = np.load(ht_all)[full["is_signal"] | np.isin(full["fileclass"], [8, 9])]

v = np.load(big); h = np.load(htn)
s = v["is_signal"]; q = ~s
neff = lambda x: x.sum() ** 2 / max((x ** 2).sum(), 1e-300)
wo = np.where(q, v["w"] * k_old, 0.0)
wn = h["w"] * h["fac"] / int(h["n_files"])
hi = q & (HT_SUB > 1200)
a_, b_ = neff(wo[hi]), neff(wn); al = a_ / (a_ + b_)
wo = np.where(hi, wo * al, wo); wn = wn * (1 - al)
cat = lambda k: np.concatenate([v[k], h[k]])
d = {"m1": cat("pn_m1"), "m2": cat("pn_m2"), "psig": cat("psig"), "w": np.concatenate([wo, wn]),
     "sig": np.concatenate([s, np.zeros(len(wn), bool)]), "pm": np.concatenate([v["pmass"], np.zeros(len(wn))])}
d["mavg"] = 0.5 * (d["m1"] + d["m2"]); d["A"] = np.abs(d["m1"] - d["m2"]) / np.maximum(d["m1"] + d["m2"], 1e-9)
d["valid"] = cat("pn_valid").astype(bool); d["ht"] = np.concatenate([HT_SUB, h["ht"]])
TAILDIR = os.environ.get("TAILDIR")  # enriched-tail QCD (in-job PairNet filter), merged inside its filter region
tail_info = {}
if TAILDIR:
    import tailmerge as TM
    first = {"qcdvlmT": int(os.environ.get("TAIL_VLM_FIRST", "900")), "qcdhtT": int(os.environ.get("TAIL_HT_FIRST", "500"))}
    for tname, spec in TM.TAILS.items():
        t, nf = TM.load_tail(TAILDIR, tname, "r14A", first[tname])
        if t is None:
            continue
        Fb = TM.in_filter(d["valid"], d["m1"], d["m2"], d["psig"], None, None, spec, use_delta=False) & ~d["sig"]
        Ft = TM.in_filter(t["r14A_valid"], t["r14A_m1"], t["r14A_m2"], t["r14A_psig"], None, None, spec, use_delta=False)
        d["w"] = np.where(Fb & (d["ht"] < 900), 0.0, d["w"])  # the p_T-hat > 200 sample is incomplete below H_T = 900 GeV
        tw, ti = TM.combine([(d["w"], d["ht"], Fb)], (t["wfb"], t["ht"], Ft), spec)
        n = len(tw); tail_info[tname] = {"files": nf, "events": n, **ti}
        add = {"m1": t["r14A_m1"], "m2": t["r14A_m2"], "psig": t["r14A_psig"], "w": np.where(Ft, tw, 0.0), "sig": np.zeros(n, bool),
               "pm": np.zeros(n), "valid": t["r14A_valid"].astype(bool), "ht": t["ht"]}
        add["mavg"] = 0.5 * (add["m1"] + add["m2"]); add["A"] = np.abs(add["m1"] - add["m2"]) / np.maximum(add["m1"] + add["m2"], 1e-9)
        d = {k: np.concatenate([d[k], add[k]]) for k in add}
    print("tail merge:", tail_info, flush=True)
d["half"] = np.arange(len(d["w"])) % 2
NFIT_MIN = 300
NEFF_MIN = float(os.environ.get("NEFF_MIN", "0"))  # effective QCD MC events required in the window when tuning
COUNT_F = float(os.environ["COUNT_F"]) if os.environ.get("COUNT_F") else None
FT = json.load(open(os.environ["FSYS_JSON"])) if os.environ.get("FSYS_JSON") else None


def fr(M):
    tm_ = np.array(sorted(int(k) for k in FT)); tf_ = np.array([FT[str(k)]["f_corr"] for k in tm_])
    return float(np.exp(np.interp(M, tm_, np.log(tf_))))


FB = np.linspace(0.4, 2.0, 49); FC = 0.5 * (FB[1:] + FB[:-1])


NPAR = int(os.environ.get("NPAR", "4"))


def fitbasis(X, x):
    return np.column_stack([X, np.log(x) ** 3]) if NPAR == 5 else X


def basis(x):
    return np.stack([np.ones_like(x), np.log1p(-x), np.log(x), np.log(x) ** 2], 1)


def peak(m, w):
    y, _ = np.histogram(m, np.arange(0.4, 1.6001, 0.05), weights=w)
    y = np.convolve(y, np.ones(3) / 3, mode="same")
    return 0.425 + 0.05 * np.argmax(y) if y.sum() > 0 else np.nan


def fom(M, t, hh):
    sM = d["sig"] & (d["pm"] == M); base = (d["A"] < 0.1) & (d["mavg"] > 0.4 * M) & (d["mavg"] < 2.0 * M) & (d["half"] == hh) & (d["psig"] >= t)
    ss = base & sM; qq = base & ~d["sig"]
    ns_tot = (sM & (d["half"] == hh)).sum()
    sw = np.full(ss.sum(), filt[M] / ns_tot * L)  # events per fb
    qw = 2 * d["w"][qq] * L
    sm, qm = d["mavg"][ss] / M, d["mavg"][qq] / M
    win_s, win_q = (sm > 0.85) & (sm < 1.1), (qm > 0.85) & (qm < 1.1)
    info = {"nS": int(win_s.sum()), "nB": int(win_q.sum()), "neffB": float(neff(qw[win_q])),
            "count_f0_fb": float(s_up(qw[win_q].sum(), 0.0) / sw[win_s].sum()) if win_s.any() else np.inf,
            "sig_peak_over_M": float(peak(sm, sw))}
    allq = (d["A"] < 0.1) & (d["mavg"] > 0.4 * M) & (d["mavg"] < 2.0 * M) & (d["psig"] >= t) & ~d["sig"]  # both halves: robust peak
    info["qcd_peak_over_M"] = float(peak(d["mavg"][allq] / M, d["w"][allq]))
    alls = (d["A"] < 0.1) & (d["mavg"] > 0.4 * M) & (d["mavg"] < 2.0 * M) & (d["psig"] >= t) & sM  # both halves: robust peak
    info["sig_peak_over_M"] = float(peak(d["mavg"][alls] / M, np.ones(alls.sum())))
    feas = info["qcd_peak_over_M"] <= PEAK_MAX and 0.85 <= info["sig_peak_over_M"] <= 1.1
    sh, _ = np.histogram(sm, FB, weights=sw); qh, _ = np.histogram(qm, FB, weights=qw); q2, _ = np.histogram(qm, FB, weights=qw ** 2)
    if qq.sum() < NFIT_MIN:  # too few QCD MC events to fit the shape: take it from the looser P cut that keeps NFIT_MIN events
        lq = (d["A"] < 0.1) & (d["mavg"] > 0.4 * M) & (d["mavg"] < 2.0 * M) & (d["half"] == hh) & ~d["sig"]
        tq = np.sort(d["psig"][lq])[-NFIT_MIN] if lq.sum() > NFIT_MIN else -1.0
        lq &= d["psig"] >= min(t, tq)
        lw = 2 * d["w"][lq] * L; lm = d["mavg"][lq] / M
        qh, _ = np.histogram(lm, FB, weights=lw); q2, _ = np.histogram(lm, FB, weights=lw ** 2)
        k = qw.sum() / max(lw.sum(), 1e-300); qh, q2 = qh * k, q2 * k * k
        info["shape_from_psig"] = float(min(t, tq))
    if qh.sum() <= 0 or sh.sum() <= 0:
        return np.inf, info, feas
    i0 = int(np.argmax(np.convolve(qh, np.ones(3) / 3, mode="same")))
    fit = (np.arange(len(qh)) >= i0) & (qh > 0) & (q2 > 0)
    use = np.arange(len(qh)) >= i0
    if fit.sum() < 8:
        return np.inf, info, feas
    x = FC * M / 13600.0
    X = basis(x)
    wt = qh[fit] / np.sqrt(q2[fit])
    th, *_ = np.linalg.lstsq(X[fit] * wt[:, None], np.log(qh[fit]) * wt, rcond=None)
    b = np.exp(X[use] @ th)
    A = fitbasis(X[use], x[use]) * np.sqrt(b)[:, None]; vv = sh[use] / np.sqrt(b)
    r = vv - A @ np.linalg.lstsq(A, vv, rcond=None)[0]
    rr = float(r @ r)
    # Gaussian Fisher fails when the background is tiny: scale the Poisson window-counting limit by the
    # normalisation-independent ratio sigma_mu(profiled shape fit) / sigma_mu(window counting), same fitted b
    wb = use[use] & (FC[use] > 0.85) & (FC[use] < 1.1)
    sig_win = np.sqrt(b[wb].sum()) / sh[use][wb].sum() if sh[use][wb].sum() > 0 else np.inf
    info["fisher_fb"] = float(1.96 / np.sqrt(rr)) if rr > 0 else np.inf
    info["shape_over_window"] = float((1 / np.sqrt(rr)) / sig_win) if rr > 0 else np.inf
    lim = info["count_f0_fb"] * info["shape_over_window"]
    if COUNT_F is not None:  # mass-window counting with a fractional background uncertainty instead of the shape fit
        info["S_per_fb"] = float(sw[win_s].sum()); info["B"] = float(qw[win_q].sum())
        lim = s_up(info["B"], COUNT_F) / info["S_per_fb"] if info["S_per_fb"] > 0 else np.inf
        return lim, info, feas
    if FT is not None:  # spurious-signal term f(M) x B in the window, B and S from the MC
        sw_ = sw[win_s].sum()
        lim = 1.96 * np.hypot(lim / 1.96, fr(M) * qw[win_q].sum() / sw_) if sw_ > 0 else np.inf
    return lim, info, feas


def info_cat(M, lo, hi, hh):
    sM = d["sig"] & (d["pm"] == M)
    base = (d["A"] < 0.1) & (d["mavg"] > 0.4 * M) & (d["mavg"] < 2.0 * M) & (d["half"] == hh) & (d["psig"] >= lo) & (d["psig"] < hi)
    ss = base & sM; qq = base & ~d["sig"]
    if qq.sum() < 30 or ss.sum() < 30:
        return 0.0
    ns_tot = (sM & (d["half"] == hh)).sum()
    sh, _ = np.histogram(d["mavg"][ss] / M, FB, weights=np.full(ss.sum(), filt[M] / ns_tot * L))
    qw = 2 * d["w"][qq] * L
    qh, _ = np.histogram(d["mavg"][qq] / M, FB, weights=qw); q2, _ = np.histogram(d["mavg"][qq] / M, FB, weights=qw ** 2)
    i0 = int(np.argmax(np.convolve(qh, np.ones(3) / 3, mode="same")))
    if 0.425 + 0.0333 * i0 > PEAK_MAX + 0.05:
        return 0.0
    fit = (np.arange(len(qh)) >= i0) & (qh > 0) & (q2 > 0); use = np.arange(len(qh)) >= i0
    if fit.sum() < 8:
        return 0.0
    x = FC * M / 13600.0; X = basis(x); wt = qh[fit] / np.sqrt(q2[fit])
    th, *_ = np.linalg.lstsq(X[fit] * wt[:, None], np.log(qh[fit]) * wt, rcond=None)
    b = np.exp(X[use] @ th)
    A = fitbasis(X[use], x[use]) * np.sqrt(b)[:, None]; vv = sh[use] / np.sqrt(b)
    r = vv - A @ np.linalg.lstsq(A, vv, rcond=None)[0]
    return float(r @ r)



res = {}
for M in range(500, 2401, 100):
    sM = d["sig"] & (d["pm"] == M)
    if not sM.any():
        continue
    lims, halves = [], []
    for hh in (0, 1):
        ps = d["psig"][sM & (d["A"] < 0.1) & (d["half"] == hh)]
        grid = [-1.0] + list(np.quantile(ps, 1 - np.concatenate([np.linspace(0.96, 0.2, 20), np.geomspace(0.15, 0.01, 10)])))
        best, bt = np.inf, None
        for t in grid:
            l, info, feas = fom(M, t, hh)
            if feas and info["nS"] >= 50 and info["nB"] >= 10 and info["neffB"] >= NEFF_MIN and l < best:
                best, bt = l, t
        if bt is None:
            lims.append(np.inf); halves.append({"feasible": False}); continue
        l, info, feas = fom(M, bt, 1 - hh)
        # extra P categories below bt (greedy, tuned on this half); the gain is the Gaussian information ratio on the other half
        bounds = [bt]
        tot = lambda bd, h_: sum(info_cat(M, lo, hi, h_) for lo, hi in zip(bd, [2.0] + bd[:-1]))
        for _ in range(2):
            cand = [(tot(bounds + [t], hh), t) for t in grid if t < bounds[-1]]
            if not cand or max(cand)[0] <= tot(bounds, hh) * 1.02:
                break
            bounds.append(max(cand)[1])
        e1, e3 = tot([bt], 1 - hh), tot(bounds, 1 - hh)
        info["cats_bounds"] = [float(b) for b in bounds]; info["cats_gain"] = float(np.sqrt(e1 / e3)) if e1 > 0 and e3 > 0 else 1.0
        if COUNT_F is not None:  # counting categories in P below bt, each with its own (uncorrelated) COUNT_F on B
            def cst(bd, h_, floors=True):
                sl, bl = [], []
                for lo, hi in zip(bd, [2.0] + bd[:-1]):
                    base = (d["A"] < 0.1) & (d["mavg"] > 0.85 * M) & (d["mavg"] < 1.1 * M) & (d["half"] == h_) & (d["psig"] >= lo) & (d["psig"] < hi)
                    qw = 2 * d["w"][base & ~d["sig"]] * L
                    if floors and ((base & ~d["sig"]).sum() < 10 or neff(qw) < max(NEFF_MIN, 1)):
                        return None
                    sl.append((base & sM).sum() / (sM & (d["half"] == h_)).sum() * filt[M] * L); bl.append(qw.sum())
                return sl, bl
            cb = [bt]; cur = mu_up_multi(*cst(cb, hh, False), COUNT_F)
            for _ in range(2):
                cand = [(mu_up_multi(*c, COUNT_F), t) for t in grid if 0.8 <= t < cb[-1] for c in [cst(cb + [t], hh)] if c]
                if not cand or min(cand)[0] >= cur * 0.98:
                    break
                cur, t = min(cand); cb.append(t)
            info["count_cats_bounds"] = [float(b) for b in cb]
            info["count_cats_fb"] = float(mu_up_multi(*cst(cb, 1 - hh, False), COUNT_F))
        lims.append(l if feas else np.inf); halves.append({"psig": float(bt), "feasible": bool(feas), **info})
    lim = float(np.exp(np.mean(np.log(lims))))
    cg = [x.get("cats_gain", 1.0) for x in halves]
    ccl = [x.get("count_cats_fb", np.inf) if x.get("feasible") else np.inf for x in halves]
    res[M] = {"count_cats_fb": float(np.exp(np.mean(np.log(ccl)))), "limit_cats_fb": float(np.exp(np.mean(np.log([l_ * g_ for l_, g_ in zip(lims, cg)])))), "limit_fb": lim, "count_f0_fb": float(np.exp(np.mean(np.log([x.get("count_f0_fb", np.inf) for x in halves])))), "halves": halves}
    if halves[0].get("psig") is not None:  # spectra of the half-0 cut on all events
        t = halves[0]["psig"]; base = (d["A"] < 0.1) & (d["mavg"] > 0.4 * M) & (d["mavg"] < 2.0 * M) & (d["psig"] >= t)
        os.makedirs(f"{out}/dump{os.environ.get('FTAG', '')}", exist_ok=True)
        ss = base & sM; qq = base & ~d["sig"]
        np.savez(f"{out}/dump{os.environ.get('FTAG', '')}/spec_r14A_m{M}_trig450_fisher.npz", sel="P", cuts=json.dumps({"psig": t}),
                 sig_mavg=d["mavg"][ss], sig_w=np.full(ss.sum(), filt[M] / sM.sum()), qcd_mavg=d["mavg"][qq], qcd_w=d["w"][qq])
    print(M, "shape-fit limit %.3g fb (P categories %.3g), window counting (same cut, no syst.) %.3g fb" % (lim, res[M]["limit_cats_fb"], res[M]["count_f0_fb"]),
          [(round(x.get("psig", -9), 4), x.get("qcd_peak_over_M"), x.get("feasible"), round(x.get("shape_over_window", -1), 2), x.get("nB")) for x in halves], flush=True)
json.dump({"results": res, "peak_max": PEAK_MAX, "alpha_old": al, "tail_merge": tail_info}, open(f"{out}/fisher_high{os.environ.get('FTAG', '')}.json", "w"), indent=1, default=float)
