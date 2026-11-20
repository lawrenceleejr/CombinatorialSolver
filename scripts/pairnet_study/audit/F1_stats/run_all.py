"""Deterministic re-runs with full bookkeeping (tuning-half values, composition, peaks).
usage: run_all.py variant nproc     variant: main | peakboth | t700_full | t700_fullw | t700_f5 | t700_f5w | high"""
import json, pickle, sys, time
from multiprocessing import Pool
import numpy as np
import pipe
variant, nproc = sys.argv[1], int(sys.argv[2])
CFG = {"main": dict(q="main", masses=[250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000], f=0.1, win=(0.85, 1.1), extra=False),
       "peakboth": dict(q="main", masses=[250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000], f=0.1, win=(0.85, 1.1), extra=False, both=True),
       "t700_full": dict(q="t700", masses=[600, 650, 700, 750, 800], f=0.1, win=(0.85, 1.1), extra=True),
       "t700_fullw": dict(q="t700", masses=[600, 650, 700, 750, 800], f=0.1, win=(0.92, 1.04), extra=True),
       "t700_f5": dict(q="t700", masses=[600, 650, 700, 750, 800], f=0.05, win=(0.85, 1.1), extra=True),
       "t700_f5w": dict(q="t700", masses=[600, 650, 700, 750, 800], f=0.05, win=(0.92, 1.04), extra=True)}


def job(M):
    c = CFG[variant]; t0 = time.time()
    sig = pipe.merged_signal(M, c["extra"])
    cands = pipe.prep(sig, QCD, M)
    out = {}
    for trig in (450.0, 500.0, 600.0):
        pipe.TRIG = trig
        out[int(trig)] = pipe.run_mass(sig, QCD, M, detail=True, cands=cands)
    print(variant, M, {t: {k: round(v["limit_fb"], 4) for k, v in r.items()} for t, r in out.items()}, f"{time.time() - t0:.0f}s", flush=True)
    return M, out


def jobh(M):
    A = HA[M]; out = {}
    for f in (0.1, 0.05):
        out[f"f{f}"] = pipe.run_high(A, f, detail=True)
    out["f0.1_valid"] = pipe.run_high(A, 0.1, detail=True, extra=lambda d: d["valid"])
    out["f0.1_valid_ht450"] = pipe.run_high(A, 0.1, detail=True, extra=lambda d: d["valid"] & (d["ht"] > 450))
    out["f0.1_nopeak"] = pipe.run_high(A, 0.1, peak_max=99)
    print("high", M, {k: round(v["limit_fb"], 4) for k, v in out.items()}, flush=True)
    return M, out


if variant == "high":
    HA, hi = pickle.load(open(f"{pipe.D}/data/high_arrays.pkl", "rb"))
    res = dict(Pool(nproc).map(jobh, sorted(HA), chunksize=1))
else:
    c = CFG[variant]
    QCD = dict(np.load(f"{pipe.D}/data/qcd_sel_{c['q']}.npz")); pipe.SRCARR = QCD["src"]
    pipe.FSYS, pipe.WIN, pipe.PEAK_BOTH = c["f"], c["win"], bool(c.get("both"))
    res = dict(Pool(nproc).map(job, c["masses"][::-1], chunksize=1))
json.dump(res, open(f"{pipe.D}/rerun_{variant}.json", "w"), indent=1, default=float)
