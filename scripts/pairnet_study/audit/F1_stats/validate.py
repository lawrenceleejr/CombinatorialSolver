"""Check that pipe.py reproduces the published JSON files (low-mass trig 450/500/600 and high-mass)."""
import json, sys, time, pickle
import numpy as np
import pipe
t0 = time.time()
QCD, merge = pipe.build_qcd(verbose=True)
pub = json.load(open("/scratch/llee/pairnet/lowmass14win10t_pk0.75/lowmass_eval_r14A_trig450_f0.1.json"))
print("merge 900-1200 alpha old", merge["900-1200"]["old"]["alpha"], "published", pub["merge"]["900-1200"]["old"]["alpha"], "tail events", merge["qcdvlmT"]["events"], merge["qcdhtT"]["events"], f"{time.time() - t0:.0f}s", flush=True)
pipe.TRIG = 450.0
for M in (700, 400):
    sig = pipe.merged_signal(M); t1 = time.time()
    r = pipe.run_mass(sig, QCD, M, confs=("P", "P+boost", "P+all"))
    for k, x in r.items():
        p = pub["results"][str(M)][k]
        print(M, k, "mine %.5g published %.5g" % (x["limit_fb"], p["limit_fb"]), ("corr %.5g / %.5g" % (x["limit_corr_fb"], p["limit_corr_fb"])) if "limit_corr_fb" in x else "", f"{time.time() - t1:.0f}s", flush=True)
d, filt, hi = pipe.build_high(verbose=True)
ph = json.load(open("/scratch/llee/pairnet/fisher_high/fisher_high_win10t_pk0.75.json"))
for M in (1000, 1500, 2000):
    A = pipe.high_arrays(d, filt, M); r = pipe.run_high(A, 0.1)
    print(M, "high: mine %.5g published %.5g; cats %.5g / %.5g" % (r["limit_fb"], ph["results"][str(M)]["limit_fb"], r["count_cats_fb"], ph["results"][str(M)]["count_cats_fb"]), flush=True)
