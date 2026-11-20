"""Bootstrap of the MC (Poisson(1) multiplicities per MC event, signal and QCD, halves inherited): redo tune + evaluate.
usage: boot.py low M N seed trig[,trig..]   |   boot.py high N seed"""
import json, pickle, sys, time
import numpy as np
import pipe
kind = sys.argv[1]
if kind == "low":
    M, N, seed = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]); trigs = [float(t) for t in sys.argv[5].split(",")]
    QCD = dict(np.load(f"{pipe.D}/data/qcd_sel_main.npz")); pipe.SRCARR = QCD["src"]
    sig = pipe.merged_signal(M)
    out = f"{pipe.D}/boot_low_m{M}_s{seed}.jsonl"
    for i in range(N):
        t0 = time.time(); rng = np.random.default_rng([seed, i])
        cands = pipe.prep(sig, QCD, M, rng); rec = {}
        for trig in trigs:
            pipe.TRIG = trig; r = pipe.run_mass(sig, QCD, M, cands=cands)
            rec[int(trig)] = {c: ({"limit_fb": x["limit_fb"], "lim": [h["lim"] for h in x["halves"]], "tune": [h["tune"]["lim"] for h in x["halves"]],
                                   "B": [h["xsB_fb"] for h in x["halves"]], "Bt": [h["tune"]["xsB_fb"] for h in x["halves"]]} if not c.endswith("+cats") else
                                  {"limit_fb": x["limit_fb"], "limit_corr_fb": x["limit_corr_fb"], "lim_u": [h["lim_u"] for h in x["halves"]], "lim_c": [h["lim_c"] for h in x["halves"]],
                                   "tune_u": [h["tune_u"] for h in x["halves"]]}) for c, x in r.items()}
        open(out, "a").write(json.dumps(rec, default=float) + "\n")
        print(M, i, f"{time.time() - t0:.0f}s", {t: {c: round(x["limit_fb"], 3) for c, x in v.items()} for t, v in rec.items()}, flush=True)
else:
    N, seed = int(sys.argv[2]), int(sys.argv[3])
    HA, _ = pickle.load(open(f"{pipe.D}/data/high_arrays.pkl", "rb"))
    out = f"{pipe.D}/boot_high_s{seed}.jsonl"
    for i in range(N):
        rec = {}
        for M in sorted(HA):
            if M < 1000:
                continue
            r = pipe.run_high(HA[M], 0.1, rng=np.random.default_rng([seed, i, M]))
            rec[M] = {"single": r["limit_fb"], "cats": r["count_cats_fb"], "cats_corr": r["count_cats_corr_fb"], "lim": [h["lim"] for h in r["halves"]], "tune": [h["tune"]["lim"] for h in r["halves"]],
                      "B": [h.get("B") for h in r["halves"]], "Bt": [h["tune"].get("B") for h in r["halves"]]}
        open(out, "a").write(json.dumps(rec, default=float) + "\n")
        if i % 10 == 0:
            print(i, {M: round(x["single"], 3) for M, x in rec.items()}, flush=True)
