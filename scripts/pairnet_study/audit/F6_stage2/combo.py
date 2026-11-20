#!/usr/bin/env python
"""Item 5: combined recipe = current P+all cuts with the mass-window edges re-tuned (nested), then 3 P categories.
Compared with the current recipe (P+all+cats) re-run under the same nested procedure. usage: combo.py out.json [masses]"""
import json, sys, time
import numpy as np
import f6lib as F

out = sys.argv[1]
MASSES = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else F.LOW_MASSES + F.HIGH_MASSES
WG = {"wlo": [0.85, 0.80, 0.83, 0.88, 0.90, 0.92, 0.94], "whi": [1.10, 1.03, 1.05, 1.07, 1.15]}
t0 = time.time()
Q = F.load_qcd()
res = {}
for M in MASSES:
    S = F.load_sig(M)
    if S is None:
        continue
    w = F.weights(Q, M)
    Bt, St = F.make_tab(Q, M, w), F.make_tab(S, M, S["wfb"])
    res[M] = {}
    for f in (0.10, 0.05):
        folds = [{}, {}]
        for name, extra, g in (("P+all", [], None), ("P+all|win", ["wlo", "whi"], WG)):
            s, sc = F.run_conf(St, Bt, F.CONF["P+all"] + extra, f, grids=g, cats=3, smin=0.8)
            for i in (0, 1):
                folds[i][name] = s[i]; folds[i][name + "+cats"] = sc[i]
        r = F.summarize(folds, f, len(S["w"]), len(Q["w"]), "P+all+cats")
        res[M][f"f{f:g}"] = {"sel": r}
        print(f"M={M} f={f:g}: ref {r['P+all+cats']['limit_fb']:.4g} (neff {r['P+all+cats']['neffB']:.0f}) | "
              + "  ".join(f"{k} {v['ratio']:.2f}+-{v['ratio_err']:.2f}" for k, v in r.items()) + f"  {time.time() - t0:.0f}s", flush=True)
    json.dump(res, open(out, "w"), indent=1, default=float)
