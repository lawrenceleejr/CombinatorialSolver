"""Score one chunk under every variation. usage: scan.py chunk.npz variants.txt out.npz [nmax]"""
import sys, time
import numpy as np
from lib import session, widen, perturb, evaluate
c = np.load(sys.argv[1]); V = [l.strip() for l in open(sys.argv[2]) if l.strip()]
n = int(sys.argv[4]) if len(sys.argv) > 4 else len(c["w"])
raw = widen(*(c[k][:n].astype(np.float64) if k != "mask" else c[k][:n] for k in ("pt", "eta", "phi", "m", "mask")))
sess = session(1); out = {}; t0 = time.time()
for i, v in enumerate(V):
    o = evaluate(sess, *perturb(v, *raw))
    for k, x in o.items():
        if k in ("gap",) and i:
            continue
        out[f"v{i}_{k}"] = x.astype(np.float32) if x.dtype == np.float64 else x
    print(i, v, round(time.time() - t0), flush=True)
np.savez_compressed(sys.argv[3], variants=np.array(V), n=n, **out)
print("done", n, "events", len(V), "variants", round(time.time() - t0), "s")
