"""IR-safety probe: add one extra jet with pT = x GeV (random eta in [-2.5,2.5], phi, m = 0.1 pT) to events with < 12 jets; compare P."""
import numpy as np
from lib import *
sess = session(1); rng = np.random.default_rng(9)
for g, n in (("TvlmT_3", 2500), ("sig700_5", 2500)):
    c = np.load(f"cache/chunks/{g}.npz"); raw = [c[k][:n].astype(np.float64) if k != "mask" else c[k][:n] for k in ("pt", "eta", "phi", "m", "mask")]
    pt, eta, phi, m, mask = widen(*raw)
    nj = mask.sum(1); sel = nj < 12
    base = evaluate(sess, pt, eta, phi, m, mask)
    print(g, "events", n, "with <12 jets", sel.mean().round(3))
    for x in (1e-3, 1.0, 5.0, 10.0, 20.0, 30.0):
        q = [a.copy() for a in (pt, eta, phi, m, mask)]
        ar = np.arange(n); k = nj
        q[0][ar, k] = x; q[1][ar, k] = rng.uniform(-2.5, 2.5, n); q[2][ar, k] = rng.uniform(-np.pi, np.pi, n); q[3][ar, k] = 0.1 * x; q[4][ar, k] = True
        o = evaluate(sess, *q)
        l0 = np.log(base["psig"]) - np.log1p(-base["psig"]); l1 = np.log(o["psig"]) - np.log1p(-o["psig"])
        s = sel & base["valid"]
        print(f"  +jet {x:7.3f} GeV: |dlogit| median {np.median(np.abs(l1 - l0)[s]):.3f}, 90% {np.quantile(np.abs(l1 - l0)[s], .9):.3f};"
              f" pairing changed {(o['best'] != base['best'])[s].mean():.3f}; P>=0.997: {(base['psig'] >= .997)[s].mean():.4f} -> {(o['psig'] >= .997)[s].mean():.4f}; mean dlogit {np.mean((l1 - l0)[s]):+.3f}")
