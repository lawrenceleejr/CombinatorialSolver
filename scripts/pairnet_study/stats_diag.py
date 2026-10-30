"""Which MC runs out first: signal or QCD? PairNet pairing, A < 0.1, m_avg window, P(signal) threshold scan on the full
sample (no half split, no statistics floor). At each threshold: signal MC events kept, raw QCD MC events kept, effective
QCD MC statistics (sum w)^2 / sum w^2, and the expected limit (300 fb^-1, f = 0 and 5%). Reports the unconstrained
optimum and the MC content there; optionally with an extra H_T > c M cut."""
import json, os, sys
import numpy as np

big, ht_npy, big_ht, xs_json = sys.argv[1:5]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qcdmerge  # noqa: E402
from limits import s_up  # noqa: E402

L = 300.0
d, _ = qcdmerge.load(big, ht_npy, big_ht, ["pn_m1", "pn_m2", "psig"])
qm = 0.5 * (d["pn_m1"] + d["pn_m2"]); qA = np.abs(d["pn_m1"] - d["pn_m2"]) / np.maximum(d["pn_m1"] + d["pn_m2"], 1e-9)
v = np.load(big); s = v["is_signal"]
sm = 0.5 * (v["pn_m1"][s] + v["pn_m2"][s]); sA = np.abs(v["pn_m1"][s] - v["pn_m2"][s]) / (v["pn_m1"][s] + v["pn_m2"][s])
sp, spm, sht = v["psig"][s], v["pmass"][s], np.load(ht_npy)[:s.sum()]
filt = {int(k): x for k, x in json.load(open(xs_json))["filter_eff"].items()}
for M in (500, 800, 1000, 1500, 2000, 2400):
    for htc in (0.0, 2.3):
        sw = (spm == M) & (sA < 0.1) & (sm > 0.85 * M) & (sm < 1.1 * M) & (sht > htc * M)
        qw = (qA < 0.1) & (qm > 0.85 * M) & (qm < 1.1 * M) & (d["ht"] > htc * M)
        nsig = int((spm == M).sum())
        for f in (0.0, 0.05):
            best = None
            for eff in np.concatenate([np.linspace(0.96, 0.2, 20), np.geomspace(0.15, 0.003, 25)]):
                t = np.quantile(sp[sw], 1 - eff) if sw.sum() else 1.0
                ks = sw & (sp >= t); kq = qw & (d["psig"] >= t)
                nS = int(ks.sum()); nB = int(kq.sum()); w = d["wfb"][kq]
                neff = w.sum() ** 2 / max((w ** 2).sum(), 1e-300)
                eS = nS / nsig * filt[M]; B = w.sum() * L
                lim = s_up(B, f) / (L * eS) if eS > 0 else np.inf
                if best is None or lim < best[0]:
                    best = (lim, nS, nB, neff, B, eff)
            lim, nS, nB, neff, B, eff = best
            print(f"M={M:4d} HT>{htc:.1f}M f={f:<4} optimum: limit {lim:8.3g} fb | signal MC {nS:5d} (stat {1/np.sqrt(max(nS,1)):.0%}) "
                  f"| QCD MC raw {nB:6d}, effective {neff:7.1f} (stat {1/np.sqrt(max(neff,1e-9)):.0%}) | B = {B:9.3g} events "
                  f"| signal kept {eff:.3f} of the window", flush=True)
