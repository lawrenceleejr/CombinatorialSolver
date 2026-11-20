"""Build the merged QCD once per variant and store the events of the analysis phase space (PairNet pairing valid,
A < 0.1, H_T > 400; all weights incl. zero-weight events outside a sample's complete region) + high-mass arrays."""
import pickle, sys, numpy as np, pipe
for tag, tails in (("main", ("qcdvlmT", "qcdhtT")), ("t700", ("qcdvlmT", "qcdhtT", "qcd700T", "qcd700U"))):
    QCD, merge = pipe.build_qcd(tails)
    k = (QCD["pn_asym"] < 0.1) & QCD["pn_valid"].astype(bool) & (QCD["ht"] > 400)
    out = {key: v[k] for key, v in QCD.items() if not key.startswith("am_")}
    np.savez(f"{pipe.D}/data/qcd_sel_{tag}.npz", **out)
    pickle.dump(merge, open(f"{pipe.D}/data/merge_{tag}.pkl", "wb"))
    print(tag, "events", len(k), "kept", int(k.sum()), flush=True)
    del QCD, out
d, filt, hi = pipe.build_high()
pickle.dump(({M: pipe.high_arrays(d, filt, M) for M in range(500, 2401, 100)}, hi), open(f"{pipe.D}/data/high_arrays.pkl", "wb"))
print("high done", flush=True)
