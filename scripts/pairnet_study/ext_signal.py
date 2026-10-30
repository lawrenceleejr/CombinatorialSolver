"""Append heavier test-only signal (prod_sighi/sig_m{2600,2800,3000}_c{9,19}.h5) to big_vars.npz and ht_all.npy.

Writes <out>/big_vars_ext.npz (new signal rows first, then all rows of big_vars.npz), <out>/ht_all_ext.npy in the same
order, and <out>/filt_extra.json with the 6-jet filter efficiency (written / tried) per new mass."""
import glob, json, os, sys
from collections import defaultdict
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import big_vars  # noqa: E402

big, ht_all, out = sys.argv[1:4]
files = sorted(glob.glob("/scratch/llee/pairnet/prod_sighi/sig_m*_c*.h5"))
parts, hts, eff = [], [], defaultdict(list)
for f in files:
    with h5py.File(f) as h:
        pt = h["INPUTS/Source/pt"][:]; m = h["INPUTS/Source/MASK"][:].astype(bool)
        eff[int(h.attrs["mass"])].append(len(pt) / h.attrs["n_tried"])
    _, r = big_vars.work((f, True, True))
    ht = (pt * m).sum(1)[m.sum(1) >= 6]
    assert len(ht) == len(r["w"])
    parts.append(r); hts.append(ht.astype(np.float32))
    print(f, len(ht), flush=True)
v = dict(np.load(big))
new = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
n = len(new["w"])
new["has_pn"] = np.ones(n, bool)
res = {}
for k, x in v.items():
    if k in new:
        res[k] = np.concatenate([new[k].astype(x.dtype), x])
    else:
        raise KeyError(k)
np.savez(f"{out}/big_vars_ext.npz", **res)
np.save(f"{out}/ht_all_ext.npy", np.concatenate(hts + [np.load(ht_all)]))
json.dump({M: float(np.mean(e)) for M, e in eff.items()}, open(f"{out}/filt_extra.json", "w"), indent=1)
print("new signal events", n, "filter eff", {M: round(float(np.mean(e)), 3) for M, e in eff.items()})
