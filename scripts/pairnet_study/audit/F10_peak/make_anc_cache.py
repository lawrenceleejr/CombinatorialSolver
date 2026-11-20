"""Convert the JetAssign (ancestry-labelled) caches to train_pairnet.py's array format, with the pairing
target taken from the parton ancestry: among the 70 3+3 pairings of the leading 7 jets, the one that puts the
most jet pT with its true parent and the least with the wrong parent or with neither (both label orders tried).
Every signal event with a labelled jet gets an exact target (no quark-matching proxy)."""
import sys
import numpy as np
sys.path.insert(0, "/scratch/llee/pairnet/audit/F5_train/wt")
from src.pairnet import enumerate_interpretations

NA = 7
interp = enumerate_interpretations(NA, ((3, 3),))
M1 = np.zeros((len(interp), NA)); M2 = np.zeros((len(interp), NA))
for i, (a, b) in enumerate(interp):
    M1[i, list(a)] = 1; M2[i, list(b)] = 1


def targets(d):
    lab = d["labels"][:, :NA].astype(np.float64)  # (N, 7, 3): parent 1 / parent 2 / other
    pt = (d["pt"][:, :NA] * d["mask"][:, :NA]).astype(np.float64)
    good1 = pt * (lab[..., 0] - lab[..., 1] - lab[..., 2])
    good2 = pt * (lab[..., 1] - lab[..., 0] - lab[..., 2])
    s = np.maximum(good1 @ M1.T + good2 @ M2.T, good2 @ M1.T + good1 @ M2.T)  # (N, 70)
    valid = d["mask"][:, :NA].astype(float) @ (M1 + M2).T > 5.5
    s = np.where(valid, s, -np.inf)
    t = s.argmax(1)
    has = d["is_signal"] & (lab.sum(-1) > 0.5).any(1) & np.isfinite(s.max(1))
    return np.where(has, t, -1).astype(np.int64)


pairs = [a.split("=") for a in sys.argv[1:]]
for src, dst in pairs:
    d = dict(np.load(src))
    d["target"] = targets(d)
    out = {k: d[k] for k in ("pt", "eta", "phi", "m", "mask", "target", "pmass", "w", "njall", "is_signal")}
    np.savez(dst, **out)
    s = d["is_signal"]
    print(dst, len(s), "signal", int(s.sum()), "with target", float((out["target"][s] >= 0).mean()), flush=True)
