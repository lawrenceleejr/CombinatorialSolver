"""Build validation/test arrays for scoring the main-branch network with the same
metrics as train_pairnet.py: leading-7-jet four-vectors (GeV) + HT of all jets
(the legacy normalisation), 7-jet truth targets, and the map from the legacy
assignment order to PairNet's interpretation order."""
import glob, sys
import numpy as np
import torch

repo, prod, out = sys.argv[1:4]
sys.path.insert(0, repo)
from src.combinatorics import enumerate_assignments  # noqa: E402
from src.pairdata import concat, load_events  # noqa: E402
from src.pairnet import enumerate_interpretations, to_cartesian  # noqa: E402

asg = enumerate_assignments(7)
interp = enumerate_interpretations(7, ((3, 3),))
key = lambda a, b: frozenset([frozenset(a), frozenset(b)])
pos = {key(g1, g2): j for j, (g1, g2) in enumerate(interp[: len(interp)])}
legacy_to_pairnet = np.array([pos[key(a[1], a[2])] for a in asg])
assert len(set(legacy_to_pairnet)) == len(asg) == 70

res = {"legacy_to_pairnet": legacy_to_pairnet}
for split, sig, bkg in (("val", "sig_*_c8.h5", "qcd_c*8.h5"), ("test", "sig_*_c9.h5", "qcd_c*9.h5")):
    fs, fb = sorted(glob.glob(f"{prod}/{sig}")), sorted(glob.glob(f"{prod}/{bkg}"))
    full = concat(load_events(fs, 20, 3, True, None, min_jets=6), load_events(fb, 20, 3, False, None, min_jets=6))
    tru = concat(load_events(fs, 7, 3, True, None, min_jets=6), load_events(fb, 7, 3, False, None, min_jets=6))
    assert len(full["pt"]) == len(tru["pt"]) and np.allclose(full["pt"][:, :7], tru["pt"][:, :7])
    pt, eta, phi, m = (torch.from_numpy(full[k]).float() for k in ("pt", "eta", "phi", "m"))
    mask = torch.from_numpy(full["mask"])
    p4 = (to_cartesian(pt, eta, phi, m) * mask.unsqueeze(-1))[:, :7]
    res[f"{split}/p4"] = p4.numpy().astype(np.float32)
    res[f"{split}/ht"] = (pt * mask).sum(1).numpy().astype(np.float32)
    for k in ("target", "is_signal", "pmass", "w"):
        res[f"{split}/{k}"] = tru[k]
    print(split, len(pt), "events,", int(tru["is_signal"].sum()), "signal,", int((~tru["is_signal"]).sum()), "QCD", flush=True)
np.savez_compressed(out, **res)
