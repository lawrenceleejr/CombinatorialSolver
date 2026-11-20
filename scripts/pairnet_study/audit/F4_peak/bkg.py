"""F4 audit: background cost of the grouping schemes on unfiltered QCD (standard stored jets only).
usage: bkg.py <tag> <outjson> <nproc> <masses comma> <files...>
Per scheme and mass window: weighted fraction of QCD events with (any) some grouping at A < 0.1 inside the window,
(minA) the minimum-asymmetry grouping inside the window with A < 0.1, (rand) a uniformly random symmetric grouping
inside the window; relative to the events of the sample."""
import json
import sys
from multiprocessing import Pool

import h5py
import numpy as np

import f4lib as L

SCHEMES = [("33", N, 3, 3) for N in (7, 8, 9, 10)] + [("s25", N, 2, 5) for N in (7, 8, 9, 10)] + \
          [("s24", N, 2, 4) for N in (7, 8, 10)]
FAT = [("s13", 6, 1, 3), ("s14", 6, 1, 4), ("s14", 8, 1, 4)]


def one(args):
    fn, masses = args
    with h5py.File(fn, "r") as f:
        x = np.stack([f["INPUTS/Source/" + k][:, :L.K] for k in ("pt", "eta", "phi", "mass")], -1)
        w = f["EventVars/normweight"][:] * f.attrs["sigma_mb"] / f.attrs["weight_sum"]
    valid = x[..., 0] > 0
    jc = {"valid": valid, "p4": L.p4_from_ptetaphim(x) * valid[..., None]}
    jc["tp4"], jc["lpt"], jc["lp4"] = jc["p4"], np.zeros(valid.shape + (3,)), np.zeros(valid.shape + (3, 4))
    wins = {f"{M}_{k}": (lo * M, hi * M) for M in masses for k, (lo, hi) in L.WIN.items()}
    out = {"n": len(w), "sw": float(w.sum()), "sw2": float((w**2).sum())}
    cols = {jn: j for jn, j in (("std", jc), ("R08", L.recluster(jc, 0.8, 30.0)))}
    for jn, schemes in (("std", SCHEMES), ("R08", FAT)):
        j = cols[jn]
        for tag, N, a, b in schemes:
            r = L.exhaustive(j["p4"], j["valid"], N, a, b, wins)
            sym = r["has"] & (r["minA_A"] < 0.1)
            for wn, (lo, hi) in wins.items():
                k = f"{jn}/{tag}_{N}/{wn}"
                for kind, v in (("any", r["any_" + wn].astype(float)),
                                ("minA", (sym & (r["minA_m"] > lo) & (r["minA_m"] < hi)).astype(float)),
                                ("rand", r["nwin_" + wn] / np.maximum(r["nsym"], 1))):
                    out[f"{k}/{kind}"] = [float((w * v).sum()), float((w**2 * v**2).sum()), int((v > 0).sum())]
            out[f"{jn}/{tag}_{N}/nsym"] = float((w * r["nsym"]).sum())
    return out


if __name__ == "__main__":
    tag, outjson, nproc, masses = sys.argv[1], sys.argv[2], int(sys.argv[3]), [int(m) for m in sys.argv[4].split(",")]
    files = sys.argv[5:]
    tot = {}
    with Pool(nproc) as p:
        for i, o in enumerate(p.imap_unordered(one, [(f, masses) for f in files])):
            for k, v in o.items():
                tot[k] = (np.array(tot[k]) + np.array(v)).tolist() if k in tot else v
            if i % 10 == 0:
                print(i, len(files), flush=True)
                json.dump({"tag": tag, "nfiles": i + 1, "tot": tot}, open(outjson, "w"))
    json.dump({"tag": tag, "nfiles": len(files), "tot": tot}, open(outjson, "w"), indent=1)
