"""
Why are signal events not fully truth-matched?  Per-mass breakdown.

For each signal file, every event that loses at least one decay quark is
assigned its first failure reason (priority: forward, merged, soft, other):
  forward : a quark outside the jet |eta| acceptance
  merged  : a quark within dR<0.4 of another decay quark (shares its jet)
  soft    : a quark below the jet pT threshold after its own FSR
  other   : none of the above
Also reports the extra loss from keeping only the leading N jets.

  python scripts/matching_breakdown.py "data/pythia/sig_*.h5" --out results/matching.json
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import h5py
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pairdata import load_events  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern")
    ap.add_argument("--etamax", type=float, default=3.0)
    ap.add_argument("--ptmin", type=float, default=30.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    by_mass = defaultdict(list)
    for fp in sorted(glob.glob(args.pattern)):
        with h5py.File(fp) as f:
            by_mass[float(f.attrs["mass"])].append(fp)

    res = {}
    for m, files in sorted(by_mass.items()):
        q, tg = [], []
        for fp in files:
            with h5py.File(fp) as f:
                q.append(f["truth/quarks"][:])
                tg.append(f["truth/jet_index"][:])
        q, tg = np.concatenate(q), np.concatenate(tg)
        qpt, qeta, qphi = q[..., 0], q[..., 1], q[..., 2]
        lost = tg < 0
        dphi = (qphi[:, :, None] - qphi[:, None] + np.pi) % (2 * np.pi) - np.pi
        dr = np.sqrt((qeta[:, :, None] - qeta[:, None]) ** 2 + dphi**2)
        dr[:, range(6), range(6)] = 9
        fwd = lost & (np.abs(qeta) > args.etamax)
        mer = lost & ~fwd & (dr.min(2) < 0.4)
        soft = lost & ~fwd & ~mer & (qpt < args.ptmin)
        oth = lost & ~fwd & ~mer & ~soft
        cat = np.select([fwd.any(1), mer.any(1), soft.any(1), oth.any(1)], [1, 2, 3, 4], 0)
        close = dr < 0.4
        same = close.copy()
        same[:, :3, 3:] = False
        same[:, 3:, :3] = False
        cross = close & ~same
        r = {
            "close_pair_any": float(close.any((1, 2)).mean()),
            "close_pair_same_parent_only": float((same.any((1, 2)) & ~cross.any((1, 2))).mean()),
            "close_pair_cross_parent": float(cross.any((1, 2)).mean()),
            "n_events": int(len(q)),
            "all_matched": float((cat == 0).mean()),
            "first_reason_forward": float((cat == 1).mean()),
            "first_reason_merged": float((cat == 2).mean()),
            "first_reason_soft": float((cat == 3).mean()),
            "first_reason_other": float((cat == 4).mean()),
        }
        for n in (6, 7, 8, 10):
            d = load_events(files, n_jets=n)
            r[f"reconstructable_top{n}"] = float((d["target"] >= 0).mean())
        res[int(m)] = r
        print(int(m), json.dumps({k: round(v, 3) for k, v in r.items()}))
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
