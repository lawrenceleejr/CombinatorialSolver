"""
Why is exact pairing efficiency low on reconstructed jets?

For events whose true 3+3 pairing exists among the leading 7 jets, compares
  - the network's signal-scorer choice (and the gated mixture's choice),
  - the rank of the truth in the network's ordering (top-k),
  - a MASS ORACLE that knows the gluino mass M and picks the pairing with
    |m1 - M| + |m2 - M| smallest,
split by whether the event has exactly 6 jets or a 7th jet.  If the oracle
cannot find the truth either, the ambiguity is physical, not the network's.

  python scripts/pairing_ambiguity.py --model runs/gated_n7/model.pt --masses 500 1000 2000
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features import four_vectors, inv_mass  # noqa: E402
from src.pairdata import load_events  # noqa: E402
from src.pairnet import enumerate_interpretations, load_checkpoint  # noqa: E402

I7 = enumerate_interpretations(7, ((3, 3),))
G1 = np.array([a for a, _ in I7])
G2 = np.array([b for _, b in I7])


@torch.no_grad()
def scores(net, d, bs=2000):
    t = {k: torch.from_numpy(np.ascontiguousarray(d[k])) for k in ("pt", "eta", "phi", "m", "mask")}
    t = {k: (v.float() if v.dtype != torch.bool else v) for k, v in t.items()}
    P, L = [], []
    for i in range(0, len(t["pt"]), bs):
        o = net(*(t[k][i: i + bs] for k in ("pt", "eta", "phi", "m", "mask")))
        P.append(o["assign_logits"].softmax(1))
        L.append(o["logits"])
    return torch.cat(P).numpy(), torch.cat(L).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/gated_n7/model.pt")
    ap.add_argument("--pattern", default="data/pythia/sig_m{M}_s2.h5")
    ap.add_argument("--masses", nargs="+", type=int, default=[500, 1000, 2000])
    a = ap.parse_args()
    net, _ = load_checkpoint(a.model)
    for M in a.masses:
        d = load_events(a.pattern.format(M=M), 7, 3, True, None, min_jets=6)
        P, L = scores(net, d)
        t = d["target"]
        rec = t >= 0
        nj = d["mask"].sum(1)
        sb, mb = P.argmax(1), L.argmax(1)
        p4 = four_vectors(d)
        m1, m2 = inv_mass(p4[:, G1].sum(2)), inv_mass(p4[:, G2].sum(2))
        valid = d["mask"][:, G1].all(-1) & d["mask"][:, G2].all(-1)
        oracle = np.where(valid, np.abs(m1 - M) + np.abs(m2 - M), np.inf).argmin(1)
        rank = (P > P[np.arange(len(t)), np.maximum(t, 0)][:, None]).sum(1)
        print(f"M = {M} GeV: true pairing in the leading 7 jets for {rec.mean():.1%} of events")
        for lab, sel in (("all available", rec), ("exactly 6 jets", rec & (nj == 6)), ("with a 7th jet", rec & (nj >= 7))):
            print(f"  {lab:15s} network {np.mean(sb[sel] == t[sel]):.3f}  mixture {np.mean(mb[sel] == t[sel]):.3f}"
                  f"  mass oracle {np.mean(oracle[sel] == t[sel]):.3f}   (N={sel.sum()})")
        print("  truth in network top-k: " + "  ".join(f"top{k} {np.mean(rank[rec] < k):.2f}" for k in (1, 2, 3, 5, 10)))
        print(f"  of ALL signal: network {np.mean(sb == t):.3f}, mixture {np.mean(mb == t):.3f}")


if __name__ == "__main__":
    main()
