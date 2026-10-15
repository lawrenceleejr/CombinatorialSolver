"""
Which decay-quark momenta define the "true" jets?

  before_fsr : the quark after recoil copies, before its own shower (Pythia
               iBotCopy; CMS GenParticle isLastCopyBeforeFSR)
  after_fsr  : traced through q -> q g to the end of its shower (iBotCopyId;
               CMS isLastCopy).  The files in data/pythia use this one.

Both are stored by scripts/generate_pythia.py (truth/quarks_before_fsr,
truth/quarks_after_fsr), e.g. in data/pythia_bf (same events as data/pythia).
For each mass and definition this prints
  - the fraction of events whose true 3+3 pairing is among the leading 7 jets,
  - the core width of m_avg / M with the true pairing,
  - exact pairing of each network and of a mass oracle (knows M, picks the
    pairing with |m1 - M| + |m2 - M| smallest), on available and on all events.

  python scripts/truth_definition.py --models gated7=runs/gated_n7/model.pt
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from pairing_ambiguity import G1, G2, scores  # noqa: E402
from src.features import four_vectors, inv_mass  # noqa: E402
from src.pairdata import load_events  # noqa: E402
from src.pairnet import load_checkpoint  # noqa: E402

KEYS = {"before FSR": "truth/quarks_before_fsr", "after FSR": "truth/quarks_after_fsr"}


def core_halfwidth(x, frac=0.68):
    x = np.sort(x)
    k = max(int(frac * len(x)), 2)
    i = int(np.argmin(x[k - 1:] - x[: len(x) - k + 1]))
    return (x[i + k - 1] - x[i]) / (x[i + k - 1] + x[i])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="data/pythia_bf/sig_m{M}_s2.h5")
    ap.add_argument("--masses", nargs="+", type=int, default=[500, 750, 1000, 1250, 1500, 2000])
    ap.add_argument("--models", nargs="+", default=["gated7=runs/gated_n7/model.pt"],
                    help="name=checkpoint of 7-jet (3+3) networks")
    ap.add_argument("--out", default="results/truth_definition.json")
    a = ap.parse_args()
    nets = {}
    for spec in a.models:
        name, path = spec.split("=", 1)
        if os.path.exists(path):
            nets[name] = load_checkpoint(path)[0]
    res = {}
    for M in a.masses:
        path = a.pattern.format(M=M)
        if not os.path.exists(path):
            continue
        choice = None
        nj = max([7] + [net.n_jets for net in nets.values()])
        for lab, key in KEYS.items():
            d = load_events(path, nj, 3, True, None, min_jets=6, n_assign=7, truth_key=key)
            if choice is None:  # network choices do not depend on the truth definition
                choice = {n: scores(net, {k: v[:, :net.n_jets] for k, v in d.items() if k in
                                          ("pt", "eta", "phi", "m", "mask")})[0].argmax(1)
                          for n, net in nets.items()}
                p4 = four_vectors(d)
                m1, m2 = inv_mass(p4[:, G1].sum(2)), inv_mass(p4[:, G2].sum(2))
                valid = d["mask"][:, G1].all(-1) & d["mask"][:, G2].all(-1)
                oracle = np.where(valid, np.abs(m1 - M) + np.abs(m2 - M), np.inf).argmin(1)
                choice["mass oracle"] = oracle
            t = d["target"]
            av = t >= 0
            ar = np.arange(len(t))
            tm = 0.5 * (m1[ar, np.maximum(t, 0)] + m2[ar, np.maximum(t, 0)]) / M
            r = {"available": float(av.mean()),
                 "truth_mavg_median": float(np.median(tm[av])),
                 "truth_mavg_halfwidth68": float(core_halfwidth(tm[av]))}
            for n, c in choice.items():
                ok = (c == t) & av
                r[n] = {"of_available": float(ok[av].mean()), "of_all": float(ok.mean())}
            res.setdefault(str(M), {})[lab] = r
            print(f"M={M:5d} {lab:10s} available {r['available']:.3f}  truth m_avg/M {r['truth_mavg_median']:.3f} "
                  f"(±{r['truth_mavg_halfwidth68']:.3f})  " +
                  "  ".join(f"{n}: {r[n]['of_available']:.3f}/{r[n]['of_all']:.3f}" for n in choice), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("(exact pairing: of available / of all events)")


if __name__ == "__main__":
    main()
