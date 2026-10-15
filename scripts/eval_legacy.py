"""
Evaluate a legacy JetAssignmentTransformer checkpoint (or the classical solver)
on the testbed, scored against generator truth, in the same format as
train_pairnet.py's eval.npz.

  python scripts/eval_legacy.py --ckpt checkpoints/best_model.pt \
      --sig "data/pythia/sig_*_s2.h5" --bkg data/pythia/qcd_s8.h5 --out runs/legacy/eval.npz
  python scripts/eval_legacy.py --classical --sig ... --out runs/classical/eval.npz

Event-level discriminant for methods without a background class: the mass
asymmetry of the chosen interpretation (signal-like = small), which is how the
legacy design intends QCD to be separated.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.combinatorics import enumerate_assignments  # noqa: E402
from src.pairdata import concat, load_events  # noqa: E402
from src.pairnet import enumerate_interpretations, minkowski_m2, to_cartesian  # noqa: E402

N_REF = 10


def truth_sets(d):
    interp = enumerate_interpretations(N_REF, ((3, 3),))
    return [frozenset([frozenset(interp[t][0]), frozenset(interp[t][1])]) if t >= 0 else None
            for t in d["target"]]


def legacy_logits(ckpt, d, num_jets=7):
    from src.model import JetAssignmentTransformer

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    mc = ck["config"]["model"]
    net = JetAssignmentTransformer(
        d_model=mc["d_model"], nhead=mc["nhead"], num_layers=mc["num_layers"],
        dim_feedforward=mc["dim_feedforward"], dropout=0.0, num_jets=num_jets,
        group_num_layers=mc.get("group_num_layers", 1),
    )
    net.load_state_dict(ck["model_state_dict"])
    net.eval()
    pt, eta, phi, m = (torch.from_numpy(d[x]).float() for x in ("pt", "eta", "phi", "m"))
    mask = torch.from_numpy(d["mask"])
    p4 = to_cartesian(pt, eta, phi, m) * mask.unsqueeze(-1)
    ht = (pt * mask).sum(1).clamp(min=1e-6)  # legacy: HT of all jets
    x = (p4[:, :num_jets] / ht.view(-1, 1, 1)).contiguous()
    out = []
    with torch.no_grad():
        torch.set_num_threads(4)
        for i in range(0, len(x), 512):
            out.append(net(x[i : i + 512])["logits"])
    return torch.cat(out), p4[:, :num_jets]


def classical_logits(d, num_jets=7):
    pt, eta, phi, m = (torch.from_numpy(d[x]).float() for x in ("pt", "eta", "phi", "m"))
    mask = torch.from_numpy(d["mask"])
    p4 = (to_cartesian(pt, eta, phi, m) * mask.unsqueeze(-1))[:, :num_jets]
    asg = enumerate_assignments(num_jets)
    g1 = torch.tensor([a[1] for a in asg])
    g2 = torch.tensor([a[2] for a in asg])
    m1 = torch.sqrt(minkowski_m2(p4[:, g1].sum(2)).clamp(min=0))
    m2 = torch.sqrt(minkowski_m2(p4[:, g2].sum(2)).clamp(min=0))
    valid = mask[:, :num_jets][:, g1].all(-1) & mask[:, :num_jets][:, g2].all(-1)
    return torch.where(valid, -(m1 - m2).abs(), -1e9), p4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--classical", action="store_true")
    ap.add_argument("--num-jets", type=int, default=7)
    ap.add_argument("--sig", nargs="+", required=True)
    ap.add_argument("--bkg", nargs="*", default=[])
    ap.add_argument("--min-jets", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    parts = [load_events(args.sig, 20, 3, True, None, min_jets=args.min_jets)]
    if args.bkg:
        parts.append(load_events(args.bkg, 20, 3, False, None, min_jets=args.min_jets))
    d = concat(*parts) if len(parts) > 1 else parts[0]
    # truth with the common N_REF bookkeeping
    tparts = [load_events(args.sig, N_REF, 3, True, None, min_jets=args.min_jets)]
    if args.bkg:
        tparts.append(load_events(args.bkg, N_REF, 3, False, None, min_jets=args.min_jets))
    dt = concat(*tparts) if len(tparts) > 1 else tparts[0]
    T = truth_sets(dt)

    if args.classical:
        logits, p4 = classical_logits(d, args.num_jets)
    else:
        logits, p4 = legacy_logits(args.ckpt, d, args.num_jets)
    asg = enumerate_assignments(args.num_jets)
    best = logits.argmax(1).numpy()
    g1 = np.array([asg[b][1] for b in best])
    g2 = np.array([asg[b][2] for b in best])
    ar = np.arange(len(best))[:, None]
    P = p4.numpy()
    m1 = np.sqrt(np.maximum(minkowski_m2(torch.from_numpy(P[ar, g1].sum(1))).numpy(), 0))
    m2 = np.sqrt(np.maximum(minkowski_m2(torch.from_numpy(P[ar, g2].sum(1))).numpy(), 0))
    pred = [frozenset([frozenset(a.tolist()), frozenset(b.tolist())]) for a, b in zip(g1, g2)]
    correct = np.array([t is not None and p == t for p, t in zip(pred, T)])
    asym = np.abs(m1 - m2) / (m1 + m2 + 1e-9)
    ev = {
        "correct": correct, "target": dt["target"], "is_signal": d["is_signal"],
        "pmass": d["pmass"], "w": d["w"], "mavg": 0.5 * (m1 + m2), "asym": asym,
        "psig": 1 - asym,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, **ev)
    s = d["is_signal"]
    rec = s & (dt["target"] >= 0)
    r = {"eff_on_all_signal": float(correct[s].mean()), "eff_on_reconstructable10": float(correct[rec].mean())}
    for mm in np.unique(d["pmass"][s]):
        sel = s & (d["pmass"] == mm)
        r[f"eff_all_m{int(mm)}"] = float(correct[sel].mean())
    print(json.dumps(r, indent=1))
    json.dump(r, open(args.out.replace(".npz", ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
