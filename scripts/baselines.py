"""
Classical and legacy-network baselines, scored against generator truth.

Every method's prediction is converted to a (g1, g2) pair of jet-index sets in
the pT-sorted jet list, then compared with the truth pair, so methods that look
at different numbers of jets (6, 7, 8, ...) are compared on the same footing:
efficiency over ALL signal events ("full sample") and over reconstructable ones.

  python scripts/baselines.py --sig "data/pythia/sig_m1000_*.h5" --bkg "data/pythia/qcd_*.h5"
  python scripts/baselines.py --legacy-ckpt checkpoints/best_model.pt ...
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pairdata import load_events  # noqa: E402
from src.pairnet import enumerate_pairings, minkowski_m2, to_cartesian  # noqa: E402

N_REF = 10  # jets kept for truth bookkeeping


def pair_masses(d, n, k=3):
    """Group masses (GeV) for every pairing of the leading n jets."""
    g1, g2 = enumerate_pairings(n, k)
    pt, eta, phi, m = (torch.from_numpy(d[x][:, :n]).float() for x in ("pt", "eta", "phi", "m"))
    mask = torch.from_numpy(d["mask"][:, :n])
    p4 = to_cartesian(pt, eta, phi, m) * mask.unsqueeze(-1)
    P1, P2 = p4[:, g1].sum(2), p4[:, g2].sum(2)
    m1 = torch.sqrt(minkowski_m2(P1).clamp(min=0))
    m2 = torch.sqrt(minkowski_m2(P2).clamp(min=0))
    valid = mask[:, g1].all(-1) & mask[:, g2].all(-1)
    return g1, g2, m1.numpy(), m2.numpy(), valid.numpy()


def classical(d, n, k=3, score="absdiff"):
    g1, g2, m1, m2, valid = pair_masses(d, n, k)
    if score == "absdiff":
        s = np.abs(m1 - m2)
    else:
        s = np.abs(m1 - m2) / (m1 + m2 + 1e-9)
    s = np.where(valid, s, np.inf)
    best = s.argmin(1)
    ar = np.arange(len(best))
    return {
        "g1": g1.numpy()[best], "g2": g2.numpy()[best],
        "mavg": 0.5 * (m1[ar, best] + m2[ar, best]),
        "asym": np.abs(m1 - m2)[ar, best] / (m1 + m2)[ar, best],
    }


def truth_sets(d, k=3):
    g1, g2 = enumerate_pairings(N_REF, k)
    t = d["target"]
    ok = t >= 0
    a = np.full((len(t), k), -1)
    b = np.full((len(t), k), -1)
    a[ok], b[ok] = g1.numpy()[t[ok]], g2.numpy()[t[ok]]
    return a, b, ok


def match(pg1, pg2, tg1, tg2, ok):
    s = lambda x: [frozenset(r.tolist()) for r in x]
    P = [frozenset([x, y]) for x, y in zip(s(pg1), s(pg2))]
    T = [frozenset([x, y]) for x, y in zip(s(tg1), s(tg2))]
    return np.array([ok[i] and P[i] == T[i] for i in range(len(P))])


def legacy_predict(ckpt_path, d, num_jets=7):
    from src.combinatorics import enumerate_assignments
    from src.model import JetAssignmentTransformer

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {})
    mc = cfg.get("model", {})
    net = JetAssignmentTransformer(
        d_model=mc.get("d_model", 256), nhead=mc.get("nhead", 8),
        num_layers=mc.get("num_layers", 6), dim_feedforward=mc.get("dim_feedforward", 1024),
        dropout=0.0, num_jets=num_jets, group_num_layers=mc.get("group_num_layers", 2),
    )
    net.load_state_dict(ck["model_state_dict"])
    net.eval()
    pt, eta, phi, m = (torch.from_numpy(d[x][:, :num_jets]).float() for x in ("pt", "eta", "phi", "m"))
    mask = torch.from_numpy(d["mask"][:, :num_jets])
    p4 = to_cartesian(pt, eta, phi, m) * mask.unsqueeze(-1)
    ht = (pt * mask).sum(1)  # legacy normalises by HT of ALL jets; close enough for 7+ jets
    ht_all = torch.from_numpy((d["pt_all_ht"] if "pt_all_ht" in d else (pt * mask).sum(1).numpy())).float()
    x = p4 / ht_all.view(-1, 1, 1)
    logits = []
    with torch.no_grad():
        for i in range(0, len(x), 512):
            logits.append(net(x[i : i + 512])["logits"])
    logits = torch.cat(logits)
    asg = enumerate_assignments(num_jets)
    best = logits.argmax(1).numpy()
    g1 = np.array([asg[b][1] for b in best])
    g2 = np.array([asg[b][2] for b in best])
    P1 = torch.from_numpy(p4.numpy()[np.arange(len(best))[:, None], g1].sum(1))
    P2 = torch.from_numpy(p4.numpy()[np.arange(len(best))[:, None], g2].sum(1))
    m1 = torch.sqrt(minkowski_m2(P1).clamp(min=0)).numpy()
    m2 = torch.sqrt(minkowski_m2(P2).clamp(min=0)).numpy()
    return {"g1": g1, "g2": g2, "mavg": 0.5 * (m1 + m2), "asym": np.abs(m1 - m2) / (m1 + m2 + 1e-9),
            "pmax": logits.softmax(1).max(1).values.numpy()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sig", nargs="+", required=True)
    ap.add_argument("--bkg", nargs="*", default=[])
    ap.add_argument("--legacy-ckpt", default=None)
    ap.add_argument("--max", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sig = load_events(args.sig, N_REF, 3, True, args.max)
    tg1, tg2, ok = truth_sets(sig)
    res = {"frac_reconstructable_top%d" % N_REF: float(ok.mean())}
    for n in (6, 7, 8):
        okn = ok & (np.maximum(tg1.max(1), tg2.max(1)) < n)
        res[f"frac_reconstructable_top{n}"] = float(okn.mean())
        c = classical(sig, n)
        hit = match(c["g1"], c["g2"], tg1, tg2, ok)
        res[f"classical_absdiff_top{n}_eff_all"] = float(hit.mean())
        res[f"classical_absdiff_top{n}_eff_rec{n}"] = float(hit[okn].mean())
    if args.legacy_ckpt:
        lp = legacy_predict(args.legacy_ckpt, sig)
        hit = match(lp["g1"], lp["g2"], tg1, tg2, ok)
        ok7 = ok & (np.maximum(tg1.max(1), tg2.max(1)) < 7)
        res["legacy_eff_all"] = float(hit.mean())
        res["legacy_eff_rec7"] = float(hit[ok7].mean())
    print(json.dumps(res, indent=1))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
