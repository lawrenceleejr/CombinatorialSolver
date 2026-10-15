"""
Stacked event classifier: gradient-boosted trees on expert four-vector features
(event shapes, features of the chosen pairing) PLUS the PairNet outputs (gate
logit, pairing probabilities).  Must be trained on events PairNet did not see.

  python scripts/train_stack.py --model runs/gated_n7/model.pt \
      --sig "data/pythia/sig_*_s3.h5" --bkg data/pythia/qcd_s1[1-4].h5 --out runs/stack
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from src.features import event_shape_features, pairing_features, stack  # noqa: E402
from src.pairdata import load_events  # noqa: E402
from src.pairnet import enumerate_interpretations, load_checkpoint  # noqa: E402

I7 = enumerate_interpretations(7, ((3, 3),))
G1 = np.array([a for a, _ in I7])
G2 = np.array([b for _, b in I7])


@torch.no_grad()
def pairnet_feats(net, d, bs=2048):
    n = net.n_jets
    t = {k: torch.from_numpy(np.ascontiguousarray(d[k][:, :n])) for k in ("pt", "eta", "phi", "m", "mask")}
    t = {k: (v.float() if v.dtype != torch.bool else v) for k, v in t.items()}
    P, g = [], []
    for i in range(0, len(t["pt"]), bs):
        o = net(*(t[k][i: i + bs] for k in ("pt", "eta", "phi", "m", "mask")))
        P.append(o["assign_logits"].softmax(1))
        g.append(o["gate_logit"])
    return torch.cat(P).numpy(), torch.cat(g).numpy()


def stack_features(net, d):
    P, gl = pairnet_feats(net, d)
    best = P.argmax(1)
    srt = -np.sort(-P, 1)
    ent = -(P * np.log(np.maximum(P, 1e-12))).sum(1)
    nn = {"gate_logit": gl, "p1": srt[:, 0], "p2": srt[:, 1], "p3": srt[:, 2], "p_entropy": ent}
    X, keys = stack([nn, pairing_features(d, G1[best], G2[best]), event_shape_features(d)])
    return X, keys, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/gated_n7/model.pt")
    ap.add_argument("--sig", nargs="+", required=True)
    ap.add_argument("--bkg", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    torch.set_num_threads(2)
    net, _ = load_checkpoint(a.model)
    s = load_events(a.sig, 20, 3, True, None, min_jets=6)
    b = load_events(a.bkg, 20, 3, False, None, min_jets=6)
    Xs, keys, _ = stack_features(net, s)
    Xb, _, _ = stack_features(net, b)
    X = np.concatenate([Xs, Xb])
    y = np.concatenate([np.ones(len(Xs)), np.zeros(len(Xb))])
    w = np.where(y == 1, 0.5 / len(Xs), 0.5 / len(Xb)) * len(y)
    clf = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, early_stopping=True,
                                         validation_fraction=0.2, random_state=0)
    clf.fit(X, y, sample_weight=w)
    os.makedirs(a.out, exist_ok=True)
    pickle.dump({"model": clf, "keys": keys, "pairnet": a.model}, open(os.path.join(a.out, "stack.pkl"), "wb"))
    json.dump({"n_sig": len(Xs), "n_bkg": len(Xb), "n_iter": clf.n_iter_, "keys": keys},
              open(os.path.join(a.out, "train.json"), "w"), indent=1)
    print(f"stacked classifier: {len(Xs)} signal, {len(Xb)} QCD, {len(keys)} features, {clf.n_iter_} iterations")


if __name__ == "__main__":
    main()
