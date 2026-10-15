"""
Conventional baseline: gradient-boosted trees on expert four-vector features
(src/features.py), with the classical argmin |m1 - m2| pairing over the leading
7 jets.  Event-level signal vs QCD only; the pairing is not learned.

  python scripts/train_bdt.py --sig "data/pythia/sig_*_s1.h5" --bkg data/pythia/qcd_s[1-7].h5 \
      --out runs/bdt
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features import classical_pairing, event_shape_features, pairing_features, stack  # noqa: E402
from src.pairdata import load_events  # noqa: E402


def features(d, drop=()):
    g1, g2 = classical_pairing(d)
    X, keys = stack([event_shape_features(d), pairing_features(d, g1, g2)])
    keep = [i for i, k in enumerate(keys) if k not in drop]
    return X[:, keep], [keys[i] for i in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sig", nargs="+", required=True)
    ap.add_argument("--bkg", nargs="+", required=True)
    ap.add_argument("--drop", default="", help="comma list of features to exclude")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    drop = tuple(x for x in a.drop.split(",") if x)
    s = load_events(a.sig, 20, 3, True, None, min_jets=6)
    b = load_events(a.bkg, 20, 3, False, None, min_jets=6)
    Xs, keys = features(s, drop)
    Xb, _ = features(b, drop)
    X = np.concatenate([Xs, Xb])
    y = np.concatenate([np.ones(len(Xs)), np.zeros(len(Xb))])
    w = np.where(y == 1, 0.5 / len(Xs), 0.5 / len(Xb)) * len(y)  # balanced classes
    clf = HistGradientBoostingClassifier(max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
                                         l2_regularization=1.0, early_stopping=True,
                                         validation_fraction=0.15, random_state=0)
    clf.fit(X, y, sample_weight=w)
    auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])
    os.makedirs(a.out, exist_ok=True)
    pickle.dump({"model": clf, "keys": keys, "drop": drop}, open(os.path.join(a.out, "bdt.pkl"), "wb"))
    json.dump({"train_auc": auc, "n_iter": clf.n_iter_, "keys": keys},
              open(os.path.join(a.out, "train.json"), "w"), indent=1)
    print(f"trained: {len(Xs)} signal, {len(Xb)} QCD, {len(keys)} features, "
          f"{clf.n_iter_} iterations, train AUC {auc:.4f}")


if __name__ == "__main__":
    main()
