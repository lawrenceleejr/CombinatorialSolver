"""
Better mass variables from the same network outputs.

The peak is broadened by wrong pairings: with the TRUE pairing m_avg has a
~7% core half-width, with the chosen pairing 16-20%.  Try:

  argmax  m_avg of the most probable pairing (baseline)
  mode    posterior mode of m_avg over all 70 pairings, weighted by the
          signal-scorer probabilities (near-miss pairings share 5 of 6 jets
          and cluster near the right mass)
  reg     learned estimator of M/HT from scale-free features (pairing masses,
          probabilities, event shapes); M_reg = HT * estimate.  Trained on
          training-split signal of all masses, so it carries no absolute-mass prior.

Each is scored with the gate as the event score (search_study.fom_table).

  python scripts/mass_estimators.py --model runs/gated_n7/model.pt --out results/mass_estimators
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from search_study import fom_table  # noqa: E402
from src.features import event_shape_features, four_vectors, inv_mass, pairing_features, stack  # noqa: E402
from src.pairdata import concat, load_events  # noqa: E402
from src.pairnet import enumerate_interpretations, load_checkpoint  # noqa: E402

I7 = enumerate_interpretations(7, ((3, 3),))
G1 = np.array([a for a, _ in I7])
G2 = np.array([b for _, b in I7])


@torch.no_grad()
def net_probs(net, d, bs=2048):
    n = net.n_jets
    t = {k: torch.from_numpy(np.ascontiguousarray(d[k][:, :n])) for k in ("pt", "eta", "phi", "m", "mask")}
    t = {k: (v.float() if v.dtype != torch.bool else v) for k, v in t.items()}
    P, gate = [], []
    for i in range(0, len(t["pt"]), bs):
        o = net(*(t[k][i: i + bs] for k in ("pt", "eta", "phi", "m", "mask")))
        P.append(o["assign_logits"].softmax(1))
        gate.append(torch.sigmoid(o["gate_logit"]))
    return torch.cat(P).numpy(), torch.cat(gate).numpy()


def interp_masses(d):
    p = four_vectors(d)[:, :7]
    m1 = inv_mass(p[:, G1].sum(2))
    m2 = inv_mass(p[:, G2].sum(2))
    return 0.5 * (m1 + m2), np.abs(m1 - m2) / np.maximum(m1 + m2, 1e-9)


def posterior_mode(P, ma, rel_sigma=0.07, chunk=4000):
    out = np.empty(len(P))
    lm = np.log(np.maximum(ma, 1e-3))
    for i in range(0, len(P), chunk):
        l = lm[i: i + chunk]
        K = np.exp(-0.5 * ((l[:, :, None] - l[:, None, :]) / rel_sigma) ** 2)  # (n, A, A)
        dens = np.einsum("nab,nb->na", K, P[i: i + chunk])
        out[i: i + chunk] = ma[i: i + chunk][np.arange(len(l)), dens.argmax(1)]
    return out


def reg_features(d, P, ma, asy, gate):
    ht = (d["pt"] * d["mask"]).sum(1)
    order = np.argsort(-P, 1)[:, :3]
    ar = np.arange(len(P))[:, None]
    top_m = ma[ar, order] / ht[:, None]
    top_p = P[ar, order]
    top_a = asy[ar, order]
    best = order[:, 0]
    pf = pairing_features(d, G1[best], G2[best])
    ef = event_shape_features(d)
    extra = {"gate": gate, "mode_over_ht": posterior_mode(P, ma) / ht,
             "pmean_over_ht": (P * ma).sum(1) / ht}
    for k in range(3):
        extra[f"top{k}_m"], extra[f"top{k}_p"], extra[f"top{k}_asym"] = top_m[:, k], top_p[:, k], top_a[:, k]
    X, keys = stack([extra, pf, ef])
    return X, keys, ht


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/gated_n7/model.pt")
    ap.add_argument("--train-sig", default="data/pythia/sig_*_s1.h5")
    ap.add_argument("--sig", default="data/pythia/sig_*_s2.h5")
    ap.add_argument("--bkg", nargs="+", default=None)
    ap.add_argument("--out", default="results/mass_estimators")
    a = ap.parse_args()
    from sklearn.ensemble import HistGradientBoostingRegressor
    torch.set_num_threads(4)
    net, _ = load_checkpoint(a.model)
    bkg_files = a.bkg or sorted(set(glob.glob("data/pythia/qcd_s8.h5") + glob.glob("data/pythia/qcd_s9.h5")
                                    + glob.glob("data/pythia/qcd_s1[0-9].h5")))

    # --- train the regressor on training-split signal
    tr = load_events(a.train_sig, 20, 3, True, None, min_jets=6)
    Ptr, gtr = net_probs(net, tr)
    matr, astr = interp_masses(tr)
    Xtr, keys, httr = reg_features(tr, Ptr, matr, astr, gtr)
    ytr = np.log(tr["pmass"] / httr)
    reg = HistGradientBoostingRegressor(loss="absolute_error", max_iter=500, learning_rate=0.05,
                                        max_leaf_nodes=63, early_stopping=True, random_state=0)
    reg.fit(Xtr, ytr)
    print(f"regressor: {len(ytr)} training events, {reg.n_iter_} iterations", flush=True)
    os.makedirs(a.out, exist_ok=True)
    import pickle
    pickle.dump({"model": reg, "pairnet": a.model}, open(os.path.join(a.out, "reg.pkl"), "wb"))

    # --- evaluate
    d = concat(load_events(a.sig, 20, 3, True, None, min_jets=6),
               load_events(bkg_files, 20, 3, False, None, min_jets=6))
    P, gate = net_probs(net, d)
    ma, asy = interp_masses(d)
    X, _, ht = reg_features(d, P, ma, asy, gate)
    best = P.argmax(1)
    variables = {
        "argmax": ma[np.arange(len(P)), best],
        "mode": posterior_mode(P, ma),
        "reg": ht * np.exp(reg.predict(X)),
    }
    s, pm, w = d["is_signal"], d["pmass"], d["w"]
    sys.path.insert(0, HERE)
    from mass_variables_fom import shortest
    res = {}
    for name, m in variables.items():
        r = fom_table(gate, m, s, pm, w, ht)
        widths = {}
        for M in np.unique(pm[s]):
            lo, hi = shortest(m[s & (pm == M)] / M, 0.68)
            widths[int(M)] = {"centre": float((lo + hi) / 2), "half_width": float((hi - lo) / (hi + lo))}
        res[name] = {"fom": r, "core68": widths}
        print(f"{name:8s} FOM " + " ".join(f"{M}:{r[M]['fom']:6.1f}" for M in sorted(r)) +
              " | core68 half-width " + " ".join(f"{M}:{widths[M]['half_width']:.3f}" for M in sorted(widths)),
              flush=True)
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(a.out, "per_event.npz"), is_signal=s, pmass=pm, w=w, ht=ht, gate=gate,
                        **variables)


if __name__ == "__main__":
    main()
