"""
Validate an exported PairNet ONNX against PyTorch and against physics numbers.

  python scripts/validate_onnx.py --onnx models/pairnet_gated_7jet.onnx --model runs/gated_n7/model.pt

Checks
  1. ONNX Runtime vs PyTorch outputs (max |diff|, argmax agreement)
  2. legacy ordering: argmax index -> enumerate_assignments(A) -> same jets as PairNet's choice
     (A = the jets the network pairs: 7, or 6 for 6-jet models)
  3. scale invariance: raw GeV vs HT-normalised input give the same logits and p_signal
  4. physics: exact pairing at 1 TeV and P(signal) AUC from the ONNX outputs alone
"""

import argparse
import os
import sys

import numpy as np
import onnxruntime as ort
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from export_pairnet_onnx import PairNetONNX  # noqa: E402
from src.combinatorics import enumerate_assignments  # noqa: E402
from src.features import four_vectors  # noqa: E402
from src.pairdata import concat, load_events  # noqa: E402
from src.pairnet import enumerate_interpretations, load_checkpoint  # noqa: E402


def run_onnx(sess, x, bs=1000):
    names = [o.name for o in sess.get_outputs()]
    parts = [sess.run(None, {"four_momenta": x[i: i + bs]}) for i in range(0, len(x), bs)]
    return {n: np.concatenate([p[k] for p in parts]) for k, n in enumerate(names)}


@torch.no_grad()
def run_torch(wrap, x, bs=1000):
    parts = [[t.numpy() for t in wrap(torch.from_numpy(x[i: i + bs]))] for i in range(0, len(x), bs)]
    return [np.concatenate([p[k] for p in parts]) for k in range(len(parts[0]))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=4000)
    a = ap.parse_args()
    net, _ = load_checkpoint(a.model)
    J = net.n_jets
    A = net.n_assign_jets
    sig = load_events("data/pythia/sig_m1000_s2.h5", J, 3, True, None, min_jets=6, n_assign=A)
    bkg = load_events("data/pythia/qcd_s8.h5", J, 3, False, None, min_jets=6, n_assign=A)
    d = concat(sig, bkg)
    x = four_vectors(d).astype(np.float32)  # (N, J, 4) E, px, py, pz in GeV, zero-padded
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    on = run_onnx(sess, x)

    wrap = PairNetONNX(net).eval()
    tout = run_torch(wrap, x)
    tl, tp, ts, tm = tout[:4]
    fin = tl > -1e3
    print("1. ONNX vs PyTorch: max|d logits| %.2e  max|d p_signal| %.2e  max|d signal_logits| %.2e  "
          "max rel|d masses| %.2e" % (np.abs(on["logits"] - tl)[fin].max(), np.abs(on["p_signal"] - tp).max(),
                                     np.abs(on["signal_logits"] - ts)[fin].max(),
                                     (np.abs(on["masses"] - tm) / np.maximum(tm, 1)).max()))
    if "m_est" in on:
        print("   max rel|d m_est| %.2e" % (np.abs(on["m_est"] - tout[4]) / np.maximum(tout[4], 1)).max())
    print("   argmax agreement: logits %.4f, signal_logits %.4f" % (
        np.mean(on["logits"].argmax(1) == tl.argmax(1)), np.mean(on["signal_logits"].argmax(1) == ts.argmax(1))))

    # 2. legacy ordering decodes to the same jets as PairNet's own table
    asg = enumerate_assignments(A)
    ours = enumerate_interpretations(A, ((3, 3),))
    own = []
    with torch.no_grad():
        t = {k: torch.from_numpy(np.ascontiguousarray(d[k])) for k in ("pt", "eta", "phi", "m", "mask")}
        for i in range(0, len(x), 1000):
            o = net(*(t[k][i: i + 1000].float() if k != "mask" else t[k][i: i + 1000]
                      for k in ("pt", "eta", "phi", "m", "mask")))
            own.append(o["assign_logits"].argmax(1).numpy())
    own = np.concatenate(own)
    leg = on["signal_logits"].argmax(1)
    same = [frozenset([frozenset(asg[i][1]), frozenset(asg[i][2])]) ==
            frozenset([frozenset(ours[j][0]), frozenset(ours[j][1])]) for i, j in zip(leg, own)]
    print(f"2. legacy-order index decodes to PairNet's chosen jets: {np.mean(same):.4f}")

    # 3. scale invariance
    ht = (d["pt"] * d["mask"]).sum(1)[:, None, None].astype(np.float32)
    on2 = run_onnx(sess, x / ht)
    print("3. HT-normalised input: max|d p_signal| %.2e, argmax agreement %.4f, masses scale by HT: %s" % (
        np.abs(on2["p_signal"] - on["p_signal"]).max(), np.mean(on2["logits"].argmax(1) == on["logits"].argmax(1)),
        np.allclose(on2["masses"] * ht[:, :, 0][..., None], on["masses"], rtol=1e-3, atol=1e-2)))

    # 4. physics from ONNX outputs alone
    s = d["is_signal"]
    tgt = d["target"]
    rec = s & (tgt >= 0)
    pick = on["signal_logits"].argmax(1)
    exact = np.array([tgt[i] >= 0 and frozenset([frozenset(asg[pick[i]][1]), frozenset(asg[pick[i]][2])]) ==
                      frozenset([frozenset(ours[tgt[i]][0]), frozenset(ours[tgt[i]][1])]) for i in range(len(tgt))])
    from sklearn.metrics import roc_auc_score
    w = np.where(s, 1.0 / s.sum(), d["w"] / d["w"][~s].sum())
    print(f"4. 1 TeV signal: exact pairing {exact[rec].mean():.3f} of available, {exact[s].mean():.3f} of all; "
          f"P(signal) AUC vs QCD {roc_auc_score(s, on['p_signal'], sample_weight=w):.4f}")
    if "m_est" in on:
        def hw(r):
            r = np.sort(r)
            k = int(0.68 * len(r))
            i = int(np.argmin(r[k:] - r[:-k]))
            return (r[i + k] - r[i]) / (r[i + k] + r[i])
        best = on["masses"][np.arange(len(x)), on["signal_logits"].argmax(1)].mean(1)
        print(f"   1 TeV signal mass peak (68% relative half-width): m_est {hw(on['m_est'][s] / 1000):.3f}, "
              f"m_avg of chosen pairing {hw(best[s] / 1000):.3f}")


if __name__ == "__main__":
    main()
