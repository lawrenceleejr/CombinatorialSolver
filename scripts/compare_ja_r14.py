"""
Head-to-head: JetAssign vs PairNet r14A on identical test events (same signal and QCD arrays).

For each signal mass M and each model: events with A < 0.1 and m_avg in the window W(M); the P(signal)
cut is scanned and, at fixed weighted QCD in the window (B), the signal efficiency is compared.
B is set by r14A's own P cuts that keep a fraction f of the r14A window QCD (f = 0.1, 0.01, 0.002).
Output: JSON table and, per mass, S_JA / S_r14A at equal B (>1 means JetAssign is better; the
counting-limit gain at fixed S/B is the same factor).

  python scripts/compare_ja_r14.py --ja runs/ja/model.pt --r14 r14A/model.pt --data cache/ja_test_*.npz
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.jetassign import load_jetassign  # noqa: E402
from src.pairnet import load_checkpoint  # noqa: E402

WINDOWS = {"wide": (0.85, 1.1), "narrow": (0.92, 1.04)}


@torch.no_grad()
def score(kind, net, d, bs=8192):
    out = {"logit": [], "mavg": [], "asym": []}
    for i in range(0, len(d["pt"]), bs):
        b = [torch.from_numpy(np.ascontiguousarray(d[k][i:i + bs])) for k in ("pt", "eta", "phi", "m", "mask")]
        b = [x.float() if x.dtype == torch.float64 else x for x in b]
        o = net(*b)
        if kind == "r14":
            best = o["logits"].argmax(1)
            ar = torch.arange(len(best))
            m1, m2 = o["m1"][ar, best], o["m2"][ar, best]
        else:
            m1, m2 = o["m1"], o["m2"]
        out["logit"].append(o["gate_logit"].float())
        out["mavg"].append((0.5 * (m1 + m2) * o["ht"]).float())
        out["asym"].append(((m1 - m2).abs() / (m1 + m2).clamp(min=1e-6)).float())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def s_at_b(sc_s, sel_s, sc_b, w_b, sel_b, B):
    """Signal fraction passing at the loosest logit cut with weighted QCD <= B."""
    lb, wb = sc_b[sel_b], w_b[sel_b]
    order = np.argsort(-lb)
    cw = np.cumsum(wb[order])
    k = np.searchsorted(cw, B, side="right")
    thr = lb[order][k] if k < len(order) else -np.inf
    return float((sel_s & (sc_s > thr)).mean()), float(cw[k - 1] if k > 0 else 0.0), int(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ja", required=True)
    ap.add_argument("--r14", required=True)
    ap.add_argument("--data", required=True, help="npz from train_jetassign.py --cache-dir (test split)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    d = dict(np.load(a.data))
    sig = d["is_signal"].astype(bool)
    w = d["w"]
    sc = {}
    for m, path in (("ja", a.ja), ("r14", a.r14)):
        cache = os.path.join(os.path.dirname(os.path.abspath(path)),
                             f"scores_{m}_{os.path.basename(a.data).replace('.npz', '')}.npz")
        if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(path):
            z = np.load(cache)
            sc[m] = {k: z[k] for k in ("logit", "mavg", "asym")}
        else:
            net = load_jetassign(path) if m == "ja" else load_checkpoint(path)[0]
            sc[m] = score(m, net, d)
            np.savez_compressed(cache, **sc[m])
    res = {}
    for M in sorted(set(np.round(d["pmass"][sig]).astype(int))):
        ms = sig & (np.abs(d["pmass"] - M) < 1)
        for wn, (lo, hi) in WINDOWS.items():
            sel = {}
            for m in sc:
                rel = sc[m]["mavg"] / M
                sel[m] = (sc[m]["asym"] < 0.1) & (rel > lo) & (rel < hi)
            # reference B: r14A window QCD kept by r14A P cuts
            b14 = (w * sel["r14"])[~sig].sum()
            row = {"n_sig": int(ms.sum()), "B_r14_window": float(b14)}
            for f in (0.1, 0.01, 0.002):
                B = f * b14
                pts = {}
                for m in sc:
                    s_, b_, n_ = s_at_b(sc[m]["logit"][ms], sel[m][ms], sc[m]["logit"][~sig], w[~sig], sel[m][~sig], B)
                    pts[m] = {"S": s_, "B": b_, "nB": n_}
                pts["ratio"] = pts["ja"]["S"] / max(pts["r14"]["S"], 1e-9)
                row[f"f{f:g}"] = pts
            row["no_P_cut"] = {m: {"S": float(sel[m][ms].mean()), "B": float((w * sel[m])[~sig].sum())} for m in sc}
            res[f"{M}_{wn}"] = row
            r = row["f0.01"]
            print(f"M={M:5d} {wn:6s}  noP S ja {row['no_P_cut']['ja']['S']:.3f} r14 {row['no_P_cut']['r14']['S']:.3f} | "
                  f"B=1% : S ja {r['ja']['S']:.4f} r14 {r['r14']['S']:.4f} ratio {r['ratio']:.2f} (nB {r['ja']['nB']},{r['r14']['nB']}) | "
                  f"B=0.2%: ratio {row['f0.002']['ratio']:.2f}", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
