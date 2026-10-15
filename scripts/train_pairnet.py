"""
Train / evaluate PairNet on signal (+ optional QCD) samples.

  python scripts/train_pairnet.py --sig "data/pythia/sig_*.h5" \
      --bkg "data/pythia/qcd_*.h5" --n-jets 8 --epochs 30 --out runs/pairnet

Writes <out>/model.pt, <out>/metrics.json and <out>/eval.npz (per-event
predictions on the test split, for plotting).
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pairdata import concat, load_events, split, to_tensors  # noqa: E402
from src.pairnet import PairNet, pairnet_loss  # noqa: E402


def batches(t, idx, bs):
    for i in range(0, len(idx), bs):
        j = idx[i : i + bs]
        yield {k: v[j] for k, v in t.items()}


@torch.no_grad()
def predict(net, t, bs=1024):
    net.eval()
    outs = {"logits": [], "m1": [], "m2": [], "ht": []}
    idx = torch.arange(len(t["pt"]))
    for b in batches(t, idx, bs):
        o = net(b["pt"], b["eta"], b["phi"], b["m"], b["mask"])
        for k in outs:
            outs[k].append(o[k])
    return {k: torch.cat(v) for k, v in outs.items()}


def evaluate(net, t, A):
    o = predict(net, t)
    logits = o["logits"]
    best = logits[:, :A].argmax(1)
    p = logits.softmax(1)
    psig = 1 - p[:, A] if logits.shape[1] == A + 1 else p[:, :A].max(1).values
    ar = torch.arange(len(best))
    m1, m2 = o["m1"][ar, best], o["m2"][ar, best]
    mavg = 0.5 * (m1 + m2) * o["ht"]
    asym = (m1 - m2).abs() / (m1 + m2)
    return {
        "best": best.numpy(),
        "psig": psig.numpy(),
        "mavg": mavg.numpy(),
        "asym": asym.numpy(),
        "target": t["target"].numpy(),
        "is_signal": t["is_signal"].numpy(),
        "pmass": t["pmass"].numpy(),
        "w": t["w"].numpy(),
    }


def summarize(ev):
    s = ev["is_signal"]
    tgt = ev["target"]
    rec = s & (tgt >= 0)
    r = {
        "n_sig": int(s.sum()),
        "n_bkg": int((~s).sum()),
        "frac_reconstructable": float(rec.sum() / max(s.sum(), 1)),
        "eff_on_reconstructable": float((ev["best"][rec] == tgt[rec]).mean()) if rec.any() else 0.0,
    }
    r["eff_on_all_signal"] = r["eff_on_reconstructable"] * r["frac_reconstructable"]
    if (~s).any() and s.any():
        from sklearn.metrics import roc_auc_score

        y = s.astype(int)
        r["auc"] = float(roc_auc_score(y, ev["psig"], sample_weight=None))
        thr = np.quantile(ev["psig"][s], 0.5)
        fb = (ev["psig"][~s] >= thr).mean()
        r["bkg_rej_at_50"] = float(1 / max(fb, 1e-6))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sig", nargs="+", required=True)
    ap.add_argument("--bkg", nargs="*", default=[])
    ap.add_argument("--n-jets", type=int, default=8)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sizes", default=None,
                    help='group-size hypotheses, e.g. "33,32" (3+3 and 3+2 for merged jets)')
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--no-pair-bias", action="store_true")
    ap.add_argument("--no-group-kin", action="store_true")
    ap.add_argument("--no-null", action="store_true")
    ap.add_argument("--bkg-weight", type=float, default=1.0)
    ap.add_argument("--max-sig", type=int, default=None)
    ap.add_argument("--max-bkg", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sizes = tuple((int(x[0]), int(x[1])) for x in args.sizes.split(",")) if args.sizes else None
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    os.makedirs(args.out, exist_ok=True)

    sig = load_events(args.sig, args.n_jets, args.k, True, args.max_sig, sizes=sizes)
    parts = [sig]
    if args.bkg:
        bkg = load_events(args.bkg, args.n_jets, args.k, False, args.max_bkg, sizes=sizes)
        parts.append(bkg)
    data = concat(*parts) if len(parts) > 1 else sig
    tr, va, te = split(data, seed=args.seed)
    ttr, tva, tte = map(to_tensors, (tr, va, te))
    print(f"train {len(tr['pt'])}  val {len(va['pt'])}  test {len(te['pt'])}; "
          f"sig reconstructable {np.mean(sig['target'] >= 0):.3f}", flush=True)

    net = PairNet(
        n_jets=args.n_jets, k=args.k, d_model=args.d_model, nhead=args.heads,
        num_layers=args.layers, dim_ff=2 * args.d_model, dropout=args.dropout,
        use_pair_bias=not args.no_pair_bias, use_group_kin=not args.no_group_kin,
        use_null=not args.no_null, sizes=sizes,
    )
    A = net.n_assign
    print(f"params {sum(p.numel() for p in net.parameters())/1e3:.0f}k  interpretations {A}")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * math.ceil(len(tr["pt"]) / args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    hist = []
    best_score, best_state = -1, None
    for ep in range(args.epochs):
        net.train()
        t0 = time.time()
        perm = torch.randperm(len(tr["pt"]))
        tot, nb = 0.0, 0
        for b in batches(ttr, perm, args.bs):
            o = net(b["pt"], b["eta"], b["phi"], b["m"], b["mask"])
            loss = pairnet_loss(o, b["target"], b["is_signal"], bkg_weight=args.bkg_weight)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            nb += 1
        ev = evaluate(net, tva, A)
        r = summarize(ev)
        r.update(epoch=ep, loss=tot / nb, time=time.time() - t0)
        hist.append(r)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
        score = r["eff_on_reconstructable"] + (r.get("auc", 0) if args.bkg else 0)
        if score > best_score:
            best_score = score
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    torch.save({"state": best_state, "args": vars(args)}, os.path.join(args.out, "model.pt"))
    ev = evaluate(net, tte, A)
    np.savez_compressed(os.path.join(args.out, "eval.npz"), **ev)
    final = summarize(ev)
    print("TEST", json.dumps(final), flush=True)
    json.dump({"test": final, "history": hist, "args": vars(args)},
              open(os.path.join(args.out, "metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
