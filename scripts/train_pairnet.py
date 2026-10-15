"""
Train / evaluate PairNet on signal (+ optional QCD) samples.  Runs on CUDA,
Apple MPS or CPU (--device auto picks the first available).

  python scripts/train_pairnet.py --gated --proxy --n-jets 7 \
      --sig "data/prod/sig_*_c[0-7].h5" --bkg "data/prod/qcd_c[0-7].h5" \
      --val-sig "data/prod/sig_*_c8.h5" --val-bkg "data/prod/qcd_c8.h5" \
      --test-sig "data/prod/sig_*_c9.h5" --test-bkg "data/prod/qcd_c9.h5" \
      --epochs 40 --bs 1024 --lr 2e-3 --amp --cache-dir cache --out runs/gated_n7

Writes <out>/model.pt (selected checkpoint), <out>/last.pt (resumable),
<out>/metrics.json and <out>/eval.npz (per-event test predictions).

Checkpoint selection (--select): "sig_at_bkg" keeps the epoch with the most
signal in the mass window at a fixed weighted QCD pass rate (--select-bkg-pass)
of the asymmetry cut on validation, so runs are compared at one operating point.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pairdata import concat, load_events, split, to_tensors  # noqa: E402
from src.pairnet import PairNet, gated_loss, pairnet_loss  # noqa: E402


def batches(t, idx, bs):
    for i in range(0, len(idx), bs):
        j = idx[i : i + bs]
        yield {k: v[j] for k, v in t.items()}


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(net, t, bs=4096, amp=False):
    net.eval()
    keep = ("logits", "m1", "m2", "ht", "gate_logit", "log_m_over_ht")
    outs = {k: [] for k in keep}
    idx = torch.arange(len(t["pt"]), device=t["pt"].device)
    for b in batches(t, idx, bs):
        with autocast(t["pt"].device, amp):
            o = net(b["pt"], b["eta"], b["phi"], b["m"], b["mask"])
        for k in keep:
            if k in o:
                outs[k].append(o[k].float().cpu())
    return {k: torch.cat(v) for k, v in outs.items() if v}


def autocast(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def evaluate(net, t, A, bs=4096, amp=False):
    o = predict(net, t, bs, amp)
    logits = o["logits"]
    best = logits[:, :A].argmax(1)
    p = logits.softmax(1)
    if "gate_logit" in o:
        psig = torch.sigmoid(o["gate_logit"])
    elif logits.shape[1] == A + 1:
        psig = 1 - p[:, A]
    else:
        psig = p[:, :A].max(1).values
    ar = torch.arange(len(best))
    m1, m2 = o["m1"][ar, best], o["m2"][ar, best]
    mavg = 0.5 * (m1 + m2) * o["ht"]
    asym = (m1 - m2).abs() / (m1 + m2)
    extra = {}
    if "log_m_over_ht" in o:
        extra["m_est"] = (torch.exp(o["log_m_over_ht"]) * o["ht"]).numpy()
    return {
        **extra,
        "best": best.numpy(),
        "psig": psig.numpy(),
        "mavg": mavg.numpy(),
        "asym": asym.numpy(),
        "target": t["target"].cpu().numpy(),
        "is_signal": t["is_signal"].cpu().numpy(),
        "pmass": t["pmass"].cpu().numpy(),
        "w": t["w"].cpu().numpy(),
    }


ASYM_CUT = 0.1  # stand-in for the analysis preselection on mass asymmetry


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
    if "m_est" in ev and s.any():
        rel = ev["m_est"][s] / np.maximum(ev["pmass"][s], 1)
        lo, hi = np.quantile(rel, [0.16, 0.84])
        r["m_est_rel_halfwidth68"] = float((hi - lo) / (hi + lo))
        relb = ev["mavg"][s] / np.maximum(ev["pmass"][s], 1)
        lo, hi = np.quantile(relb, [0.16, 0.84])
        r["mavg_rel_halfwidth68"] = float((hi - lo) / (hi + lo))
    pas = ev["asym"] < ASYM_CUT
    r["sig_pass_asym"] = float(pas[s].mean()) if s.any() else 0.0
    if s.any():
        rel = ev["mavg"][s] / np.maximum(ev["pmass"][s], 1)
        r["sig_pass_asym_in_window"] = float((pas[s] & (rel > 0.85) & (rel < 1.1)).mean())
    if (~s).any():
        w = ev["w"][~s]
        r["bkg_pass_asym"] = float((w * pas[~s]).sum() / w.sum())  # weighted
        r["bkg_mavg_median_passing"] = float(np.median(ev["mavg"][~s & pas])) if (pas & ~s).any() else 0.0
    if (~s).any() and s.any():
        from sklearn.metrics import roc_auc_score

        y = s.astype(int)
        r["auc"] = float(roc_auc_score(y, ev["psig"], sample_weight=None))
        thr = np.quantile(ev["psig"][s], 0.5)
        fb = (ev["psig"][~s] >= thr).mean()
        r["bkg_rej_at_50"] = float(1 / max(fb, 1e-6))
        r["sig_in_window_at_bkg"] = sig_at_bkg_curve(ev)
    return r


def sig_at_bkg_curve(ev, targets=(0.02, 0.05, 0.08)):
    """Signal passing the asymmetry cut with m_avg in [0.85, 1.1] M, read off at
    fixed weighted QCD pass rates by scanning the asymmetry cut."""
    s = ev["is_signal"]
    w = ev["w"][~s]
    rel = ev["mavg"] / np.maximum(ev["pmass"], 1)
    win = (rel > 0.85) & (rel < 1.1)
    cuts = np.linspace(0.002, 0.6, 300)
    q = np.array([(w * (ev["asym"][~s] < c)).sum() / w.sum() for c in cuts])
    sg = np.array([((ev["asym"][s] < c) & win[s]).mean() for c in cuts])
    return {f"{t:g}": float(np.interp(t, q, sg)) for t in targets}


def load_split(args, sizes, which):
    files_s = {"train": args.sig, "val": args.val_sig, "test": args.test_sig}[which]
    files_b = {"train": args.bkg, "val": args.val_bkg, "test": args.test_bkg}[which]
    if not files_s:
        return None
    parts = [load_events(files_s, args.n_jets, args.k, True, args.max_sig if which == "train" else None,
                         sizes=sizes, min_jets=args.min_jets, proxy=args.proxy, n_assign=args.n_assign)]
    if files_b:
        parts.append(load_events(files_b, args.n_jets, args.k, False,
                                 args.max_bkg if which == "train" else None,
                                 sizes=sizes, min_jets=args.min_jets, n_assign=args.n_assign))
    return concat(*parts) if len(parts) > 1 else parts[0]


def cached(args, sizes, which):
    """Parsing millions of events from h5 takes minutes; cache the arrays."""
    if not args.cache_dir:
        return load_split(args, sizes, which)
    key = json.dumps([which, args.sig, args.bkg, args.val_sig, args.val_bkg, args.test_sig,
                      args.test_bkg, args.n_jets, args.k, str(sizes), args.min_jets, args.proxy,
                      args.n_assign,
                      args.max_sig, args.max_bkg])
    path = os.path.join(args.cache_dir, f"{which}_{hashlib.md5(key.encode()).hexdigest()[:12]}.npz")
    if os.path.exists(path):
        print(f"cache hit {path}", flush=True)
        return dict(np.load(path))
    d = load_split(args, sizes, which)
    if d is not None:
        os.makedirs(args.cache_dir, exist_ok=True)
        np.savez(path, **d)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sig", nargs="+", required=True)
    ap.add_argument("--bkg", nargs="*", default=[])
    ap.add_argument("--val-sig", nargs="*", default=[], help="explicit validation files")
    ap.add_argument("--val-bkg", nargs="*", default=[])
    ap.add_argument("--test-sig", nargs="*", default=[], help="explicit test files (else random split)")
    ap.add_argument("--test-bkg", nargs="*", default=[])
    ap.add_argument("--device", default="auto", help="auto | cuda | cuda:1 | mps | cpu")
    ap.add_argument("--amp", action="store_true", help="bfloat16 autocast on CUDA")
    ap.add_argument("--eval-bs", type=int, default=4096)
    ap.add_argument("--cache-dir", default=None, help="cache parsed arrays here (.npz)")
    ap.add_argument("--select", choices=["sig_at_bkg", "combined"], default="sig_at_bkg")
    ap.add_argument("--select-bkg-pass", type=float, default=0.05,
                    help="weighted QCD pass rate at which --select sig_at_bkg compares epochs")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from <out>/last.pt (same --epochs)")
    ap.add_argument("--n-jets", type=int, default=8)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n-assign", type=int, default=None,
                    help="pairings use the leading N jets; --n-jets sets the context (encoder) size")
    ap.add_argument("--mass-head", action="store_true",
                    help="also learn a scale-free mass estimate log(M/HT) (output m_est)")
    ap.add_argument("--w-mass", type=float, default=1.0)
    ap.add_argument("--gate-interp", action="store_true",
                    help="gate also sees the softmax-pooled interpretation summary and best scores")
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
    ap.add_argument("--gated", action="store_true",
                    help="signal scorer + fixed anti-signal choice, mixed by a learned P(signal) gate")
    ap.add_argument("--proxy", action="store_true",
                    help="train unmatched signal toward the most signal-like interpretation")
    ap.add_argument("--w-proxy", type=float, default=0.5)
    ap.add_argument("--bg-beta", type=float, default=1.0)
    ap.add_argument("--bkg-weight", type=float, default=1.0)
    ap.add_argument("--max-sig", type=int, default=None)
    ap.add_argument("--max-bkg", type=int, default=None)
    ap.add_argument("--min-jets", type=int, default=6, help="signal-region jet multiplicity")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sizes = tuple((int(x[0]), int(x[1])) for x in args.sizes.split(",")) if args.sizes else None
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    dev = pick_device(args.device)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)

    data = cached(args, sizes, "train")
    va = cached(args, sizes, "val")
    te = cached(args, sizes, "test")
    if va is None and te is None:
        tr, va, te = split(data, seed=args.seed)
    elif va is None:
        tr, va, _ = split(data, fracs=(0.85, 0.15, 0.0), seed=args.seed)
    else:
        tr = data
    if te is None:
        te = va
    ttr, tva, tte = ({k: v.to(dev) for k, v in to_tensors(d).items()} for d in (tr, va, te))
    s_tr = tr["is_signal"]
    print(f"device {dev}  train {len(tr['pt'])}  val {len(va['pt'])}  test {len(te['pt'])}; "
          f"train signal with exact truth {np.mean(tr['target'][s_tr] >= 0):.3f}", flush=True)

    net = PairNet(
        n_jets=args.n_jets, k=args.k, d_model=args.d_model, nhead=args.heads,
        num_layers=args.layers, dim_ff=2 * args.d_model, dropout=args.dropout,
        use_pair_bias=not args.no_pair_bias, use_group_kin=not args.no_group_kin,
        use_null=not args.no_null, sizes=sizes, gated=args.gated, bg_beta=args.bg_beta,
        n_assign=args.n_assign, gate_interp=args.gate_interp, mass_head=args.mass_head,
    ).to(dev)
    A = net.n_assign
    print(f"params {sum(p.numel() for p in net.parameters())/1e3:.0f}k  interpretations {A}")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * math.ceil(len(tr["pt"]) / args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    hist, start = [], 0
    best_score, best_state = -1.0, None
    last = os.path.join(args.out, "last.pt")
    if args.resume and os.path.exists(last):
        ck = torch.load(last, map_location=dev, weights_only=False)
        if ck["args"]["epochs"] != args.epochs:
            sys.exit(f"--resume continues an interrupted run: use --epochs {ck['args']['epochs']} "
                     f"as before (the one-cycle LR schedule is fixed to that length)")
        net.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        hist, start = ck["hist"], ck["epoch"] + 1
        best_score, best_state = ck["best_score"], ck["best_state"]
        print(f"resumed at epoch {start}", flush=True)

    has_bkg = bool((~va["is_signal"]).any())
    for ep in range(start, args.epochs):
        net.train()
        t0 = time.time()
        perm = torch.randperm(len(tr["pt"]), device=dev)
        tot = torch.zeros((), device=dev)
        nb = 0
        for b in batches(ttr, perm, args.bs):
            with autocast(dev, args.amp):
                o = net(b["pt"], b["eta"], b["phi"], b["m"], b["mask"])
                o = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                     for k, v in o.items()}
                if args.gated:
                    loss = gated_loss(o, b["target"], b["is_signal"], w_proxy=args.w_proxy,
                                      bkg_weight=args.bkg_weight, pmass=b["pmass"], w_mass=args.w_mass)
                else:
                    loss = pairnet_loss(o, b["target"].clamp(min=-1), b["is_signal"],
                                        bkg_weight=args.bkg_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.detach()
            nb += 1
        ev = evaluate(net, tva, A, args.eval_bs, args.amp)
        r = summarize(ev)
        r.update(epoch=ep, loss=float(tot) / nb, time=time.time() - t0)
        hist.append(r)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}),
              flush=True)
        if args.select == "sig_at_bkg" and has_bkg:
            score = sig_at_bkg_curve(ev, (args.select_bkg_pass,))[f"{args.select_bkg_pass:g}"]
        else:
            score = r["eff_on_all_signal"] + (1 - r.get("bkg_pass_asym", 1)) * has_bkg
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        torch.save({"state": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "hist": hist, "epoch": ep, "best_score": best_score, "best_state": best_state,
                    "args": vars(args)}, last)

    net.load_state_dict(best_state)
    torch.save({"state": {k: v.cpu() for k, v in best_state.items()}, "args": vars(args)},
               os.path.join(args.out, "model.pt"))
    ev = evaluate(net, tte, A, args.eval_bs, args.amp)
    np.savez_compressed(os.path.join(args.out, "eval.npz"), **ev)
    final = summarize(ev)
    print("TEST", json.dumps(final), flush=True)
    json.dump({"test": final, "history": hist, "args": vars(args)},
              open(os.path.join(args.out, "metrics.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
