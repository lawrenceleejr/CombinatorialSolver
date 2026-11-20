"""
Train / evaluate JetAssign (per-jet parent labels, free group sizes) on ancestry-labelled signal + QCD.

  python scripts/train_jetassign.py --sig "anc/sigan_m*_c0[0-9].h5" --bkg "prod/qcd_c*[0-7].h5" \
      --val-sig ... --val-bkg ... --test-sig ... --test-bkg ... --epochs 20 --bs 1024 --out runs/ja

Writes <out>/model.pt (selected epoch), <out>/last.pt (resumable), <out>/metrics.json and
<out>/eval.npz.  Every epoch logs the validation loss and the same loss on a training subset of
the validation size (overtraining check), and the selection metric: signal with the hard groups in
[0.85, 1.1] M and A < 0.1 at the P(signal) threshold that keeps a weighted fraction --select-bkg-pass
of the QCD passing A < 0.1.
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
from src.jadata import load  # noqa: E402
from src.jetassign import JetAssign, jetassign_loss  # noqa: E402

KEYS = ("pt", "eta", "phi", "m", "mask", "labels", "pmass", "w", "is_signal")


def batches(t, idx, bs):
    for i in range(0, len(idx), bs):
        j = idx[i: i + bs]
        yield {k: v[j] for k, v in t.items()}


def augment(b, a):
    """Smear jet pT and mass (training only).  Jets stay in their original order."""
    b = dict(b)
    pt, m = b["pt"], b["m"]
    if a.aug_jer:
        r = (1 + a.aug_jer * torch.randn_like(pt)).clamp(min=0.3)
        pt, m = pt * r, m * r
    if a.aug_msmear:
        m = m * torch.exp(a.aug_msmear * torch.randn_like(m))
    b["pt"], b["m"] = pt * b["mask"], m * b["mask"]
    return b


def lossargs(a):
    return dict(bkg_weight=a.bkg_weight, w_gate=a.w_gate, w_mass=a.w_mass, w_grpmass=a.w_grpmass,
                w_win=a.w_win, w_qcd_pass=a.w_qcd_pass, w_pair=a.w_pair)


def fwd(net, b):
    return net(b["pt"], b["eta"], b["phi"], b["m"], b["mask"])


@torch.no_grad()
def predict(net, t, args, bs=4096):
    net.eval()
    keep = ("hard", "m1", "m2", "asym", "ht", "gate_logit", "log_m_over_ht")
    outs = {k: [] for k in keep}
    tot, n = 0.0, 0
    for b in batches(t, torch.arange(len(t["pt"]), device=t["pt"].device), bs):
        o = fwd(net, b)
        loss, _ = jetassign_loss(o, b["labels"], b["is_signal"], b["mask"], b["pmass"], **lossargs(args))
        tot += float(loss) * len(b["pt"])
        n += len(b["pt"])
        for k in keep:
            outs[k].append(o[k].float().cpu())
    r = {k: torch.cat(v).numpy() for k, v in outs.items()}
    r["loss"] = tot / max(n, 1)
    return r


def summarize(o, t, bkg_pass=(0.005, 0.001, 0.0002)):
    s = t["is_signal"].cpu().numpy().astype(bool)
    lab = t["labels"].cpu().numpy()
    mask = t["mask"].cpu().numpy()
    pm = t["pmass"].cpu().numpy()
    w = t["w"].cpu().numpy()
    mavg = 0.5 * (o["m1"] + o["m2"]) * o["ht"]
    rel = mavg / np.maximum(pm, 1)
    passA = o["asym"] < 0.1
    win = passA & (rel > 0.85) & (rel < 1.1)
    narrow = passA & (rel > 0.92) & (rel < 1.04)
    psig = 1 / (1 + np.exp(-o["gate_logit"]))
    r = {"val_loss": o["loss"]}
    if s.any():
        has = (lab.sum(-1) > 0.5) & mask
        hard = o["hard"].argmax(-1)
        truth = lab.argmax(-1)
        swap = np.where(hard == 0, 1, np.where(hard == 1, 0, 2))  # parent labels are interchangeable
        c1 = ((hard == truth) & has).sum(1)
        c2 = ((swap == truth) & has).sum(1)
        hard = np.where((c2 > c1)[:, None], swap, hard)
        ok = (hard == truth) | ~has
        r["jet_acc"] = float((hard == truth)[has & s[:, None]].mean())
        r["event_exact"] = float(ok[s].all(1).mean())
        r["sig_win"] = float(win[s].mean())
        r["sig_narrow"] = float(narrow[s].mean())
        lo, hi = np.quantile(rel[s], [0.16, 0.84])
        r["mavg_rel_halfwidth68"] = float((hi - lo) / (hi + lo))
        r["sig_passA"] = float(passA[s].mean())
    if (~s).any():
        r["bkg_passA_w"] = float((w[~s] * passA[~s]).sum() / w[~s].sum())
    if s.any() and (~s).any():
        from sklearn.metrics import roc_auc_score
        r["auc"] = float(roc_auc_score(s.astype(int), o["gate_logit"]))
        qb = ~s & passA
        wq, pq = w[qb], o["gate_logit"][qb]
        order = np.argsort(-pq)
        cw = np.cumsum(wq[order]) / wq.sum()
        r["sig_win_at_bkg"] = {}
        for bp in bkg_pass:
            thr = pq[order][min(np.searchsorted(cw, bp), len(order) - 1)]
            r["sig_win_at_bkg"][f"{bp:g}"] = float((win & (o["gate_logit"] >= thr))[s].mean())
    return r


def load_split(a, which):
    fs = {"train": a.sig, "val": a.val_sig, "test": a.test_sig}[which]
    fb = {"train": a.bkg, "val": a.val_bkg, "test": a.test_bkg}[which]
    tr = which == "train"
    d = [load(fs, a.n_jets, True, a.min_jets, a.max_sig if tr else None)]
    if fb:
        d.append(load(fb, a.n_jets, False, a.min_jets, a.max_bkg if tr else None))
    return {k: np.concatenate([x[k] for x in d]) for k in d[0]}


def cached(a, which):
    if not a.cache_dir:
        return load_split(a, which)
    key = json.dumps([which, a.sig, a.bkg, a.val_sig, a.val_bkg, a.test_sig, a.test_bkg, a.n_jets,
                      a.min_jets, a.max_sig, a.max_bkg, "ja1"])
    path = os.path.join(a.cache_dir, f"ja_{which}_{hashlib.md5(key.encode()).hexdigest()[:12]}.npz")
    if os.path.exists(path):
        print(f"cache hit {path}", flush=True)
        return dict(np.load(path))
    d = load_split(a, which)
    os.makedirs(a.cache_dir, exist_ok=True)
    np.savez(path, **d)
    return d


def to_dev(d, dev):
    t = {}
    for k in KEYS:
        v = d[k]
        if v.dtype == np.float64 and k != "w":
            v = v.astype(np.float32)
        t[k] = torch.from_numpy(np.ascontiguousarray(v)).to(dev)
    t["w"] = t["w"].float()
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sig", nargs="+", required=True)
    ap.add_argument("--bkg", nargs="*", default=[])
    ap.add_argument("--val-sig", nargs="+", required=True)
    ap.add_argument("--val-bkg", nargs="*", default=[])
    ap.add_argument("--test-sig", nargs="*", default=[])
    ap.add_argument("--test-bkg", nargs="*", default=[])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--n-jets", type=int, default=12)
    ap.add_argument("--min-jets", type=int, default=6)
    ap.add_argument("--max-sig", type=int, default=None)
    ap.add_argument("--max-bkg", type=int, default=None)
    ap.add_argument("--max-val", type=int, default=None, help="random subset of the validation set (quick runs)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--refine", type=int, default=2, help="layers of the refinement pass (0: none)")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--pair-head", action="store_true",
                    help="pairwise same-parent head; search groups from its exact best 2-colouring")
    ap.add_argument("--w-pair", type=float, default=1.0)
    ap.add_argument("--massless", action="store_true")
    ap.add_argument("--bkg-weight", type=float, default=3.0)
    ap.add_argument("--w-gate", type=float, default=1.0)
    ap.add_argument("--w-mass", type=float, default=1.0)
    ap.add_argument("--w-grpmass", type=float, default=0.0)
    ap.add_argument("--w-win", type=float, default=0.0)
    ap.add_argument("--w-qcd-pass", type=float, default=0.0)
    ap.add_argument("--select-bkg-pass", type=float, default=0.001)
    ap.add_argument("--aug-msmear", type=float, default=0.0,
                    help="training only: multiply each jet mass by exp(N(0, s)) (jet-mass resolution)")
    ap.add_argument("--aug-jer", type=float, default=0.0,
                    help="training only: multiply each jet pT and mass by 1 + N(0, s) (jet energy resolution)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    dev = torch.device("cuda" if a.device == "auto" and torch.cuda.is_available() else
                       ("cpu" if a.device == "auto" else a.device))
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(a.out, exist_ok=True)
    tr, va = cached(a, "train"), cached(a, "val")
    if a.max_val and a.max_val < len(va["pt"]):
        pick = np.sort(np.random.default_rng(7).choice(len(va["pt"]), a.max_val, replace=False))
        va = {k: v[pick] for k, v in va.items()}
    te = cached(a, "test") if a.test_sig else va
    ttr, tva, tte = (to_dev(d, dev) for d in (tr, va, te))
    print(f"device {dev}  train {len(tr['pt'])} (sig {tr['is_signal'].sum()})  val {len(va['pt'])}  "
          f"test {len(te['pt'])}", flush=True)

    net = JetAssign(n_jets=a.n_jets, d_model=a.d_model, nhead=a.heads, num_layers=a.layers,
                    dim_ff=2 * a.d_model, dropout=a.dropout, massless=a.massless, refine=a.refine,
                    pair_head=a.pair_head).to(dev)
    print(f"params {sum(p.numel() for p in net.parameters()) / 1e3:.0f}k", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * math.ceil(len(tr["pt"]) / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)

    hist, start, best, best_state = [], 0, -1.0, None
    last = os.path.join(a.out, "last.pt")
    if a.resume and os.path.exists(last):
        ck = torch.load(last, map_location=dev, weights_only=False)
        net.load_state_dict(ck["state"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        hist, start, best, best_state = ck["hist"], ck["epoch"] + 1, ck["best"], ck["best_state"]
        print(f"resumed at epoch {start}", flush=True)

    # training subset with the validation set's signal / QCD counts (overtraining check)
    g = torch.Generator().manual_seed(1234)
    trs = torch.from_numpy(tr["is_signal"])
    sub = []
    for want, cls in ((int(va["is_signal"].sum()), True), (int((~va["is_signal"]).sum()), False)):
        idx = torch.nonzero(trs == cls).squeeze(1)
        sub.append(idx[torch.randperm(len(idx), generator=g)[:want]])
    sub = torch.cat(sub).to(dev)
    tsub = {k: v[sub] for k, v in ttr.items()}

    for ep in range(start, a.epochs):
        net.train()
        t0 = time.time()
        tot, nb = 0.0, 0
        for b in batches(ttr, torch.randperm(len(tr["pt"]), device=dev), a.bs):
            if a.aug_msmear or a.aug_jer:
                b = augment(b, a)
            loss, _ = jetassign_loss(fwd(net, b), b["labels"], b["is_signal"], b["mask"], b["pmass"], **lossargs(a))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            tot += float(loss.detach()); nb += 1
        r = summarize(predict(net, tva, a), tva)
        rt = summarize(predict(net, tsub, a), tsub)
        r.update(epoch=ep, loss=tot / nb, time=time.time() - t0, train_eval_loss=rt["val_loss"],
                 train_sub={k: rt.get(k) for k in ("jet_acc", "event_exact", "sig_win", "auc", "sig_win_at_bkg")})
        hist.append(r)
        print(json.dumps(r), flush=True)
        score = r.get("sig_win_at_bkg", {}).get(f"{a.select_bkg_pass:g}", r.get("sig_win", 0.0))
        if score > best:
            best = score
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        torch.save({"state": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "hist": hist, "epoch": ep, "best": best, "best_state": best_state, "args": vars(a)}, last)

    net.load_state_dict(best_state)
    torch.save({"state": {k: v.cpu() for k, v in best_state.items()}, "args": vars(a)},
               os.path.join(a.out, "model.pt"))
    o = predict(net, tte, a)
    np.savez_compressed(os.path.join(a.out, "eval.npz"), **{k: v for k, v in o.items() if k != "loss"},
                        labels=te["labels"], mask=te["mask"], pmass=te["pmass"], w=te["w"],
                        is_signal=te["is_signal"], pt=te["pt"])
    final = summarize(o, tte)
    print("TEST", json.dumps(final), flush=True)
    json.dump({"test": final, "history": hist, "args": vars(a)}, open(os.path.join(a.out, "metrics.json"), "w"),
              indent=1)


if __name__ == "__main__":
    main()
