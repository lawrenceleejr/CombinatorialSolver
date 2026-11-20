"""Decompose the r14A training loss into its terms on held-out events, and measure the irreducible floor
of each term (what a perfect pairing / perfect gate would still pay)."""
import sys, glob, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/home/llee/CombinatorialSolver")
from src.pairdata import load_events, concat
from src.pairnet import load_checkpoint
torch.set_num_threads(4)
net, a = load_checkpoint("/scratch/llee/pairnet/sweep/out_r14A_small_12j_win5_lowmass_boost/r14A_small_12j_win5_lowmass_boost/model.pt")
P = "/scratch/llee/pairnet/"
sig = load_events(sorted(glob.glob(P + "prod/sig_*_c*9.h5"))[::3], 12, 3, True, 40000, proxy=True, n_assign=7, min_jets=6)
bkg = load_events(sorted(glob.glob(P + "prod/qcd_c*9.h5"))[:20] + sorted(glob.glob(P + "prod_ht/qcdht_c17[0-9][0-9].h5"))[:30]
                  + sorted(glob.glob(P + "stage_r14/lm/qcdvlm_c10[5-9][0-9].h5"))[:10], 12, 3, False, 40000, n_assign=7, min_jets=6)
d = concat(sig, bkg)
t = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in d.items()}
for k in ("pt", "eta", "phi", "m", "pmass"):
    t[k] = t[k].float()
with torch.no_grad():
    o = {}
    for i in range(0, len(t["pt"]), 4096):
        oo = net(*(t[k][i:i + 4096] for k in ("pt", "eta", "phi", "m", "mask")))
        for k in ("logits", "bg_logits", "gate_logit", "log_m_over_ht", "m1", "m2", "ht"):
            o.setdefault(k, []).append(oo[k])
    o = {k: torch.cat(v) for k, v in o.items()}
is_sig = t["is_signal"].bool(); target = t["target"]
exact = is_sig & (target >= 0); proxy = is_sig & (target <= -2)
tgt = torch.where(proxy, -2 - target, target)
bg_t = o["bg_logits"].argmax(-1)
tgt = torch.where(is_sig, tgt, bg_t).clamp(min=0)
nll = -o["logits"].gather(1, tgt.unsqueeze(1)).squeeze(1)
bw = 3.0
w = torch.where(exact, 1.0, torch.where(proxy, 0.5, 0.0)); w = torch.where(is_sig, w, torch.full_like(w, bw))
W = w.sum()
# pure anti-signal distribution: what QCD pays even with gate -> 0
lb = F.log_softmax(o["bg_logits"], -1)
floor_qcd = -lb.gather(1, bg_t.unsqueeze(1)).squeeze(1)
gate = o["gate_logit"]
bce = F.binary_cross_entropy_with_logits(gate, is_sig.float(), reduction="none")
gw = torch.where(is_sig, 1.0, bw)
tgt_m = torch.log(t["pmass"].clamp(min=1) / o["ht"])
lmass = (o["log_m_over_ht"] - tgt_m).abs()[is_sig].mean()
A = o["m1"].shape[1]; p = o["logits"][:, :A].exp()
asym = (o["m1"] - o["m2"]).abs() / (o["m1"] + o["m2"]).clamp(min=1e-6); sym = (asym < 0.1).float()
rel = 0.5 * (o["m1"] + o["m2"]) * o["ht"].unsqueeze(-1) / t["pmass"].clamp(min=1).unsqueeze(-1)
win = sym * ((rel > 0.85) & (rel < 1.1)).float()
ewin = (p * win).sum(-1)[is_sig].mean(); eqcd = (p * sym).sum(-1)[~is_sig].mean()
ns, nb = int(is_sig.sum()), int((~is_sig).sum())
print(f"events: signal {ns} (exact {int(exact.sum())}, proxy {int(proxy.sum())}), QCD {nb}")
print("--- mixture NLL term, contribution to the loss (weighted mean over all events) ---")
for name, m in (("signal exact", exact), ("signal proxy", proxy), ("QCD", ~is_sig)):
    print(f"  {name:13s}: mean NLL {nll[m].mean():.3f}  contribution {(nll * w)[m].sum() / W:.3f}")
print(f"  QCD floor with gate=0 (fixed anti-signal softmax, kappa=30): mean {floor_qcd[~is_sig].mean():.3f}, "
      f"contribution {(floor_qcd * w)[~is_sig].sum() / W:.3f}")
print(f"  QCD events whose anti-signal argmax is ambiguous (top-2 bg logit gap < 1): "
      f"{((lambda s: (s[:, 0] - s[:, 1]) < 1)(o['bg_logits'].topk(2, -1).values))[~is_sig].float().mean():.3f}")
print(f"  signal exact: P(true pairing) median {(-nll[exact]).exp().median():.3f}; floor 0 if perfect")
print(f"--- gate BCE: {(bce * gw).sum() / gw.sum():.3f} (signal {bce[is_sig].mean():.3f}, QCD {bce[~is_sig].mean():.3f}) ---")
print(f"--- mass L1: {lmass:.3f}   window reward -5*{ewin:.3f} = {-5 * ewin:.3f}   QCD pass +5*{eqcd:.3f} = {5 * eqcd:.3f}")
tot = (nll * w).sum() / W + (bce * gw).sum() / gw.sum() + lmass - 5 * ewin + 5 * eqcd
print(f"total {tot:.3f}")
anyw = (win.max(-1).values > 0)[is_sig].float().mean()
print(f"window-reward ceiling (signal events with ANY symmetric in-window pairing among 70): {anyw:.3f}; achieved {ewin:.3f}")
truew = win.gather(1, tgt.unsqueeze(1)).squeeze(1)
print(f"exact signal: true pairing itself in window {truew[exact].mean():.3f}; prob mass on in-window pairings {(p*win).sum(-1)[exact].mean():.3f}, on the true one {(-nll[exact]).exp().mean():.3f}")
nwin = win.sum(-1)[is_sig]
print(f"number of in-window symmetric pairings per signal event: median {nwin.median():.0f}, mean {nwin.mean():.1f}")
gs = torch.sigmoid(gate)
print("signal gate P quantiles 10/50/90%:", [round(float(x), 3) for x in torch.quantile(gs[is_sig], torch.tensor([.1, .5, .9]))])
