"""Per-class overtraining diagnosis: loss terms of r14A and the big models on events from TRAINING files vs
held-out (test) files.  Which class (signal pairing, signal gate, QCD gate) does a bigger network memorise?"""
import sys, glob, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/scratch/llee/pairnet/audit/F5_train/wt")
from src.pairdata import load_events, concat
from src.pairnet import load_checkpoint
torch.set_num_threads(4)
P = "/scratch/llee/pairnet/"
def files(split):
    if split == "train":
        s = sorted(glob.glob(P + "prod/sig_*_c*[0-7].h5"))[::40]
        b = sorted(glob.glob(P + "prod/qcd_c*[0-7].h5"))[:12] + sorted(glob.glob(P + "prod_ht/qcdht_c[0-9][0-9].h5"))[:30] + sorted(glob.glob(P + "stage_r14/lm/qcdvlm_c[0-9][0-9].h5"))[:8]
    else:
        s = sorted(glob.glob(P + "prod/sig_*_c*9.h5"))[::5]
        b = sorted(glob.glob(P + "prod/qcd_c*9.h5"))[:12] + sorted(glob.glob(P + "prod_ht/qcdht_c17[0-9][0-9].h5"))[:30] + sorted(glob.glob(P + "stage_r14/lm/qcdvlm_c10[5-9][0-9].h5"))[:8]
    sig = load_events(s, 12, 3, True, 30000, proxy=True, n_assign=7, min_jets=6)
    bkg = load_events(b, 12, 3, False, 30000, n_assign=7, min_jets=6)
    d = concat(sig, bkg)
    t = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in d.items()}
    for k in ("pt", "eta", "phi", "m", "pmass"):
        t[k] = t[k].float()
    return t
data = {sp: files(sp) for sp in ("train", "test")}
models = {"r14A": "/scratch/llee/pairnet/sweep/out_r14A_small_12j_win5_lowmass_boost/r14A_small_12j_win5_lowmass_boost/model.pt",
          "big8 (d256 L8)": "/scratch/llee/pairnet/audit/F5_train/runs/out_f5_big8/f5_big8/model.pt",
          "d384 L8 ff4": "/scratch/llee/pairnet/audit/F8_capacity/runs/out_f8_d384L8ff4/f8_d384L8ff4/model.pt",
          "ia2": "/scratch/llee/pairnet/audit/F8_capacity/runs/out_f8_ia2/f8_ia2/model.pt"}
for name, path in models.items():
    net, _ = load_checkpoint(path)
    row = {}
    for sp, t in data.items():
        with torch.no_grad():
            o = {}
            for i in range(0, len(t["pt"]), 2048):
                oo = net(*(t[k][i:i + 2048] for k in ("pt", "eta", "phi", "m", "mask")))
                for k in ("logits", "bg_logits", "gate_logit"):
                    o.setdefault(k, []).append(oo[k])
            o = {k: torch.cat(v) for k, v in o.items()}
        s = t["is_signal"].bool(); tg = t["target"]
        ex = s & (tg >= 0); px = s & (tg <= -2)
        idx = torch.where(px, -2 - tg, tg)
        idx = torch.where(s, idx, o["bg_logits"].argmax(-1)).clamp(min=0)
        nll = -o["logits"].gather(1, idx.unsqueeze(1)).squeeze(1)
        bce = F.binary_cross_entropy_with_logits(o["gate_logit"], s.float(), reduction="none")
        row[sp] = (nll[ex].mean().item(), nll[px].mean().item(), bce[s].mean().item(), bce[~s].mean().item())
    tr, te = row["train"], row["test"]
    print(f"{name:16s} " + "  ".join(f"{lab} {a:.4f}/{b:.4f} (gap {b - a:+.4f})" for lab, a, b in
          zip(("sig-pair exact", "sig-pair proxy", "sig gate", "QCD gate"), tr, te)), flush=True)
