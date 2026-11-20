"""Score a train_pairnet.py model on a cached test array: per event gate logit, m_avg (GeV) and asymmetry of the
chosen pairing.  usage: score_ref.py model.pt test.npz out.npz [mixture|scorer]
(scorer: pairing = argmax of the signal scorer alone, independent of P(signal))"""
import sys, numpy as np, torch
sys.path.insert(0, "/scratch/llee/pairnet/audit/F5_train/wt")
from src.pairnet import load_checkpoint
torch.set_num_threads(4)
net, _ = load_checkpoint(sys.argv[1]); d = np.load(sys.argv[2])
mode = sys.argv[4] if len(sys.argv) > 4 else "mixture"
out = {"logit": [], "mavg": [], "asym": []}
with torch.no_grad():
    for i in range(0, len(d["pt"]), 8192):
        b = [torch.from_numpy(np.ascontiguousarray(d[k][i:i + 8192])) for k in ("pt", "eta", "phi", "m", "mask")]
        o = net(*b)
        best = (o["assign_logits"] if mode == "scorer" else o["logits"][:, :o["m1"].shape[1]]).argmax(1); ar = torch.arange(len(best))
        m1, m2 = o["m1"][ar, best], o["m2"][ar, best]
        out["logit"].append(o["gate_logit"]); out["mavg"].append(0.5 * (m1 + m2) * o["ht"])
        out["asym"].append((m1 - m2).abs() / (m1 + m2).clamp(min=1e-6))
np.savez_compressed(sys.argv[3], **{k: torch.cat(v).numpy() for k, v in out.items()},
                    is_signal=d["is_signal"], pmass=d["pmass"], w=d["w"])
print("wrote", sys.argv[3])
