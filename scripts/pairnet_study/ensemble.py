"""Test-set ensemble of sweep models (each with its own jet inputs, same events), fp32 on CPU:
average P(signal) and the mixture pairing probabilities, then score with train_pairnet.summarize."""
import glob, sys
import numpy as np, torch
repo, prod = sys.argv[1], sys.argv[2]; models = sys.argv[3:]
sys.path.insert(0, repo); sys.path.insert(0, repo + "/scripts")
import train_pairnet as tp
from src.pairdata import concat, load_events, to_tensors
from src.pairnet import load_checkpoint
torch.set_num_threads(8)
fs, fb = sorted(glob.glob(f"{prod}/sig_*_c*9.h5")), sorted(glob.glob(f"{prod}/qcd_c*9.h5"))
cache, outs = {}, []
for m in models:
    net, a = load_checkpoint(m); J = a["n_jets"]
    if J not in cache:
        cache[J] = concat(load_events(fs, J, 3, True, None, min_jets=6, proxy=True, n_assign=a.get("n_assign")),
                          load_events(fb, J, 3, False, None, min_jets=6, n_assign=a.get("n_assign")))
    d = cache[J]; t = to_tensors(d)
    o = tp.predict(net, t, 8192, False)
    outs.append({k: v.numpy() for k, v in o.items()}); base = d
    ev = tp.evaluate(net, t, net.n_assign, 8192, False); r = tp.summarize(ev)
    print(f"{m.split('/')[-2]:30s} s@2/5/8% {r['sig_in_window_at_bkg']['0.02']:.3f}/{r['sig_in_window_at_bkg']['0.05']:.3f}/{r['sig_in_window_at_bkg']['0.08']:.3f} rej50 {r['bkg_rej_at_50']:.0f} auc {r['auc']:.4f} sigAsym {r['sig_pass_asym']:.3f}", flush=True)
A = outs[0]["logits"].shape[1]
p = np.mean([np.exp(o["logits"]) for o in outs], 0)
psig = np.mean([1 / (1 + np.exp(-o["gate_logit"])) for o in outs], 0)
best = p.argmax(1); ar = np.arange(len(best))
m1, m2, ht = outs[0]["m1"][ar, best], outs[0]["m2"][ar, best], outs[0]["ht"]
ev = {"best": best, "psig": psig, "mavg": 0.5 * (m1 + m2) * ht, "asym": np.abs(m1 - m2) / (m1 + m2),
      "target": base["target"], "is_signal": base["is_signal"], "pmass": base["pmass"], "w": base["w"]}
r = tp.summarize(ev)
print(f"{'ENSEMBLE':30s} s@2/5/8% {r['sig_in_window_at_bkg']['0.02']:.3f}/{r['sig_in_window_at_bkg']['0.05']:.3f}/{r['sig_in_window_at_bkg']['0.08']:.3f} rej50 {r['bkg_rej_at_50']:.0f} auc {r['auc']:.4f} sigAsym {r['sig_pass_asym']:.3f} eff {r['eff_on_reconstructable']:.3f}")
