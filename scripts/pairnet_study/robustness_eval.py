"""Shower-variation robustness of finished sweep models (fp32, CPU).

For each model and each sample set (nominal test files, alpha_s(FSR, ISR) = 0.123 and 0.150):
signal in window at A < 0.1, weighted QCD pass at A < 0.1 and their ratio, pairing efficiency,
and the QCD efficiency at the P(signal) threshold fixed on the nominal sample (50% nominal
signal efficiency), as an analysis would apply it."""
import glob, json, sys
import numpy as np, torch

repo, out = sys.argv[1], sys.argv[2]; models = sys.argv[3:]
sys.path.insert(0, repo); sys.path.insert(0, repo + "/scripts")
import train_pairnet as tp
from src.pairdata import concat, load_events, to_tensors
from src.pairnet import load_checkpoint
torch.set_num_threads(8)

D = "/scratch/llee/pairnet"
MASSES = (500, 1000, 1500, 2000)
SETS = {
    "nominal": ([f"{D}/data_r3/prod/sig_m{M}_c{c}.h5" for M in MASSES for c in (9, 19)], sorted(glob.glob(f"{D}/data_r3/prod/qcd_c*9.h5"))[:50]),
    "alphas 0.123": (sorted(glob.glob(f"{D}/prod_as123/sig_*.h5")), sorted(glob.glob(f"{D}/prod_as123/qcd_*.h5"))),
    "alphas 0.150": (sorted(glob.glob(f"{D}/prod_as150/sig_*.h5")), sorted(glob.glob(f"{D}/prod_as150/qcd_*.h5"))),
}
res = {}
for m in models:
    net, a = load_checkpoint(m); name = m.split("/")[-2]; J = a["n_jets"]
    thr = None
    for sname, (fs, fb) in SETS.items():
        d = concat(load_events(fs, J, 3, True, None, min_jets=6, proxy=True, n_assign=a.get("n_assign")),
                   load_events(fb, J, 3, False, None, min_jets=6, n_assign=a.get("n_assign")))
        ev = tp.evaluate(net, to_tensors(d), net.n_assign, 8192, False)
        s = ev["is_signal"]; q = ~s; w = ev["w"]; A = ev["asym"] < 0.1
        rel = ev["mavg"] / np.maximum(ev["pmass"], 1); win = (rel > 0.85) & (rel < 1.1)
        if thr is None:
            thr = np.quantile(ev["psig"][s], 0.5)  # fixed on nominal
        r = {"n_sig": int(s.sum()), "n_qcd": int(q.sum()),
             "win": float((A & win)[s].mean()), "qcd_pass_A": float((w * A)[q].sum() / w[q].sum()),
             "sig_eff_at_nominal_thr": float((ev["psig"][s] >= thr).mean()),
             "qcd_eff_at_nominal_thr": float((w * (ev["psig"] >= thr))[q].sum() / w[q].sum()),
             "pairing_eff": float((ev["best"] == ev["target"])[s & (ev["target"] >= 0)].mean())}
        r["ratio"] = r["win"] / r["qcd_pass_A"] ** 0.5
        for M in MASSES:
            sel = s & (ev["pmass"] == M)
            r[f"win_m{M}"] = float((A & win)[sel].mean())
        res[f"{name} | {sname}"] = r
        print(f"{name:28s} {sname:13s} win {r['win']:.3f} qcdA {r['qcd_pass_A']:.4f} ratio {r['ratio']:.3f} | at nominal P thr: sig {r['sig_eff_at_nominal_thr']:.3f} qcd {r['qcd_eff_at_nominal_thr']:.2e} | pairing {r['pairing_eff']:.3f} | "
              + " ".join(f"m{M}:{r[f'win_m{M}']:.3f}" for M in MASSES) + f" | nQCD {r['n_qcd']}", flush=True)
json.dump(res, open(out, "w"), indent=1)
