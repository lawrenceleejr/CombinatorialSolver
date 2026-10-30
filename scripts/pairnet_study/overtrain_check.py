"""Overtraining check for a finished sweep run: loss (eval mode) and metrics of the selected
(model.pt) and final (last.pt) weights on 200k training events vs 200k validation events."""
import argparse, glob, json, sys
import numpy as np, torch
repo, run_dir, name, prod, out = sys.argv[1:6]
sys.path.insert(0, repo); sys.path.insert(0, repo + "/scripts")
import train_pairnet as tp
from src.pairdata import concat, load_events, to_tensors
from src.pairnet import load_checkpoint, PairNet
torch.set_num_threads(8)
net_sel, a = load_checkpoint(f"{run_dir}/model.pt")
last = torch.load(f"{run_dir}/last.pt", map_location="cpu", weights_only=False)
net_fin, _ = load_checkpoint(f"{run_dir}/model.pt"); net_fin.load_state_dict(last["state"]); net_fin.eval()
args = argparse.Namespace(**{"w_sym": 0.0, "w_qcd_asym": 0.0, "w_win": 0.0, "w_qcd_pass": 0.0, **a})
rng = np.random.default_rng(0)
def sample(sg, bg, n=100000):
    """n signal + n QCD events, so training and validation samples have the same class mix."""
    ds = load_events(sorted(glob.glob(f"{prod}/{sg}")), a["n_jets"], 3, True, None, min_jets=6, proxy=True, n_assign=a.get("n_assign"))
    db = load_events(sorted(glob.glob(f"{prod}/{bg}")), a["n_jets"], 3, False, None, min_jets=6, n_assign=a.get("n_assign"))
    i_s = rng.choice(len(ds["pt"]), n, replace=False)  # one index per class, shared by all arrays
    i_b = rng.choice(len(db["pt"]), n, replace=False)
    ds = {k: v[i_s] for k, v in ds.items()}
    db = {k: v[i_b] for k, v in db.items()}
    return to_tensors(concat(ds, db))
sets = {"train": sample("sig_*_c[0-1].h5", "qcd_c*[0-1].h5"), "val": sample("sig_*_c*8.h5", "qcd_c*8.h5")}
res = {"run": name, "selected_epoch": None, "final_epoch": last["epoch"]}
h = last["hist"]; key = (lambda r: r["sig_pass_asym_in_window"] / max(r["bkg_pass_asym"], 1e-6) ** 0.5) if a.get("select") == "win_a01" else (lambda r: r["sig_in_window_at_bkg"]["0.05"])
sel = max(h, key=key)
res["selected_epoch"] = sel["epoch"] if sel else None
for wname, net in (("selected", net_sel), ("final", net_fin)):
    for sname, t in sets.items():
        ev = tp.evaluate(net, t, net.n_assign, 8192, False); r = tp.summarize(ev)
        r["loss"] = tp.eval_loss(net, t, args, 8192)
        res[f"{wname}/{sname}"] = {k: r[k] for k in ("loss", "auc", "bkg_rej_at_50", "eff_on_reconstructable", "sig_pass_asym", "bkg_pass_asym")} | {"s@5%": r["sig_in_window_at_bkg"]["0.05"], "win": r["sig_pass_asym_in_window"]}
        x = res[f"{wname}/{sname}"]
        print(f"{name} {wname:8s} {sname:5s} loss {x['loss']:.4f} auc {x['auc']:.4f} rej50 {x['bkg_rej_at_50']:5.0f} s@5% {x['s@5%']:.3f} winA0.1 {x['win']:.3f} eff {x['eff_on_reconstructable']:.3f} sigAsym {x['sig_pass_asym']:.3f} qcdAsym {x['bkg_pass_asym']:.3f}", flush=True)
json.dump(res, open(out, "w"), indent=1)
