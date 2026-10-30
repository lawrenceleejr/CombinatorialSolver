"""Write a test-set eval.npz for a run's final weights (last.pt), in the same format as train_pairnet's."""
import glob, sys
import numpy as np, torch
repo, run_dir, out = sys.argv[1:4]
sys.path.insert(0, repo); sys.path.insert(0, repo + "/scripts")
import train_pairnet as tp
from src.pairdata import concat, load_events, to_tensors
from src.pairnet import load_checkpoint
torch.set_num_threads(8)
net, a = load_checkpoint(f"{run_dir}/model.pt")
last = torch.load(f"{run_dir}/last.pt", map_location="cpu", weights_only=False)
net.load_state_dict(last["state"]); net.eval()
P = "/scratch/llee/pairnet/data_r3/prod"
d = concat(load_events(sorted(glob.glob(f"{P}/sig_*_c*9.h5")), a["n_jets"], 3, True, None, min_jets=6, proxy=True, n_assign=a.get("n_assign")),
           load_events(sorted(glob.glob(f"{P}/qcd_c*9.h5")), a["n_jets"], 3, False, None, min_jets=6, n_assign=a.get("n_assign")))
ev = tp.evaluate(net, to_tensors(d), net.n_assign, 8192, False)
np.savez_compressed(out, **{k: np.asarray(v) for k, v in ev.items()})
r = tp.summarize(ev)
print(f"final epoch {last['epoch']}: win {r['sig_pass_asym_in_window']:.3f} qcdA {r['bkg_pass_asym']:.3f} rej50 {r['bkg_rej_at_50']:.0f} eff {r['eff_on_reconstructable']:.3f}")
