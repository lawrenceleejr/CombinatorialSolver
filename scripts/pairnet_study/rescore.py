"""Re-score a sweep model.pt on validation and test in fp32 on CPU with train_pairnet's own code."""
import glob, sys
import numpy as np, torch
repo, model, prod, n_jets = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
sys.path.insert(0, repo); sys.path.insert(0, repo + "/scripts")
import train_pairnet as tp
from src.pairdata import concat, load_events, to_tensors
from src.pairnet import load_checkpoint
torch.set_num_threads(8)
net, a = load_checkpoint(model)
for split, sg, bg in (("val", "sig_*_c*8.h5", "qcd_c*8.h5"), ("test", "sig_*_c*9.h5", "qcd_c*9.h5")):
    d = concat(load_events(sorted(glob.glob(f"{prod}/{sg}")), n_jets, 3, True, None, min_jets=6, proxy=True, n_assign=a.get("n_assign")),
               load_events(sorted(glob.glob(f"{prod}/{bg}")), n_jets, 3, False, None, min_jets=6, n_assign=a.get("n_assign")))
    t = {k: v for k, v in to_tensors(d).items()}
    ev = tp.evaluate(net, t, net.n_assign, 8192, False)
    r = tp.summarize(ev)
    s = ev["is_signal"]; thr = np.quantile(ev["psig"][s], 0.5); q = ~s
    w = ev["w"][q]; pas = ev["psig"][q] >= thr
    print(f"{split}: n_bkg {q.sum()} rej50 {r['bkg_rej_at_50']:.0f} (weighted {w.sum() / max((w * pas).sum(), 1e-9):.0f}, n_pass {pas.sum()})  s@5% {r['sig_in_window_at_bkg']['0.05']:.3f}  auc {r['auc']:.4f}", flush=True)
