"""Compare two PairNet checkpoints (epoch 2 = model.pt, epoch 39 = last.pt) on
seen (training) and unseen QCD, to explain why --select combined picked epoch 2.
Writes per-event arrays to an npz for plotting."""
import glob, json, sys
import numpy as np
import torch

repo, run, out = sys.argv[1:4]
sys.path.insert(0, repo)
from src.pairdata import load_events, to_tensors, concat  # noqa: E402
from src.pairnet import PairNet  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

torch.set_num_threads(8)
P = "/scratch/llee/pairnet/prod/"


def build(state, a):
    net = PairNet(n_jets=a["n_jets"], k=a["k"], d_model=a["d_model"], nhead=a["heads"],
                  num_layers=a["layers"], dim_ff=2 * a["d_model"], dropout=0.0,
                  use_pair_bias=not a["no_pair_bias"], use_group_kin=not a["no_group_kin"],
                  use_null=not a["no_null"], sizes=None, gated=a["gated"], bg_beta=a["bg_beta"],
                  n_assign=a["n_assign"], gate_interp=a["gate_interp"], mass_head=a["mass_head"],
                  massless=a.get("massless", False))
    net.load_state_dict(state)
    return net.eval()


last = torch.load(f"{run}/last.pt", map_location="cpu", weights_only=False)
sel = torch.load(f"{run}/model.pt", map_location="cpu", weights_only=False)
a = last["args"]
nets = {"ep2": build(sel["state"], a), "ep39": build(last["state"], a)}

rng = np.random.default_rng(0)
samples = {
    "qcd_seen": load_events(sorted(glob.glob(P + "qcd_c[0-7].h5")), 7, 3, False, None, min_jets=6),
    "qcd_unseen": load_events(sorted(glob.glob(P + "qcd_c[1-3][0-9].h5")), 7, 3, False, None, min_jets=6),
    "sig_seen": load_events(sorted(glob.glob(P + "sig_*_c0.h5")), 7, 3, True, None, min_jets=6, proxy=True),
    "sig_test": load_events(sorted(glob.glob(P + "sig_*_c9.h5")), 7, 3, True, None, min_jets=6, proxy=True),
}
for k in ("sig_seen", "sig_test"):  # subsample signal, QCD is the limiting sample
    d = samples[k]
    idx = rng.choice(len(d["pt"]), 42000, replace=False)
    samples[k] = {kk: v[idx] for kk, v in d.items()}
print({k: len(v["pt"]) for k, v in samples.items()}, flush=True)

res, summary = {}, {}
with torch.no_grad():
    for mname, net in nets.items():
        A = net.n_assign
        for sname, d in samples.items():
            t = to_tensors(d)
            cols = {k: [] for k in ("psig", "asym", "from_bg", "smax", "best", "mavg")}
            for i in range(0, len(d["pt"]), 4096):
                o = net(*(t[k][i:i + 4096] for k in ("pt", "eta", "phi", "m", "mask")))
                best = o["logits"][:, :A].argmax(1)
                ar = torch.arange(len(best))
                m1, m2 = o["m1"][ar, best], o["m2"][ar, best]
                cols["psig"].append(torch.sigmoid(o["gate_logit"]))
                cols["asym"].append((m1 - m2).abs() / (m1 + m2))
                cols["from_bg"].append(best == o["bg_logits"].argmax(1))
                cols["smax"].append(o["assign_logits"].softmax(-1).max(-1).values)
                cols["best"].append(best)
                cols["mavg"].append(0.5 * (m1 + m2) * o["ht"])
            for k, v in cols.items():
                res[f"{mname}/{sname}/{k}"] = torch.cat(v).float().numpy()
            res[f"{mname}/{sname}/w"] = d["w"]
            res[f"{mname}/{sname}/target"] = d["target"]
            res[f"{mname}/{sname}/pmass"] = d["pmass"]
            print(mname, sname, "done", flush=True)

        for tag, qs, ss in (("seen", "qcd_seen", "sig_seen"), ("unseen", "qcd_unseen", "sig_test")):
            pq, ps = res[f"{mname}/{qs}/psig"], res[f"{mname}/{ss}/psig"]
            y = np.r_[np.zeros(len(pq)), np.ones(len(ps))]
            summary[f"{mname} gate AUC {tag}"] = round(roc_auc_score(y, np.r_[pq, ps]), 4)
            thr = np.quantile(ps, 0.5)
            summary[f"{mname} QCD pass at 50% sig eff, {tag}"] = round(float((pq >= thr).mean()), 5)
        for sname in samples:
            w = res[f"{mname}/{sname}/w"]
            pas = res[f"{mname}/{sname}/asym"] < 0.1
            fb = res[f"{mname}/{sname}/from_bg"]
            summary[f"{mname} {sname}: pass asym<0.1"] = round(float((w * pas).sum() / w.sum()), 4)
            summary[f"{mname} {sname}: pairing = fixed anti-signal choice"] = round(float(fb.mean()), 4)
            summary[f"{mname} {sname}: median P(sig)"] = round(float(np.median(res[f"{mname}/{sname}/psig"])), 4)
            summary[f"{mname} {sname}: mean max softmax of signal scorer"] = round(float(res[f"{mname}/{sname}/smax"].mean()), 4)
np.savez_compressed(out, **res)
print(json.dumps(summary, indent=1))
