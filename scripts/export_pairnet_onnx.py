"""
Export a trained PairNet checkpoint to ONNX, as a drop-in for the existing
ml_model.onnx interface (src/export_onnx.py), plus extra outputs.

  python scripts/export_pairnet_onnx.py --model runs/gated_n7/model.pt --out models/pairnet_gated_7jet.onnx

Input
  four_momenta  (batch, J, 4) float32   (E, px, py, pz) of the leading J jets,
                pT-ordered, zero-padded.  J = 7 for 7-jet models (same as the
                current network); 12 for the v2 model (pairings still use the
                leading 7).  Units are free: raw GeV or HT-normalised give the
                same pairings and p_signal (the network divides by HT itself);
                `masses` come out in the input's units.

Outputs (assignment axis in the SAME order as src.combinatorics.enumerate_assignments(7),
so argmax gives the index your code already decodes into (ISR, group1, group2))
  logits         (batch, 70)    gated-mixture log-probabilities: signal-like events
                                get their most signal-like pairing, QCD-like events
                                their high-asymmetry / low-mass pairing.  Drop-in for
                                the current `logits`.
  p_signal       (batch,)       event-level P(signal) from the gate: the recommended
                                selection variable.
  signal_logits  (batch, 70)    the signal scorer alone (pairing to use together with
                                a p_signal cut).
  masses         (batch, 70, 2) m1, m2 of the two triplets of every assignment,
                                in the input's units.
  m_est          (batch,)       only for checkpoints trained with --mass-head: the
                                learned resonance-mass estimate (HT x exp(learned
                                log M/HT)), in the input's units; a narrower peak
                                than m_avg of the chosen pairing.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from src.combinatorics import enumerate_assignments  # noqa: E402
from src.pairnet import asinh, enumerate_interpretations, load_checkpoint  # noqa: E402


def legacy_permutation(net) -> torch.Tensor:
    """perm[i] = PairNet interpretation index of legacy assignment i."""
    ours = {(tuple(a), tuple(b)): i for i, (a, b) in
            enumerate(enumerate_interpretations(net.n_assign_jets, net.sizes))}
    perm = []
    for _isr, g1, g2 in enumerate_assignments(net.n_assign_jets):
        a, b = tuple(sorted(g1)), tuple(sorted(g2))
        perm.append(ours[(min(a, b), max(a, b))])
    return torch.tensor(perm, dtype=torch.long)


class PairNetONNX(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("perm", legacy_permutation(net))

    def forward(self, four_momenta):
        E = four_momenta[..., 0]
        px = four_momenta[..., 1]
        py = four_momenta[..., 2]
        pz = four_momenta[..., 3]
        pt = torch.sqrt(px * px + py * py)
        mask = pt > 0
        zero = torch.zeros_like(pt)
        ptc = torch.clamp(pt, min=1e-6)
        eta = torch.where(mask, asinh(pz / ptc), zero)
        phi = torch.where(mask, torch.atan2(py, px), zero)
        m = torch.where(mask, torch.sqrt(torch.clamp(E * E - px * px - py * py - pz * pz, min=0.0)), zero)
        out = self.net(torch.where(mask, pt, zero), eta, phi, m, mask)
        perm = self.perm
        ht = out["ht"].unsqueeze(-1)
        logits = out["logits"].index_select(1, perm)
        signal_logits = out["assign_logits"].index_select(1, perm)
        p_signal = torch.sigmoid(out["gate_logit"])
        masses = torch.stack([out["m1"].index_select(1, perm) * ht,
                              out["m2"].index_select(1, perm) * ht], -1)
        if "log_m_over_ht" in out:
            return logits, p_signal, signal_logits, masses, torch.exp(out["log_m_over_ht"]) * out["ht"]
        return logits, p_signal, signal_logits, masses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/gated_n7/model.pt")
    ap.add_argument("--out", default="models/pairnet_gated_7jet.onnx")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    net, args = load_checkpoint(a.model)
    if not net.gated:
        sys.exit("this exporter is for gated PairNet checkpoints (--gated)")
    wrapper = PairNetONNX(net).eval()
    J = net.n_jets
    example = torch.zeros(4, J, 4)
    example[:, :7, :] = torch.tensor([[400.0, 300.0, 0.0, 200.0]])  # placeholder, overwritten below
    rng = np.random.default_rng(0)
    pt = np.sort(rng.uniform(40, 600, (4, J)), 1)[:, ::-1]
    eta, phi = rng.normal(0, 1.2, (4, J)), rng.uniform(-np.pi, np.pi, (4, J))
    px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
    example = torch.tensor(np.stack([np.sqrt(px**2 + py**2 + pz**2 + 25), px, py, pz], -1), dtype=torch.float32)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    names_out = ["logits", "p_signal", "signal_logits", "masses"] + (["m_est"] if net.mass_head else [])
    torch.onnx.export(
        wrapper, (example,), a.out,
        input_names=["four_momenta"], output_names=names_out,
        dynamic_axes={"four_momenta": {0: "batch"}, **{n: {0: "batch"} for n in names_out}},
        opset_version=a.opset, dynamo=False,
    )

    import onnx
    model = onnx.load(a.out)
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        commit = "unknown"
    meta = {
        "description": "Gated PairNet: pair-production interpretation + event-level P(signal) from jet four-vectors",
        "input": f"four_momenta (batch,{J},4) = (E,px,py,pz) of the leading {J} jets, pT-ordered, zero-padded",
        "assignment_order": "src.combinatorics.enumerate_assignments(7): index -> (isr, group1, group2)",
        "outputs": "logits: gated mixture log-probs; p_signal: event P(signal); "
                   "signal_logits: signal scorer; masses: (m1, m2) per assignment in input units"
                   + ("; m_est: learned resonance mass in input units" if net.mass_head else ""),
        "checkpoint": a.model, "git_commit": commit,
        "training_args": json.dumps({k: args[k] for k in ("n_jets", "n_assign", "gate_interp", "mass_head", "d_model", "layers",
                                                          "epochs", "sig", "bkg") if k in args}),
    }
    for k, v in meta.items():
        p = model.metadata_props.add()
        p.key, p.value = k, str(v)
    onnx.checker.check_model(model)
    onnx.save(model, a.out)
    spec = {**meta, "assignments": [[isr, list(g1), list(g2)] for isr, g1, g2 in enumerate_assignments(7)]}
    json.dump(spec, open(a.out.replace(".onnx", ".json"), "w"), indent=1)
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB) and its .json spec")


if __name__ == "__main__":
    main()
