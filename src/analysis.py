"""
Standalone bump-hunt analysis on a *final* trained network.

Loads a trained model (ONNX via onnxruntime, or a ``.pt`` checkpoint), runs it
over a signal and a background HDF5 sample, picks each event's interpretation,
computes the analysis observables on the chosen triplets, and produces:

  - a cutflow table (fraction of signal and background remaining at each cut);
  - ROC scans for every cut variable plus a cloud of many combined working
    points with the Pareto front;
  - the average-mass bump-hunt histogram before/after the cuts.

This is the offline counterpart of the per-epoch ROC animation produced during
training (``src.train``); both share the cut/observable logic in
``src.cut_analysis`` and the plotting in ``src.cut_plots``.

Example
-------
    # On the ONNX bundle written by training (ml_model_*.onnx):
    python -m src.analysis \
        --onnx onnx_snapshots/final_.../ml_model_final_....onnx \
        --signal "data/gluino_rpv_1tev*.h5" \
        --background "data/qcd*.h5" \
        --output results/bump_hunt

    # Or directly from a checkpoint:
    python -m src.analysis --checkpoint checkpoints/best_model.pt \
        --signal "data/sig*.h5" --background "data/bkg*.h5"

The cut thresholds default to the requested working point
(Δφ > 2.5, mass asymmetry < 0.4, average boost < 2); the Dalitz edge/corner
cuts are off by default and can be dialled in with ``--dalitz-edge`` /
``--dalitz-corner`` or explored via the ROC working-point cloud.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .combinatorics import enumerate_assignments
from .cut_analysis import DEFAULT_CUTS, Observables, cutflow, format_cutflow
from .cut_plots import make_summary_figure
from .dataset import JetAssignmentDataset


# --------------------------------------------------------------------------- #
# Observable computation from raw four-momenta + chosen assignment
# --------------------------------------------------------------------------- #
def _inv_mass(p: np.ndarray) -> np.ndarray:
    m2 = p[..., 0] ** 2 - p[..., 1] ** 2 - p[..., 2] ** 2 - p[..., 3] ** 2
    return np.sqrt(np.clip(m2, 0.0, None))


def compute_observables_from_momenta(
    raw_four_mom: np.ndarray,
    pred_idx: np.ndarray,
    num_jets: int,
    weight: np.ndarray | None = None,
) -> Observables:
    """Compute the analysis observables for each event's chosen interpretation.

    Args:
        raw_four_mom: ``(N, num_jets, 4)`` un-normalised (E, px, py, pz) [GeV].
        pred_idx: ``(N,)`` predicted assignment index into
            :func:`enumerate_assignments`.
        num_jets: 6 or 7.
        weight: optional ``(N,)`` per-event weights.
    """
    assignments = enumerate_assignments(num_jets)
    g1_arr = np.array([list(a[1]) for a in assignments], dtype=int)  # (na, 3)
    g2_arr = np.array([list(a[2]) for a in assignments], dtype=int)

    N = raw_four_mom.shape[0]
    rows = np.arange(N)[:, None]
    g1 = g1_arr[pred_idx]   # (N, 3)
    g2 = g2_arr[pred_idx]

    j1 = raw_four_mom[rows, g1]   # (N, 3, 4)
    j2 = raw_four_mom[rows, g2]
    p1 = j1.sum(axis=1)           # (N, 4)
    p2 = j2.sum(axis=1)

    m1 = _inv_mass(p1)
    m2 = _inv_mass(p2)
    msum = m1 + m2
    avg_mass = msum / 2.0
    mass_asym = np.abs(m1 - m2) / np.clip(msum, 1e-8, None)

    # Δφ between the two parent sums, folded to [0, π].
    phi1 = np.arctan2(p1[:, 2], p1[:, 1])
    phi2 = np.arctan2(p2[:, 2], p2[:, 1])
    dphi = np.abs(phi1 - phi2)
    dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)

    # Average Lorentz boost γ = E/m of the two triplets.  Clamp at 200 to match
    # the model's training-time definition (degenerate near-massless triplets
    # would otherwise produce huge outliers that distort the ROC threshold range).
    g1_boost = np.clip(p1[:, 0] / np.clip(m1, 1e-8, None), None, 200.0)
    g2_boost = np.clip(p2[:, 0] / np.clip(m2, 1e-8, None), None, 200.0)
    avg_boost = 0.5 * (g1_boost + g2_boost)

    # Dalitz coordinates per triplet (pT-ordered jets), normalised by M².
    def _dalitz(jets, m_parent):
        pt = np.sqrt(jets[..., 1] ** 2 + jets[..., 2] ** 2)        # (N, 3)
        order = np.argsort(-pt, axis=1)                            # descending
        js = np.take_along_axis(jets, order[..., None], axis=1)    # (N, 3, 4)
        a, b, c = js[:, 0], js[:, 1], js[:, 2]
        denom = np.clip(m_parent ** 2, 1e-8, None)
        x = _inv_mass(a + b) ** 2 / denom
        y = _inv_mass(a + c) ** 2 / denom
        return x, y

    x1, y1 = _dalitz(j1, m1)
    x2, y2 = _dalitz(j2, m2)
    dalitz_x = np.stack([x1, x2], axis=1)   # (N, 2)
    dalitz_y = np.stack([y1, y2], axis=1)

    from .cut_analysis import dalitz_edge_corner
    edge, corner = dalitz_edge_corner(dalitz_x, dalitz_y)

    return Observables(
        avg_mass=avg_mass, mass_asym=mass_asym, delta_phi=dphi,
        avg_boost=avg_boost, dalitz_edge=edge, dalitz_corner=corner,
        weight=None if weight is None else np.asarray(weight, dtype=float),
    )


# --------------------------------------------------------------------------- #
# Inference back-ends
# --------------------------------------------------------------------------- #
def _predict_onnx(onnx_path: str, four_mom: np.ndarray, batch: int = 2048) -> np.ndarray:
    """Run an ONNX model and return argmax assignment indices."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    preds = []
    for i in range(0, len(four_mom), batch):
        chunk = four_mom[i:i + batch].astype(np.float32)
        logits = sess.run([out_name], {in_name: chunk})[0]
        preds.append(logits.argmax(axis=-1))
    return np.concatenate(preds)


def _predict_checkpoint(checkpoint: str, four_mom: np.ndarray, config_path: str | None,
                        batch: int = 2048) -> np.ndarray:
    """Run a ``.pt`` checkpoint and return argmax assignment indices."""
    import torch

    from .model import JetAssignmentTransformer
    from .utils import get_config, get_device

    device = get_device()
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ckpt.get("config", get_config(config_path))
    mc, dc = config["model"], config["data"]
    model = JetAssignmentTransformer(
        d_model=mc["d_model"], nhead=mc["nhead"], num_layers=mc["num_layers"],
        dim_feedforward=mc["dim_feedforward"], dropout=mc.get("dropout", 0.1),
        num_jets=dc["num_jets"], input_dim=dc.get("input_dim", 4),
        group_num_layers=mc.get("group_num_layers", 1),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(four_mom), batch):
            chunk = torch.tensor(four_mom[i:i + batch], dtype=torch.float32, device=device)
            logits = model(chunk)["logits"]
            preds.append(logits.argmax(dim=-1).cpu().numpy())
    return np.concatenate(preds)


def _is_classical(name: str) -> bool:
    return "classical" in Path(name).name.lower()


def _load_sample(path: str, num_jets: int):
    """Load a sample twice: HT-normalised (model input) and raw (observables)."""
    norm = JetAssignmentDataset(data_paths=path, num_jets=num_jets, normalize_by_ht=True)
    raw = JetAssignmentDataset(data_paths=path, num_jets=num_jets, normalize_by_ht=False)
    assert len(norm) == len(raw), f"normalised/raw size mismatch for {path}"
    return norm.four_momenta.numpy(), raw.four_momenta.numpy()


def _observables_for(path: str, num_jets: int, *, onnx: str | None,
                     checkpoint: str | None, config: str | None,
                     input_norm: str) -> Observables:
    norm_fm, raw_fm = _load_sample(path, num_jets)

    # Decide which input the model expects.
    model_name = onnx or checkpoint or ""
    if input_norm == "auto":
        use_raw_input = _is_classical(model_name)
    else:
        use_raw_input = (input_norm == "none")
    model_input = raw_fm if use_raw_input else norm_fm

    if onnx:
        pred = _predict_onnx(onnx, model_input)
    elif checkpoint:
        pred = _predict_checkpoint(checkpoint, model_input, config)
    else:
        raise ValueError("Provide --onnx or --checkpoint.")

    return compute_observables_from_momenta(raw_fm, pred, num_jets)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_analysis(
    signal: str,
    background: str,
    output_dir: str = "results/bump_hunt",
    onnx: str | None = None,
    checkpoint: str | None = None,
    config: str | None = None,
    num_jets: int = 7,
    input_norm: str = "auto",
    cuts: dict | None = None,
    logy: bool = False,
) -> None:
    cuts = cuts or dict(DEFAULT_CUTS)
    print(f"Signal sample     : {signal}")
    print(f"Background sample : {background}")
    model_desc = onnx or checkpoint
    print(f"Model             : {model_desc}")

    sig = _observables_for(signal, num_jets, onnx=onnx, checkpoint=checkpoint,
                           config=config, input_norm=input_norm)
    bkg = _observables_for(background, num_jets, onnx=onnx, checkpoint=checkpoint,
                           config=config, input_norm=input_norm)
    print(f"Signal events: {len(sig)} | Background events: {len(bkg)}")

    rows = cutflow(sig, bkg, cuts)
    print("\n" + format_cutflow(rows) + "\n")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save the cutflow as CSV and the observable arrays for re-plotting.
    import csv
    with open(out / "cutflow.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.savez(
        out / "observables.npz",
        **{f"sig_{k}": sig.get(k) for k in
           ("avg_mass", "mass_asym", "delta_phi", "avg_boost", "dalitz_edge", "dalitz_corner")},
        **{f"bkg_{k}": bkg.get(k) for k in
           ("avg_mass", "mass_asym", "delta_phi", "avg_boost", "dalitz_edge", "dalitz_corner")},
    )

    fig_path = make_summary_figure(
        sig, bkg, cuts, out / "bump_hunt_summary.pdf",
        title="Bump-hunt cut analysis", logy=logy,
    )
    print(f"Results written to {out}/")
    print(f"  cutflow.csv              : per-stage signal/background fractions")
    print(f"  observables.npz          : per-event observable arrays")
    if fig_path:
        print(f"  bump_hunt_summary.pdf    : ROC, working points, cutflow, bump hunt")


def _build_cuts(args) -> dict:
    cuts = dict(DEFAULT_CUTS)
    cuts["delta_phi"] = ("upper", args.dphi_min)
    cuts["mass_asym"] = ("lower", args.asym_max)
    cuts["avg_boost"] = ("lower", args.boost_max)
    cuts["dalitz_corner"] = ("upper", args.dalitz_corner)
    cuts["dalitz_edge"] = ("upper", args.dalitz_edge)
    return cuts


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Standalone bump-hunt cut analysis on a trained network")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--onnx", type=str, help="Path to a trained .onnx model")
    src.add_argument("--checkpoint", type=str, help="Path to a trained .pt checkpoint")
    p.add_argument("--signal", type=str, required=True, help="Signal HDF5 (glob)")
    p.add_argument("--background", type=str, required=True, help="Background HDF5 (glob)")
    p.add_argument("--output", type=str, default="results/bump_hunt", help="Output directory")
    p.add_argument("--config", type=str, default=None, help="Config override (for --checkpoint)")
    p.add_argument("--num-jets", type=int, default=7)
    p.add_argument("--input-norm", choices=["auto", "ht", "none"], default="auto",
                   help="Input normalisation: 'ht' (ML model), 'none' (classical), "
                        "'auto' (infer from filename).")
    p.add_argument("--logy", action="store_true", help="Log-y on the bump-hunt histogram")
    # Cut thresholds (the requested nominal working point).
    p.add_argument("--dphi-min", type=float, default=2.5)
    p.add_argument("--asym-max", type=float, default=0.4)
    p.add_argument("--boost-max", type=float, default=2.0)
    p.add_argument("--dalitz-corner", type=float, default=0.0,
                   help="Keep events with median Dalitz coord > this (corner removal)")
    p.add_argument("--dalitz-edge", type=float, default=0.0,
                   help="Keep events with min Dalitz coord > this (edge-band removal)")
    args = p.parse_args()

    run_analysis(
        signal=args.signal, background=args.background, output_dir=args.output,
        onnx=args.onnx, checkpoint=args.checkpoint, config=args.config,
        num_jets=args.num_jets, input_norm=args.input_norm,
        cuts=_build_cuts(args), logy=args.logy,
    )
