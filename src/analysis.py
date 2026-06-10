"""
Standalone bump-hunt analysis on a *final* trained network.

Loads a trained model (ONNX via onnxruntime, or a ``.pt`` checkpoint), runs it
over a signal and a background HDF5 sample, picks each event's interpretation,
computes the analysis observables on the chosen triplets, and produces:

  - a cutflow table (fraction of signal and background remaining at each cut);
  - ROC scans for every cut variable plus a cloud of many combined working
    points with the Pareto front;
  - the average-mass bump-hunt histogram before/after the cuts.

The per-event ``observables.npz`` written here is the input to the next stages
of the search: ``src.background_estimate`` (chi-sideband QCD transfer +
closure) and ``src.bump_hunt`` (pyhf p0 scan / limits / signal injection).

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
``--soft-mass-topk K`` switches m_avg to the probability-weighted mean over the
network's top-K assignments (sharper signal peak on combinatorially ambiguous
events).  When the model carries a trained score head, the per-event NN signal
score is exported alongside the observables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .combinatorics import enumerate_assignments
from .cut_analysis import DEFAULT_CUTS, Observables, cutflow, format_cutflow
from .cut_plots import make_summary_figure
from .dataset import JetAssignmentDataset

# Observable keys exported to observables.npz (score handled separately since
# it may be absent for models without a trained score head).
NPZ_KEYS = (
    "avg_mass", "mass_asym", "delta_phi", "avg_boost",
    "dalitz_edge", "dalitz_corner", "y_star", "chi",
)


# --------------------------------------------------------------------------- #
# Observable computation from raw four-momenta + chosen assignment
# --------------------------------------------------------------------------- #
def _inv_mass(p: np.ndarray) -> np.ndarray:
    m2 = p[..., 0] ** 2 - p[..., 1] ** 2 - p[..., 2] ** 2 - p[..., 3] ** 2
    return np.sqrt(np.clip(m2, 0.0, None))


def _rapidity(p: np.ndarray) -> np.ndarray:
    """Rapidity y = 0.5 ln((E+pz)/(E-pz)) of four-vectors ``(..., 4)``."""
    E, pz = p[..., 0], p[..., 3]
    return 0.5 * np.log(np.clip(E + pz, 1e-9, None) / np.clip(E - pz, 1e-9, None))


def soft_avg_mass(
    raw_four_mom: np.ndarray,
    logits: np.ndarray,
    num_jets: int,
    k: int,
) -> np.ndarray:
    """Probability-weighted average candidate mass over the top-k assignments.

    Instead of committing to the argmax interpretation, m_avg is averaged over
    the network's k most probable assignments with softmax weights
    (renormalised over the k).  Combinatorially ambiguous events — where the
    argmax is a coin flip between near-degenerate interpretations — then
    contribute a smeared-but-centred value rather than a randomly wrong one,
    sharpening the signal peak; confident events (near-one-hot softmax) are
    unchanged.
    """
    assignments = enumerate_assignments(num_jets)
    g1_arr = np.array([list(a[1]) for a in assignments], dtype=int)
    g2_arr = np.array([list(a[2]) for a in assignments], dtype=int)

    N = raw_four_mom.shape[0]
    k = min(k, logits.shape[1])
    topk_idx = np.argpartition(-logits, k - 1, axis=1)[:, :k]            # (N, k)
    topk_logits = np.take_along_axis(logits, topk_idx, axis=1)
    topk_logits = topk_logits - topk_logits.max(axis=1, keepdims=True)
    w = np.exp(topk_logits)
    w /= w.sum(axis=1, keepdims=True)                                     # (N, k)

    rows = np.arange(N)[:, None, None]
    j1 = raw_four_mom[rows, g1_arr[topk_idx]]    # (N, k, 3, 4)
    j2 = raw_four_mom[rows, g2_arr[topk_idx]]
    m1 = _inv_mass(j1.sum(axis=2))               # (N, k)
    m2 = _inv_mass(j2.sum(axis=2))
    return ((m1 + m2) / 2.0 * w).sum(axis=1)


def compute_observables_from_momenta(
    raw_four_mom: np.ndarray,
    pred_idx: np.ndarray,
    num_jets: int,
    weight: np.ndarray | None = None,
    score: np.ndarray | None = None,
    logits: np.ndarray | None = None,
    soft_mass_topk: int = 0,
) -> Observables:
    """Compute the analysis observables for each event's chosen interpretation.

    Args:
        raw_four_mom: ``(N, num_jets, 4)`` un-normalised (E, px, py, pz) [GeV].
        pred_idx: ``(N,)`` predicted assignment index into
            :func:`enumerate_assignments`.
        num_jets: 6 or 7.
        weight: optional ``(N,)`` per-event weights.
        score: optional ``(N,)`` NN signal score (sigmoid of the score head).
        logits: optional ``(N, n_assign)`` assignment logits (needed for
            ``soft_mass_topk``).
        soft_mass_topk: when > 0 (and logits given), ``avg_mass`` becomes the
            probability-weighted top-k mean (:func:`soft_avg_mass`); the argmax
            value is kept in ``extra["avg_mass_argmax"]``.
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

    extra: dict = {}
    if soft_mass_topk > 0 and logits is not None:
        extra["avg_mass_argmax"] = avg_mass
        avg_mass = soft_avg_mass(raw_four_mom, logits, num_jets, soft_mass_topk)

    # Δφ between the two parent sums, folded to [0, π].
    phi1 = np.arctan2(p1[:, 2], p1[:, 1])
    phi2 = np.arctan2(p2[:, 2], p2[:, 1])
    dphi = np.abs(phi1 - phi2)
    dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)

    # Rapidity-separation variables of the two triplets: y* = |y1-y2|/2 and
    # chi = exp(2 y*).  QCD's t-channel angular shape in chi is approximately
    # independent of the mass scale (QCD scale invariance), which is what the
    # chi-sideband background transfer (src.background_estimate) relies on;
    # pair-produced signal is central (low y*).
    y_star = 0.5 * np.abs(_rapidity(p1) - _rapidity(p2))
    chi = np.exp(2.0 * y_star)

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
        extra=extra,
        y_star=y_star, chi=chi,
        score=None if score is None else np.asarray(score, dtype=float),
    )


# --------------------------------------------------------------------------- #
# Inference back-ends — both return (logits, score_logit_or_None)
# --------------------------------------------------------------------------- #
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _predict_onnx(
    onnx_path: str, four_mom: np.ndarray, batch: int = 2048
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run an ONNX model; returns (logits, score_logit or None)."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    has_score = "score_logit" in out_names
    fetch = ["logits", "score_logit"] if has_score and "logits" in out_names else [out_names[0]]

    logits_parts, score_parts = [], []
    for i in range(0, len(four_mom), batch):
        chunk = four_mom[i:i + batch].astype(np.float32)
        outs = sess.run(fetch, {in_name: chunk})
        logits_parts.append(outs[0])
        if has_score:
            score_parts.append(outs[1])
    logits = np.concatenate(logits_parts)
    score = np.concatenate(score_parts).reshape(-1) if score_parts else None
    return logits, score


def _predict_checkpoint(
    checkpoint: str, four_mom: np.ndarray, config_path: str | None, batch: int = 2048
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run a ``.pt`` checkpoint; returns (logits, score_logit or None)."""
    import torch

    from .model import JetAssignmentTransformer
    from .utils import get_config, get_device, load_compatible_state_dict

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
    load_compatible_state_dict(model, ckpt["model_state_dict"])
    model.eval()
    logits_parts, score_parts = [], []
    with torch.no_grad():
        for i in range(0, len(four_mom), batch):
            chunk = torch.tensor(four_mom[i:i + batch], dtype=torch.float32, device=device)
            out = model(chunk)
            logits_parts.append(out["logits"].cpu().numpy())
            if "score_logit" in out:
                score_parts.append(out["score_logit"].cpu().numpy())
    logits = np.concatenate(logits_parts)
    score = np.concatenate(score_parts).reshape(-1) if score_parts else None
    return logits, score


def _is_classical(name: str) -> bool:
    return "classical" in Path(name).name.lower()


def _load_sample(path: str, num_jets: int):
    """Load a sample twice: HT-normalised (model input) and raw (observables).

    Returns (normalised four-momenta, raw four-momenta, per-event weights).
    """
    norm = JetAssignmentDataset(data_paths=path, num_jets=num_jets, normalize_by_ht=True)
    raw = JetAssignmentDataset(data_paths=path, num_jets=num_jets, normalize_by_ht=False)
    assert len(norm) == len(raw), f"normalised/raw size mismatch for {path}"
    return norm.four_momenta.numpy(), raw.four_momenta.numpy(), raw.weights.numpy()


def _observables_for(path: str, num_jets: int, *, onnx: str | None,
                     checkpoint: str | None, config: str | None,
                     input_norm: str, soft_mass_topk: int = 0) -> Observables:
    norm_fm, raw_fm, weights = _load_sample(path, num_jets)

    # Decide which input the model expects.
    model_name = onnx or checkpoint or ""
    if input_norm == "auto":
        use_raw_input = _is_classical(model_name)
    else:
        use_raw_input = (input_norm == "none")
    model_input = raw_fm if use_raw_input else norm_fm

    if onnx:
        logits, score_logit = _predict_onnx(onnx, model_input)
    elif checkpoint:
        logits, score_logit = _predict_checkpoint(checkpoint, model_input, config)
    else:
        raise ValueError("Provide --onnx or --checkpoint.")

    pred = logits.argmax(axis=-1)
    score = _sigmoid(score_logit) if score_logit is not None else None
    return compute_observables_from_momenta(
        raw_fm, pred, num_jets, weight=weights, score=score,
        logits=logits, soft_mass_topk=soft_mass_topk,
    )


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
    soft_mass_topk: int = 0,
) -> None:
    cuts = cuts or dict(DEFAULT_CUTS)
    print(f"Signal sample     : {signal}")
    print(f"Background sample : {background}")
    model_desc = onnx or checkpoint
    print(f"Model             : {model_desc}")
    if soft_mass_topk > 0:
        print(f"m_avg definition  : soft (probability-weighted top-{soft_mass_topk})")

    sig = _observables_for(signal, num_jets, onnx=onnx, checkpoint=checkpoint,
                           config=config, input_norm=input_norm,
                           soft_mass_topk=soft_mass_topk)
    bkg = _observables_for(background, num_jets, onnx=onnx, checkpoint=checkpoint,
                           config=config, input_norm=input_norm,
                           soft_mass_topk=soft_mass_topk)
    print(f"Signal events: {len(sig)} | Background events: {len(bkg)}")
    if sig.score is not None:
        print("NN signal score   : available "
              f"(sig mean {sig.score.mean():.3f}, bkg mean {bkg.score.mean():.3f})")

    rows = cutflow(sig, bkg, cuts)
    print("\n" + format_cutflow(rows) + "\n")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save the cutflow as CSV and the observable arrays for re-plotting and for
    # the downstream background-estimate / bump-hunt stages.
    import csv
    with open(out / "cutflow.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    npz_payload = {}
    for tag, obs in (("sig", sig), ("bkg", bkg)):
        for k in NPZ_KEYS:
            npz_payload[f"{tag}_{k}"] = obs.get(k)
        npz_payload[f"{tag}_weight"] = obs.weights
        if obs.score is not None:
            npz_payload[f"{tag}_score"] = obs.score
    np.savez(out / "observables.npz", **npz_payload)

    fig_path = make_summary_figure(
        sig, bkg, cuts, out / "bump_hunt_summary.pdf",
        title="Bump-hunt cut analysis", logy=logy,
    )
    print(f"Results written to {out}/")
    print(f"  cutflow.csv              : per-stage signal/background fractions")
    print(f"  observables.npz          : per-event observable arrays "
          f"(input to src.background_estimate / src.bump_hunt)")
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
    p.add_argument("--soft-mass-topk", type=int, default=0,
                   help="If > 0, m_avg = probability-weighted mean over the top-K "
                        "assignments instead of the argmax (default: 0 = argmax).")
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
        soft_mass_topk=args.soft_mass_topk,
    )
