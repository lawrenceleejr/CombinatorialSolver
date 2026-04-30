"""
Evaluation and mass reconstruction for the jet assignment model.

Loads a trained model, predicts jet assignments on a test set,
reconstructs invariant masses from the predicted triplets, and
outputs results for bump-hunt analysis.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .combinatorics import enumerate_assignments
from .dataset import JetAssignmentDataset
from .model import JetAssignmentTransformer, MassAsymmetryClassicalSolver
from .utils import compute_invariant_mass, get_config, get_device


def evaluate(
    checkpoint_path: str,
    data_path: str,
    output_dir: str = "results",
    config_path: str | None = None,
    include_classical: bool = True,
    physics_blend_alpha: float = 0.0,
):
    """Evaluate model and reconstruct masses.

    Args:
        checkpoint_path: Path to saved model checkpoint.
        data_path: Path to HDF5 test data (glob pattern).
        output_dir: Directory for output files.
        config_path: Optional config override.
        include_classical: Also run the staged classical solver (mass-
            difference-first with physics tie-breaks) and
            save its results alongside the ML model results.
        physics_blend_alpha: When > 0, blend the model logits with a
            confidence-weighted physics prior at inference time so that
            uncertain events favour high-asymmetry, low-mass interpretations.
            The effective logits become::

                logits_final = logits + alpha * uncertainty * physics_score

            where ``uncertainty = 1 - max(softmax(logits))`` ∈ [0, 1] and
            ``physics_score = mass_asym_flat - mass_sum_flat`` per assignment
            (both quantities in HT-normalised units).  This is a post-hoc
            push without retraining; for a training-time equivalent use the
            ``lambda_entropy_asym`` / ``lambda_entropy_mass`` config options.
            Typical values: 0.5–2.0.  Set to 0 (default) to disable.
    """
    device = get_device()
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", get_config(config_path))
    mc = config["model"]
    dc = config["data"]

    # Model
    model = JetAssignmentTransformer(
        d_model=mc["d_model"],
        nhead=mc["nhead"],
        num_layers=mc["num_layers"],
        dim_feedforward=mc["dim_feedforward"],
        dropout=mc.get("dropout", 0.1),
        num_jets=dc["num_jets"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Data — load WITHOUT HT normalization so we can compute physical masses
    dataset_raw = JetAssignmentDataset(
        data_paths=data_path,
        num_jets=dc["num_jets"],
        normalize_by_ht=False,
    )
    # Also load normalized version for model input
    dataset_norm = JetAssignmentDataset(
        data_paths=data_path,
        num_jets=dc["num_jets"],
        normalize_by_ht=dc["normalize_by_ht"],
    )

    # Ensure both datasets have same events (they should, same filtering)
    assert len(dataset_raw) == len(dataset_norm), "Dataset size mismatch"

    loader_norm = DataLoader(
        dataset_norm,
        batch_size=1024,
        shuffle=False,
        num_workers=0,
    )

    # Get all raw four-momenta for mass computation
    raw_four_mom = dataset_raw.four_momenta  # (N, 7, 4) un-normalized

    # Enumerate assignments for index lookup
    assignments = enumerate_assignments(dc["num_jets"])

    # Run ML model inference
    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in loader_norm:
            four_mom = batch["four_momenta"].to(device)
            labels = batch["label"]

            output = model(four_mom)
            logits = output["logits"]

            # Optional inference-time confidence-weighted physics blending.
            # For events where the network is uncertain (low max-softmax
            # probability), add a bias toward high-asymmetry, low-mass
            # interpretations without retraining the model.
            if physics_blend_alpha > 0.0 and "mass_asym_flat" in output:
                probs = logits.softmax(dim=-1)
                # Per-event uncertainty via 1 - max(softmax): 0 when fully
                # confident in one assignment, 1 when uniform over all.
                # This is used instead of entropy here because it produces
                # the same ordering of events by confidence but is cheaper
                # to compute (no log) and maps to [0, 1] without requiring
                # normalization by log(N_assignments).
                uncertainty = 1.0 - probs.max(dim=-1).values   # (batch,)
                # Physics score: reward high asymmetry, penalise high mass sum.
                mass_asym = output["mass_asym_flat"]            # (batch, N)
                physics_score = mass_asym
                if "mass_sum_flat" in output:
                    physics_score = physics_score - output["mass_sum_flat"]
                logits = logits + physics_blend_alpha * uncertainty.unsqueeze(-1) * physics_score

            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels)
            all_logits.append(logits.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_logits = torch.cat(all_logits)

    n_events = len(all_preds)

    # Compute ML accuracy.
    # "Full assignment accuracy": correct when argmax(logits) matches the label,
    # which encodes the complete (ISR jet, group1_jets, group2_jets) triple.
    # This is 1 only when BOTH the ISR jet AND the 3+3 grouping are right.
    correct = (all_preds == all_labels).sum().item()
    accuracy = correct / n_events
    num_assignments = all_logits.shape[1]
    topk = min(5, num_assignments)
    _, top5 = all_logits.topk(topk, dim=-1)
    acc5 = (top5 == all_labels.unsqueeze(-1)).any(dim=-1).float().mean().item()
    print(f"Num assignments: {num_assignments} ({dc['num_jets']} jets)")
    print(f"[ML]      Full assignment top-1 accuracy: {accuracy:.4f} ({correct}/{n_events})")
    print(f"[ML]      Full assignment top-{topk} accuracy: {acc5:.4f}")

    # For factored (7-jet ISR) models report ISR accuracy separately so that
    # ISR identification and grouping performance can be distinguished.
    if model.has_isr:
        truth_isr = model.flat_to_factored[all_labels, 0]   # (N,) truth ISR jet index
        # The flat-logit argmax already encodes the combined ISR+grouping prediction;
        # extract the predicted ISR by looking up its factored index.
        pred_isr = model.flat_to_factored[all_preds, 0]     # (N,) predicted ISR jet index
        isr_correct = (pred_isr == truth_isr).sum().item()
        isr_acc = isr_correct / n_events
        # Oracle grouping accuracy: given the TRUTH ISR, does the grouping head pick
        # the right 3+3 split?  This is an upper bound on the end-to-end accuracy.
        truth_grp = model.flat_to_factored[all_labels, 1]
        pred_grp  = model.flat_to_factored[all_preds,  1]
        # Grouping acc conditioned on ISR being correct
        isr_ok_mask = (pred_isr == truth_isr)
        grp_given_isr = (
            (pred_grp[isr_ok_mask] == truth_grp[isr_ok_mask]).sum().item()
            / max(isr_ok_mask.sum().item(), 1)
        )
        print(f"[ML]      ISR jet accuracy:               {isr_acc:.4f} ({isr_correct}/{n_events})")
        print(
            f"[ML]      Grouping accuracy | ISR correct: {grp_given_isr:.4f} "
            f"(non-oracle, {isr_ok_mask.sum().item()} ISR-correct events)"
        )

    # --- Staged classical solver ---
    all_classical_preds = None
    if include_classical:
        classical_solver = MassAsymmetryClassicalSolver(num_jets=dc["num_jets"])
        classical_solver.eval()

        loader_raw = DataLoader(
            dataset_raw,
            batch_size=1024,
            shuffle=False,
            num_workers=0,
        )

        classical_preds_list = []
        cls_all_logits_parts = []
        with torch.no_grad():
            for batch in loader_raw:
                four_mom_raw = batch["four_momenta"]
                cls_logits = classical_solver(four_mom_raw)["logits"]
                classical_preds_list.append(cls_logits.argmax(dim=-1))
                cls_all_logits_parts.append(cls_logits)

        all_classical_preds = torch.cat(classical_preds_list)
        cls_all_logits = torch.cat(cls_all_logits_parts)

        cls_correct = (all_classical_preds == all_labels).sum().item()
        cls_accuracy = cls_correct / n_events
        _, cls_top5_idx = cls_all_logits.topk(topk, dim=-1)
        cls_acc5 = (cls_top5_idx == all_labels.unsqueeze(-1)).any(dim=-1).float().mean().item()
        print(f"[Classical] Top-1 accuracy: {cls_accuracy:.4f} ({cls_correct}/{n_events})")
        print(f"[Classical] Top-{topk} accuracy: {cls_acc5:.4f}")

    # Reconstruct invariant masses
    mass1_pred = []  # mass of group1 in predicted assignment
    mass2_pred = []  # mass of group2 in predicted assignment
    mass_avg_pred = []  # average of the two (for bump hunt)
    mass1_truth = []
    mass2_truth = []
    mass_avg_truth = []
    # Classical solver mass arrays (populated only if include_classical=True)
    mass1_classical = []
    mass2_classical = []
    mass_avg_classical = []

    for i in range(n_events):
        p_idx = all_preds[i].item()
        t_idx = all_labels[i].item()

        raw = raw_four_mom[i]  # (num_jets, 4) un-normalized E, px, py, pz

        # Predicted assignment — unpack (isr_or_None, g1, g2)
        _, g1_pred_idx, g2_pred_idx = assignments[p_idx]
        p_g1 = raw[list(g1_pred_idx)].sum(dim=0)
        p_g2 = raw[list(g2_pred_idx)].sum(dim=0)
        m1_p = compute_invariant_mass(p_g1).item()
        m2_p = compute_invariant_mass(p_g2).item()
        mass1_pred.append(m1_p)
        mass2_pred.append(m2_p)
        mass_avg_pred.append((m1_p + m2_p) / 2.0)

        # Truth assignment
        if t_idx >= 0:
            _, g1_truth_idx, g2_truth_idx = assignments[t_idx]
            t_g1 = raw[list(g1_truth_idx)].sum(dim=0)
            t_g2 = raw[list(g2_truth_idx)].sum(dim=0)
            m1_t = compute_invariant_mass(t_g1).item()
            m2_t = compute_invariant_mass(t_g2).item()
            mass1_truth.append(m1_t)
            mass2_truth.append(m2_t)
            mass_avg_truth.append((m1_t + m2_t) / 2.0)
        else:
            mass1_truth.append(float("nan"))
            mass2_truth.append(float("nan"))
            mass_avg_truth.append(float("nan"))

        # Classical assignment masses
        if all_classical_preds is not None:
            c_idx = all_classical_preds[i].item()
            _, g1_cls_idx, g2_cls_idx = assignments[c_idx]
            c_g1 = raw[list(g1_cls_idx)].sum(dim=0)
            c_g2 = raw[list(g2_cls_idx)].sum(dim=0)
            m1_c = compute_invariant_mass(c_g1).item()
            m2_c = compute_invariant_mass(c_g2).item()
            mass1_classical.append(m1_c)
            mass2_classical.append(m2_c)
            mass_avg_classical.append((m1_c + m2_c) / 2.0)

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Pre-compute per-event ISR correct flag (only for factored models).
    has_isr_model = model.has_isr
    if has_isr_model:
        _pred_isr_all = model.flat_to_factored[all_preds, 0]
        _truth_isr_all = model.flat_to_factored[all_labels, 0]
        isr_correct_per_event = (_pred_isr_all == _truth_isr_all).numpy()
    else:
        isr_correct_per_event = None

    # Mass distributions as CSV
    csv_path = Path(output_dir) / "mass_reconstruction.csv"
    has_classical = all_classical_preds is not None
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        # "correct" = full assignment correct (ISR jet + grouping both right).
        # "isr_correct" = only the ISR jet identification is right (factored models).
        header = [
            "event_idx", "pred_assignment", "truth_assignment",
            "correct",
        ]
        if has_isr_model:
            header += ["isr_correct"]
        header += [
            "mass1_pred", "mass2_pred", "mass_avg_pred",
            "mass1_truth", "mass2_truth", "mass_avg_truth",
        ]
        if has_classical:
            header += [
                "classical_assignment", "classical_correct",
                "mass1_classical", "mass2_classical", "mass_avg_classical",
            ]
        writer.writerow(header)
        for i in range(n_events):
            row = [
                i, all_preds[i].item(), all_labels[i].item(),
                int(all_preds[i].item() == all_labels[i].item()),
            ]
            if has_isr_model:
                row += [int(isr_correct_per_event[i])]
            row += [
                f"{mass1_pred[i]:.2f}", f"{mass2_pred[i]:.2f}", f"{mass_avg_pred[i]:.2f}",
                f"{mass1_truth[i]:.2f}", f"{mass2_truth[i]:.2f}", f"{mass_avg_truth[i]:.2f}",
            ]
            if has_classical:
                row += [
                    all_classical_preds[i].item(),
                    int(all_classical_preds[i].item() == all_labels[i].item()),
                    f"{mass1_classical[i]:.2f}",
                    f"{mass2_classical[i]:.2f}",
                    f"{mass_avg_classical[i]:.2f}",
                ]
            writer.writerow(row)

    # Summary numpy arrays (for quick plotting).
    # ``correct``     : bool array — full assignment correct (ISR + grouping).
    # ``isr_correct`` : bool array — only ISR jet correct (factored models only).
    npz_kwargs = dict(
        mass_avg_pred=np.array(mass_avg_pred),
        mass_avg_truth=np.array(mass_avg_truth),
        mass1_pred=np.array(mass1_pred),
        mass2_pred=np.array(mass2_pred),
        correct=np.array([all_preds[i].item() == all_labels[i].item() for i in range(n_events)]),
    )
    if has_isr_model:
        npz_kwargs["isr_correct"] = isr_correct_per_event
    if has_classical:
        npz_kwargs["mass_avg_classical"] = np.array(mass_avg_classical)
        npz_kwargs["mass1_classical"] = np.array(mass1_classical)
        npz_kwargs["mass2_classical"] = np.array(mass2_classical)
        npz_kwargs["correct_classical"] = np.array(
            [all_classical_preds[i].item() == all_labels[i].item() for i in range(n_events)]
        )
    np.savez(Path(output_dir) / "mass_arrays.npz", **npz_kwargs)

    print(f"\nResults saved to {output_dir}/")
    print(f"  mass_reconstruction.csv: per-event results")
    print(f"  mass_arrays.npz: numpy arrays for plotting")
    print(f"\nMass avg (ML predicted): mean={np.nanmean(mass_avg_pred):.1f} GeV, "
          f"std={np.nanstd(mass_avg_pred):.1f} GeV")
    if has_classical:
        print(f"Mass avg (classical):     mean={np.nanmean(mass_avg_classical):.1f} GeV, "
              f"std={np.nanstd(mass_avg_classical):.1f} GeV")
    print(f"Mass avg (truth):         mean={np.nanmean(mass_avg_truth):.1f} GeV, "
          f"std={np.nanstd(mass_avg_truth):.1f} GeV")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate jet assignment model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--data", type=str, required=True, help="Test HDF5 data (glob)")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--config", type=str, default=None, help="Config override")
    parser.add_argument(
        "--no-classical",
        action="store_true",
        help="Skip the staged classical solver comparison",
    )
    parser.add_argument(
        "--physics-blend-alpha",
        type=float,
        default=0.0,
        help=(
            "Inference-time physics blending strength (default: 0 = disabled). "
            "When > 0, uncertain events are pushed toward high-asymmetry, "
            "low-mass assignments without retraining. Typical values: 0.5–2.0."
        ),
    )
    args = parser.parse_args()
    evaluate(
        args.checkpoint, args.data, args.output, args.config,
        not args.no_classical,
        physics_blend_alpha=args.physics_blend_alpha,
    )
