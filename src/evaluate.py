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
from .model import JetAssignmentTransformer
from .utils import compute_invariant_mass, get_config, get_device


def evaluate(
    checkpoint_path: str,
    data_path: str,
    output_dir: str = "results",
    config_path: str | None = None,
):
    """Evaluate model and reconstruct masses.

    Args:
        checkpoint_path: Path to saved model checkpoint.
        data_path: Path to HDF5 test data (glob pattern).
        output_dir: Directory for output files.
        config_path: Optional config override.
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

    # Run inference
    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in loader_norm:
            four_mom = batch["four_momenta"].to(device)
            labels = batch["label"]

            output = model(four_mom)
            logits = output["logits"]
            preds = logits.argmax(dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels)
            all_logits.append(logits.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_logits = torch.cat(all_logits)

    n_events = len(all_preds)

    # Compute accuracy
    correct = (all_preds == all_labels).sum().item()
    accuracy = correct / n_events
    num_assignments = all_logits.shape[1]
    topk = min(5, num_assignments)
    _, top5 = all_logits.topk(topk, dim=-1)
    acc5 = (top5 == all_labels.unsqueeze(-1)).any(dim=-1).float().mean().item()
    print(f"Num assignments: {num_assignments} ({dc['num_jets']} jets)")
    print(f"Top-1 accuracy: {accuracy:.4f} ({correct}/{n_events})")
    print(f"Top-{topk} accuracy: {acc5:.4f}")

    # Reconstruct invariant masses
    mass1_pred = []  # mass of group1 in predicted assignment
    mass2_pred = []  # mass of group2 in predicted assignment
    mass_avg_pred = []  # average of the two (for bump hunt)
    mass1_truth = []
    mass2_truth = []
    mass_avg_truth = []

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

    # Save results
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Mass distributions as CSV
    csv_path = Path(output_dir) / "mass_reconstruction.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event_idx", "pred_assignment", "truth_assignment", "correct",
            "mass1_pred", "mass2_pred", "mass_avg_pred",
            "mass1_truth", "mass2_truth", "mass_avg_truth",
        ])
        for i in range(n_events):
            writer.writerow([
                i, all_preds[i].item(), all_labels[i].item(),
                int(all_preds[i].item() == all_labels[i].item()),
                f"{mass1_pred[i]:.2f}", f"{mass2_pred[i]:.2f}", f"{mass_avg_pred[i]:.2f}",
                f"{mass1_truth[i]:.2f}", f"{mass2_truth[i]:.2f}", f"{mass_avg_truth[i]:.2f}",
            ])

    # Summary numpy arrays (for quick plotting)
    np.savez(
        Path(output_dir) / "mass_arrays.npz",
        mass_avg_pred=np.array(mass_avg_pred),
        mass_avg_truth=np.array(mass_avg_truth),
        mass1_pred=np.array(mass1_pred),
        mass2_pred=np.array(mass2_pred),
        correct=np.array([all_preds[i].item() == all_labels[i].item() for i in range(n_events)]),
    )

    print(f"\nResults saved to {output_dir}/")
    print(f"  mass_reconstruction.csv: per-event results")
    print(f"  mass_arrays.npz: numpy arrays for plotting")
    print(f"\nMass avg (predicted): mean={np.nanmean(mass_avg_pred):.1f} GeV, "
          f"std={np.nanstd(mass_avg_pred):.1f} GeV")
    print(f"Mass avg (truth):     mean={np.nanmean(mass_avg_truth):.1f} GeV, "
          f"std={np.nanstd(mass_avg_truth):.1f} GeV")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate jet assignment model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--data", type=str, required=True, help="Test HDF5 data (glob)")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--config", type=str, default=None, help="Config override")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data, args.output, args.config)
