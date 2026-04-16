"""
Benchmark: mass-asymmetry baseline vs transformer model with HP scan.

Mass asymmetry baseline: for each event, pick the assignment that minimizes
|m1 - m2| / (m1 + m2) where m1, m2 are the invariant masses of the two triplets.
This is the naive physics-motivated approach with zero ML.
"""

import csv
import itertools
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.combinatorics import enumerate_assignments
from src.dataset import JetAssignmentDataset
from src.model import JetAssignmentTransformer
from src.utils import get_device


# ---------------------------------------------------------------------------
# Mass asymmetry baseline
# ---------------------------------------------------------------------------

def invariant_mass_np(four_mom):
    """Compute invariant mass from (E, px, py, pz) numpy array."""
    e, px, py, pz = four_mom[..., 0], four_mom[..., 1], four_mom[..., 2], four_mom[..., 3]
    m2 = e**2 - px**2 - py**2 - pz**2
    return np.sqrt(np.maximum(m2, 0.0))


def mass_asymmetry_baseline(dataset, num_jets=6):
    """Evaluate the mass-asymmetry baseline on the full dataset.

    For each event, pick the assignment minimizing |m1 - m2| / (m1 + m2).
    Also evaluate several other simple heuristics for comparison.

    Returns dict of {method_name: accuracy}.
    """
    assignments = enumerate_assignments(num_jets)
    # Get un-normalized four-momenta
    raw_four_mom = dataset.four_momenta.numpy()
    # If data was HT-normalized, undo it
    if dataset.normalize_by_ht:
        ht = dataset.ht.numpy()[:, None, None]
        raw_four_mom = raw_four_mom * ht

    labels = dataset.labels.numpy()
    n_events = len(labels)

    # Precompute group masses for all assignments and all events
    # assignments[i] = (isr_or_None, g1_tuple, g2_tuple)
    n_assign = len(assignments)
    g1_indices = [list(a[1]) for a in assignments]
    g2_indices = [list(a[2]) for a in assignments]

    correct_min_asym = 0
    correct_min_diff = 0
    correct_max_sum = 0
    correct_min_ratio = 0
    correct_random = 0

    rng = np.random.RandomState(42)

    for i in range(n_events):
        fmom = raw_four_mom[i]  # (num_jets, 4)
        truth = labels[i]

        best_asym = float("inf")
        best_asym_idx = -1
        best_diff = float("inf")
        best_diff_idx = -1
        best_sum = -float("inf")
        best_sum_idx = -1
        best_ratio = float("inf")
        best_ratio_idx = -1

        for j in range(n_assign):
            g1_4vec = fmom[g1_indices[j]].sum(axis=0)
            g2_4vec = fmom[g2_indices[j]].sum(axis=0)
            m1 = invariant_mass_np(g1_4vec)
            m2 = invariant_mass_np(g2_4vec)

            # Mass asymmetry: |m1 - m2| / (m1 + m2)
            denom = m1 + m2
            if denom > 0:
                asym = abs(m1 - m2) / denom
            else:
                asym = float("inf")

            if asym < best_asym:
                best_asym = asym
                best_asym_idx = j

            # Absolute mass difference
            diff = abs(m1 - m2)
            if diff < best_diff:
                best_diff = diff
                best_diff_idx = j

            # Maximum total mass (sum)
            total = m1 + m2
            if total > best_sum:
                best_sum = total
                best_sum_idx = j

            # Min mass ratio (closer to 1 is better)
            if min(m1, m2) > 0:
                ratio = max(m1, m2) / min(m1, m2)
            else:
                ratio = float("inf")
            if ratio < best_ratio:
                best_ratio = ratio
                best_ratio_idx = j

        if best_asym_idx == truth:
            correct_min_asym += 1
        if best_diff_idx == truth:
            correct_min_diff += 1
        if best_sum_idx == truth:
            correct_max_sum += 1
        if best_ratio_idx == truth:
            correct_min_ratio += 1
        if rng.randint(0, n_assign) == truth:
            correct_random += 1

    return {
        "random_chance": correct_random / n_events,
        "min_mass_asymmetry": correct_min_asym / n_events,
        "min_mass_difference": correct_min_diff / n_events,
        "max_mass_sum": correct_max_sum / n_events,
        "min_mass_ratio": correct_min_ratio / n_events,
        "theoretical_random": 1.0 / n_assign,
    }


# ---------------------------------------------------------------------------
# Training function for HP scan
# ---------------------------------------------------------------------------

def train_with_config(
    train_set,
    val_set,
    hp: dict,
    num_jets: int,
    device: torch.device,
    num_epochs: int = 100,
    patience: int = 20,
    verbose: bool = False,
):
    """Train a model with given hyperparameters, return best val accuracy.

    Uses early stopping based on validation accuracy plateau.
    """
    train_loader = DataLoader(
        train_set,
        batch_size=hp["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=hp["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = JetAssignmentTransformer(
        d_model=hp["d_model"],
        nhead=hp["nhead"],
        num_layers=hp["num_layers"],
        dim_feedforward=hp["dim_feedforward"],
        dropout=hp["dropout"],
        num_jets=num_jets,
    ).to(device)

    # Disable adversary for HP scan (single mass point)
    model.gradient_reversal.set_lambda(0.0)

    total_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
    )
    for pg in optimizer.param_groups:
        pg["initial_lr"] = pg["lr"]

    ce_loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_epoch = 0
    epochs_since_best = 0
    history = []

    t0 = time.time()

    for epoch in range(num_epochs):
        # LR schedule: linear warmup then cosine decay
        warmup = hp.get("warmup_epochs", 5)
        if epoch < warmup:
            lr_scale = (epoch + 1) / warmup
        else:
            progress = (epoch - warmup) / max(num_epochs - warmup, 1)
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = pg["initial_lr"] * lr_scale

        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            four_mom = batch["four_momenta"].to(device)
            labels = batch["label"].to(device)

            output = model(four_mom)
            loss = ce_loss_fn(output["logits"], labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = labels.shape[0]
            train_loss += loss.item() * bs
            train_correct += (output["logits"].argmax(dim=-1) == labels).sum().item()
            train_total += bs

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_correct5 = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                four_mom = batch["four_momenta"].to(device)
                labels = batch["label"].to(device)

                output = model(four_mom)
                loss = ce_loss_fn(output["logits"], labels)

                bs = labels.shape[0]
                val_loss += loss.item() * bs
                val_correct += (output["logits"].argmax(dim=-1) == labels).sum().item()
                val_total += bs

                _, top5 = output["logits"].topk(min(5, output["logits"].shape[1]), dim=-1)
                val_correct5 += (top5 == labels.unsqueeze(-1)).any(dim=-1).sum().item()

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        val_acc5 = val_correct5 / max(val_total, 1)
        avg_train_loss = train_loss / max(train_total, 1)
        avg_val_loss = val_loss / max(val_total, 1)

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc": train_acc,
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            "val_acc5": val_acc5,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
                  f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} "
                  f"val_acc5={val_acc5:.3f} | best={best_val_acc:.3f}")

        # Early stopping
        if epochs_since_best >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    elapsed = time.time() - t0

    return {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "final_train_acc": history[-1]["train_acc"],
        "final_val_acc": history[-1]["val_acc"],
        "final_val_acc5": history[-1]["val_acc5"],
        "total_params": total_params,
        "elapsed_seconds": elapsed,
        "epochs_trained": len(history),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_path = "data/gluino_rpv_1tev_uds_10000evt_20260225_164524.h5"
    num_jets = 6
    device = get_device()
    print(f"Device: {device}")

    # Load dataset (un-normalized for baseline, we'll handle normalization per-model)
    print("\nLoading dataset...")
    dataset_raw = JetAssignmentDataset(
        data_paths=data_path,
        num_jets=num_jets,
        normalize_by_ht=False,
    )
    dataset_norm = JetAssignmentDataset(
        data_paths=data_path,
        num_jets=num_jets,
        normalize_by_ht=True,
    )
    print(f"Dataset: {len(dataset_norm)} events, {num_jets} jets, "
          f"{len(enumerate_assignments(num_jets))} possible assignments")

    # Fixed train/val split (same as training script)
    n_val = max(1, int(0.1 * len(dataset_norm)))
    n_train = len(dataset_norm) - n_val
    gen = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset_norm, [n_train, n_val], generator=gen)

    # Also split the raw dataset with the same indices for baseline
    gen_raw = torch.Generator().manual_seed(42)
    train_set_raw, val_set_raw = random_split(dataset_raw, [n_train, n_val], generator=gen_raw)

    # -----------------------------------------------------------------------
    # 1. Mass asymmetry baseline (on val set)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MASS ASYMMETRY BASELINE (no ML)")
    print("=" * 70)

    # Build a mini-dataset from val indices for baseline eval
    val_indices = val_set_raw.indices
    val_subset_raw = Subset(dataset_raw, val_indices)

    # For baseline, we need direct access to four_momenta and labels
    val_four_mom = dataset_raw.four_momenta[val_indices].numpy()
    val_labels = dataset_raw.labels[val_indices].numpy()

    assignments = enumerate_assignments(num_jets)
    n_assign = len(assignments)
    g1_indices = [list(a[1]) for a in assignments]
    g2_indices = [list(a[2]) for a in assignments]

    n_val_events = len(val_labels)

    # Evaluate multiple simple baselines
    correct = {
        "min_mass_asymmetry": 0,
        "min_mass_difference": 0,
        "max_mass_sum": 0,
        "min_mass_ratio": 0,
    }

    for i in range(n_val_events):
        fmom = val_four_mom[i]
        truth = val_labels[i]

        scores = {"min_mass_asymmetry": [], "min_mass_difference": [],
                  "max_mass_sum": [], "min_mass_ratio": []}

        for j in range(n_assign):
            g1_4vec = fmom[g1_indices[j]].sum(axis=0)
            g2_4vec = fmom[g2_indices[j]].sum(axis=0)
            m1 = invariant_mass_np(g1_4vec)
            m2 = invariant_mass_np(g2_4vec)

            denom = m1 + m2
            asym = abs(m1 - m2) / denom if denom > 0 else float("inf")
            scores["min_mass_asymmetry"].append(asym)
            scores["min_mass_difference"].append(abs(m1 - m2))
            scores["max_mass_sum"].append(-(m1 + m2))  # negative for argmin
            scores["min_mass_ratio"].append(
                max(m1, m2) / min(m1, m2) if min(m1, m2) > 0 else float("inf")
            )

        for method in correct:
            pred = int(np.argmin(scores[method]))
            if pred == truth:
                correct[method] += 1

    print(f"\nVal set: {n_val_events} events, {n_assign} assignments")
    print(f"Random chance: {1.0/n_assign:.1%}")
    print()
    for method, c in correct.items():
        acc = c / n_val_events
        print(f"  {method:25s}: {acc:.4f} ({acc:.1%})")

    baseline_acc = correct["min_mass_asymmetry"] / n_val_events

    # Also compute on full dataset for reference
    print("\n--- Full dataset baselines ---")
    full_results = mass_asymmetry_baseline(dataset_raw, num_jets)
    for method, acc in full_results.items():
        print(f"  {method:25s}: {acc:.4f} ({acc:.1%})")

    # -----------------------------------------------------------------------
    # 2. Hyperparameter scan
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("HYPERPARAMETER SCAN")
    print("=" * 70)

    hp_configs = [
        # Baseline config (what we've been using)
        {
            "name": "baseline_128d_4L",
            "d_model": 128, "nhead": 8, "num_layers": 4,
            "dim_feedforward": 256, "dropout": 0.1,
            "learning_rate": 3e-4, "weight_decay": 1e-4,
            "batch_size": 256, "warmup_epochs": 5,
        },
        # Smaller model
        {
            "name": "small_64d_2L",
            "d_model": 64, "nhead": 4, "num_layers": 2,
            "dim_feedforward": 128, "dropout": 0.1,
            "learning_rate": 3e-4, "weight_decay": 1e-4,
            "batch_size": 256, "warmup_epochs": 5,
        },
        # Wider model
        {
            "name": "wide_256d_4L",
            "d_model": 256, "nhead": 8, "num_layers": 4,
            "dim_feedforward": 512, "dropout": 0.1,
            "learning_rate": 1e-4, "weight_decay": 1e-4,
            "batch_size": 256, "warmup_epochs": 5,
        },
        # Deeper model
        {
            "name": "deep_128d_8L",
            "d_model": 128, "nhead": 8, "num_layers": 8,
            "dim_feedforward": 256, "dropout": 0.1,
            "learning_rate": 1e-4, "weight_decay": 1e-4,
            "batch_size": 256, "warmup_epochs": 5,
        },
        # Higher LR
        {
            "name": "highLR_128d_4L",
            "d_model": 128, "nhead": 8, "num_layers": 4,
            "dim_feedforward": 256, "dropout": 0.1,
            "learning_rate": 1e-3, "weight_decay": 1e-4,
            "batch_size": 256, "warmup_epochs": 10,
        },
        # More dropout
        {
            "name": "highdrop_128d_4L",
            "d_model": 128, "nhead": 8, "num_layers": 4,
            "dim_feedforward": 256, "dropout": 0.2,
            "learning_rate": 3e-4, "weight_decay": 1e-4,
            "batch_size": 256, "warmup_epochs": 5,
        },
        # Lower dropout
        {
            "name": "lowdrop_128d_4L",
            "d_model": 128, "nhead": 8, "num_layers": 4,
            "dim_feedforward": 256, "dropout": 0.05,
            "learning_rate": 3e-4, "weight_decay": 1e-4,
            "batch_size": 256, "warmup_epochs": 5,
        },
        # Wider FFN
        {
            "name": "wideffn_128d_4L",
            "d_model": 128, "nhead": 8, "num_layers": 4,
            "dim_feedforward": 512, "dropout": 0.1,
            "learning_rate": 3e-4, "weight_decay": 1e-4,
            "batch_size": 256, "warmup_epochs": 5,
        },
        # Bigger batch
        {
            "name": "bigbatch_128d_4L",
            "d_model": 128, "nhead": 8, "num_layers": 4,
            "dim_feedforward": 256, "dropout": 0.1,
            "learning_rate": 5e-4, "weight_decay": 1e-4,
            "batch_size": 512, "warmup_epochs": 5,
        },
        # Sweet spot candidate: 192d, 6 layers
        {
            "name": "sweet_192d_6L",
            "d_model": 192, "nhead": 8, "num_layers": 6,
            "dim_feedforward": 384, "dropout": 0.1,
            "learning_rate": 2e-4, "weight_decay": 1e-4,
            "batch_size": 256, "warmup_epochs": 5,
        },
        # Minimal: tiny model to see floor
        {
            "name": "tiny_32d_1L",
            "d_model": 32, "nhead": 4, "num_layers": 1,
            "dim_feedforward": 64, "dropout": 0.1,
            "learning_rate": 5e-4, "weight_decay": 1e-4,
            "batch_size": 256, "warmup_epochs": 3,
        },
        # Higher weight decay (regularization)
        {
            "name": "highreg_128d_4L",
            "d_model": 128, "nhead": 8, "num_layers": 4,
            "dim_feedforward": 256, "dropout": 0.1,
            "learning_rate": 3e-4, "weight_decay": 1e-3,
            "batch_size": 256, "warmup_epochs": 5,
        },
    ]

    num_epochs = 100  # Enough for convergence comparison
    results = []

    for i, hp in enumerate(hp_configs):
        name = hp["name"]
        print(f"\n[{i+1}/{len(hp_configs)}] {name}")
        print(f"  d_model={hp['d_model']} nhead={hp['nhead']} layers={hp['num_layers']} "
              f"ffn={hp['dim_feedforward']} drop={hp['dropout']} "
              f"lr={hp['learning_rate']} wd={hp['weight_decay']} bs={hp['batch_size']}")

        result = train_with_config(
            train_set, val_set, hp, num_jets, device,
            num_epochs=num_epochs, patience=25, verbose=True,
        )

        result["name"] = name
        result["config"] = {k: v for k, v in hp.items() if k != "name"}
        results.append(result)

        print(f"  => best_val_acc={result['best_val_acc']:.4f} "
              f"(epoch {result['best_epoch']}) "
              f"params={result['total_params']:,} "
              f"time={result['elapsed_seconds']:.1f}s")

    # -----------------------------------------------------------------------
    # 3. Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nBaseline (min mass asymmetry): {baseline_acc:.4f} ({baseline_acc:.1%})")
    print(f"Random chance:                 {1.0/n_assign:.4f} ({1.0/n_assign:.1%})")
    print()

    # Sort by best_val_acc descending
    results_sorted = sorted(results, key=lambda x: x["best_val_acc"], reverse=True)

    print(f"{'Rank':<5} {'Config':<25} {'Val Acc':>8} {'Val Acc5':>9} {'Train Acc':>10} "
          f"{'Params':>10} {'Epochs':>7} {'Time':>7} {'vs Baseline':>12}")
    print("-" * 100)

    for rank, r in enumerate(results_sorted, 1):
        improvement = r["best_val_acc"] - baseline_acc
        print(f"{rank:<5} {r['name']:<25} {r['best_val_acc']:>8.4f} "
              f"{r['final_val_acc5']:>9.4f} {r['final_train_acc']:>10.4f} "
              f"{r['total_params']:>10,} {r['epochs_trained']:>7} "
              f"{r['elapsed_seconds']:>6.1f}s "
              f"{'+'if improvement>0 else ''}{improvement:>+11.4f}")

    # Save results
    os.makedirs("logs", exist_ok=True)
    results_path = Path("logs/hp_scan_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "baseline_val_acc": baseline_acc,
            "random_chance": 1.0 / n_assign,
            "num_val_events": n_val_events,
            "num_train_events": n_train,
            "full_baseline_results": full_results,
            "hp_results": results_sorted,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
