"""
Benchmark with pT smearing: mass-asymmetry baseline vs physics-enhanced transformer.

Compares accuracy with and without 5% pT smearing to simulate detector effects.
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.combinatorics import enumerate_assignments
from src.dataset import JetAssignmentDataset
from src.model import JetAssignmentTransformer
from src.utils import get_device


def invariant_mass_np(four_mom):
    e, px, py, pz = four_mom[..., 0], four_mom[..., 1], four_mom[..., 2], four_mom[..., 3]
    m2 = e**2 - px**2 - py**2 - pz**2
    return np.sqrt(np.maximum(m2, 0.0))


def eval_mass_asymmetry_baseline(dataset, num_jets):
    """Evaluate mass-asymmetry baseline accuracy on a dataset."""
    assignments = enumerate_assignments(num_jets)
    n_assign = len(assignments)
    g1_indices = [list(a[1]) for a in assignments]
    g2_indices = [list(a[2]) for a in assignments]

    # Get un-normalized four-momenta
    raw_four_mom = dataset.four_momenta.numpy()
    if dataset.normalize_by_ht:
        ht = dataset.ht.numpy()[:, None, None]
        raw_four_mom = raw_four_mom * ht

    labels = dataset.labels.numpy()
    n_events = len(labels)
    correct = 0

    for i in range(n_events):
        fmom = raw_four_mom[i]
        truth = labels[i]
        best_asym = float("inf")
        best_idx = -1

        for j in range(n_assign):
            g1_4vec = fmom[g1_indices[j]].sum(axis=0)
            g2_4vec = fmom[g2_indices[j]].sum(axis=0)
            m1 = invariant_mass_np(g1_4vec)
            m2 = invariant_mass_np(g2_4vec)
            denom = m1 + m2
            asym = abs(m1 - m2) / denom if denom > 0 else float("inf")
            if asym < best_asym:
                best_asym = asym
                best_idx = j

        if best_idx == truth:
            correct += 1

    return correct / n_events


def train_and_eval(
    train_set, val_set, num_jets, device, d_model=128, nhead=8, num_layers=4,
    dim_feedforward=256, dropout=0.1, lr=3e-4, wd=1e-4, batch_size=256,
    warmup=5, num_epochs=100, patience=25, label="",
):
    """Train model and return best val accuracy."""
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = JetAssignmentTransformer(
        d_model=d_model, nhead=nhead, num_layers=num_layers,
        dim_feedforward=dim_feedforward, dropout=dropout, num_jets=num_jets,
    ).to(device)
    model.gradient_reversal.set_lambda(0.0)

    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    for pg in optimizer.param_groups:
        pg["initial_lr"] = pg["lr"]
    ce = nn.CrossEntropyLoss()

    best_val = 0.0
    best_epoch = 0
    no_improve = 0
    t0 = time.time()

    for epoch in range(num_epochs):
        if epoch < warmup:
            lr_scale = (epoch + 1) / warmup
        else:
            progress = (epoch - warmup) / max(num_epochs - warmup, 1)
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = pg["initial_lr"] * lr_scale

        model.train()
        tc = tt = 0
        for batch in train_loader:
            fm = batch["four_momenta"].to(device)
            lab = batch["label"].to(device)
            out = model(fm)
            loss = ce(out["logits"], lab)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tc += (out["logits"].argmax(-1) == lab).sum().item()
            tt += lab.shape[0]

        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for batch in val_loader:
                fm = batch["four_momenta"].to(device)
                lab = batch["label"].to(device)
                out = model(fm)
                vc += (out["logits"].argmax(-1) == lab).sum().item()
                vt += lab.shape[0]

        train_acc = tc / tt
        val_acc = vc / vt
        if val_acc > best_val:
            best_val = val_acc
            best_epoch = epoch + 1
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 20 == 0:
            print(f"  [{label}] Epoch {epoch+1:3d}/{num_epochs} | "
                  f"train={train_acc:.3f} val={val_acc:.3f} best={best_val:.3f}")

        if no_improve >= patience:
            print(f"  [{label}] Early stop at epoch {epoch+1}")
            break

    elapsed = time.time() - t0
    return best_val, best_epoch, params, elapsed


def main():
    data_path = "data/gluino_rpv_1tev_uds_10000evt_20260225_164524.h5"
    num_jets = 6
    device = get_device()
    print(f"Device: {device}")

    smear_levels = [0.0, 0.05, 0.10, 0.15]

    all_results = []

    for smear in smear_levels:
        smear_pct = f"{smear*100:.0f}%"
        print(f"\n{'='*70}")
        print(f"pT SMEARING: {smear_pct}")
        print(f"{'='*70}")

        dataset = JetAssignmentDataset(
            data_paths=data_path, num_jets=num_jets,
            normalize_by_ht=True, pt_smear_frac=smear,
        )
        # Also un-normalized for baseline
        dataset_raw = JetAssignmentDataset(
            data_paths=data_path, num_jets=num_jets,
            normalize_by_ht=False, pt_smear_frac=smear,
        )

        n_val = max(1, int(0.1 * len(dataset)))
        n_train = len(dataset) - n_val
        gen = torch.Generator().manual_seed(42)
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=gen)
        gen_raw = torch.Generator().manual_seed(42)
        _, val_set_raw = random_split(dataset_raw, [n_train, n_val], generator=gen_raw)

        # Baseline on val set
        print(f"\n  Mass asymmetry baseline...")
        val_fm = dataset_raw.four_momenta[val_set_raw.indices].numpy()
        val_labels = dataset_raw.labels[val_set_raw.indices].numpy()

        assignments = enumerate_assignments(num_jets)
        n_assign = len(assignments)
        g1_idx = [list(a[1]) for a in assignments]
        g2_idx = [list(a[2]) for a in assignments]
        baseline_correct = 0

        for i in range(len(val_labels)):
            fmom = val_fm[i]
            truth = val_labels[i]
            best_asym = float("inf")
            best_j = -1
            for j in range(n_assign):
                g1 = fmom[g1_idx[j]].sum(axis=0)
                g2 = fmom[g2_idx[j]].sum(axis=0)
                m1 = invariant_mass_np(g1)
                m2 = invariant_mass_np(g2)
                d = m1 + m2
                asym = abs(m1 - m2) / d if d > 0 else float("inf")
                if asym < best_asym:
                    best_asym = asym
                    best_j = j
            if best_j == truth:
                baseline_correct += 1

        baseline_acc = baseline_correct / len(val_labels)
        print(f"  Baseline (min mass asym): {baseline_acc:.4f} ({baseline_acc:.1%})")

        # ML model
        print(f"\n  Training physics-enhanced transformer...")
        ml_acc, ml_epoch, ml_params, ml_time = train_and_eval(
            train_set, val_set, num_jets, device,
            d_model=128, nhead=8, num_layers=4, dim_feedforward=256,
            dropout=0.1, lr=3e-4, wd=1e-4, batch_size=256,
            warmup=5, num_epochs=150, patience=30,
            label=f"smear={smear_pct}",
        )
        print(f"  ML model: {ml_acc:.4f} ({ml_acc:.1%}) at epoch {ml_epoch}, "
              f"{ml_params:,} params, {ml_time:.1f}s")

        ratio = ml_acc / baseline_acc if baseline_acc > 0 else float("inf")
        all_results.append({
            "smear": smear_pct,
            "baseline_acc": baseline_acc,
            "ml_acc": ml_acc,
            "ml_epoch": ml_epoch,
            "ml_params": ml_params,
            "ratio": ratio,
        })

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"\n{'Smearing':<10} {'Baseline':>10} {'ML Model':>10} {'ML/Base':>10} {'Gap':>10}")
    print("-" * 55)
    for r in all_results:
        gap = r["ml_acc"] - r["baseline_acc"]
        print(f"{r['smear']:<10} {r['baseline_acc']:>10.4f} {r['ml_acc']:>10.4f} "
              f"{r['ratio']:>10.3f} {gap:>+10.4f}")

    print(f"\nRandom chance: {1.0/n_assign:.4f} ({1.0/n_assign:.1%})")
    print(f"Model params: {all_results[0]['ml_params']:,}")


if __name__ == "__main__":
    main()
