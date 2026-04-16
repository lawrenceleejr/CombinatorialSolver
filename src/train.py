"""
Training loop for the jet assignment transformer.

Supports:
  - Combined cross-entropy (assignment) + adversarial mass decorrelation loss
  - Cosine LR schedule with linear warmup
  - Automatic device selection (MPS / CUDA / CPU)
  - Checkpointing, CSV logging, and ONNX export
"""

import argparse
import csv
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.onnx
from torch.utils.data import DataLoader, random_split

from .dataset import JetAssignmentDataset
from .model import JetAssignmentTransformer
from .utils import get_config, get_device


def export_onnx(model, num_jets, device, val_acc):
    """Export model to ONNX format."""
    model.eval()
    dummy = torch.randn(1, num_jets, 4, device=device)
    onnx_path = "checkpoints/best_model.onnx"

    # Wrap to export only logits (single tensor output)
    class _Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, four_momenta):
            return self.m(four_momenta)["logits"]

    wrapper = _Wrapper(model)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["four_momenta"],
        output_names=["logits"],
        dynamic_axes={
            "four_momenta": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=18,
    )
    print(f"  -> Exported ONNX model to {onnx_path} (val_acc={val_acc:.4f})")


def cosine_with_warmup(optimizer, epoch, num_epochs, warmup_epochs):
    """Adjust learning rate: linear warmup then cosine decay."""
    if epoch < warmup_epochs:
        lr_scale = (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
        lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))

    for pg in optimizer.param_groups:
        pg["lr"] = pg["initial_lr"] * lr_scale


def train(config_path: str | None = None, data_path: str | None = None):
    """Main training function."""
    config = get_config(config_path)
    device = get_device()
    print(f"Using device: {device}")

    mc = config["model"]
    tc = config["training"]
    dc = config["data"]

    # Data
    if data_path is None:
        data_path = "data/*.h5"

    dataset = JetAssignmentDataset(
        data_paths=data_path,
        num_jets=dc["num_jets"],
        normalize_by_ht=dc["normalize_by_ht"],
        pt_smear_frac=dc.get("pt_smear_frac", 0.0),
    )
    print(f"Dataset size: {len(dataset)} events")

    # Train/val split (90/10)
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=tc["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=tc["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Model
    model = JetAssignmentTransformer(
        d_model=mc["d_model"],
        nhead=mc["nhead"],
        num_layers=mc["num_layers"],
        dim_feedforward=mc["dim_feedforward"],
        dropout=mc["dropout"],
        num_jets=dc["num_jets"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc["learning_rate"],
        weight_decay=tc["weight_decay"],
    )
    # Store initial LR for scheduler
    for pg in optimizer.param_groups:
        pg["initial_lr"] = pg["lr"]

    # Loss functions
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    # Check if adversarial training is useful (need multiple mass points)
    mass_std = dataset.parent_masses.std().item()
    use_adversary = mass_std > 0.01  # Disable if all events have same mass
    if not use_adversary:
        print("Adversary disabled: single mass point detected")
    else:
        print(f"Adversary enabled: mass std = {mass_std:.3f} TeV")

    # Logging
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    log_path = Path("logs/training_log.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_acc", "train_acc5",
            "val_loss", "val_acc", "val_acc5", "adv_r2", "lr",
        ])

    best_val_acc = 0.0
    best_epoch = 0
    patience = tc.get("patience", 25)
    no_improve = 0

    for epoch in range(tc["num_epochs"]):
        # Update learning rate
        cosine_with_warmup(optimizer, epoch, tc["num_epochs"], tc["warmup_epochs"])
        current_lr = optimizer.param_groups[0]["lr"]

        # Ramp up adversarial strength (only if multiple mass points)
        if use_adversary:
            rampup = tc.get("lambda_adv_rampup", 10)
            if rampup > 0:
                adv_scale = min(1.0, epoch / rampup)
            else:
                adv_scale = 1.0
            lambda_adv = tc["lambda_adv"] * adv_scale
            model.gradient_reversal.set_lambda(lambda_adv)
        else:
            lambda_adv = 0.0
            model.gradient_reversal.set_lambda(0.0)

        # Training
        model.train()
        train_metrics = _run_epoch(
            model, train_loader, ce_loss_fn, mse_loss_fn,
            lambda_adv, device, optimizer=optimizer,
        )

        # Validation
        model.eval()
        with torch.no_grad():
            val_metrics = _run_epoch(
                model, val_loader, ce_loss_fn, mse_loss_fn,
                lambda_adv, device, optimizer=None,
            )

        # Log
        adv_str = f" | Adv R²={val_metrics['adv_r2']:.3f}" if use_adversary else ""
        print(
            f"Epoch {epoch+1:3d}/{tc['num_epochs']} | "
            f"Train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.3f} | "
            f"Val loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.3f}"
            f"{adv_str} | "
            f"LR={current_lr:.2e}"
        )

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{train_metrics['loss']:.6f}",
                f"{train_metrics['acc']:.4f}",
                f"{train_metrics['acc5']:.4f}",
                f"{val_metrics['loss']:.6f}",
                f"{val_metrics['acc']:.4f}",
                f"{val_metrics['acc5']:.4f}",
                f"{val_metrics['adv_r2']:.4f}",
                f"{current_lr:.6e}",
            ])

        # Checkpoint
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": best_val_acc,
                    "config": config,
                },
                "checkpoints/best_model.pt",
            )
            print(f"  -> Saved best model (val_acc={best_val_acc:.4f})")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    # Reload best checkpoint before ONNX export
    ckpt = torch.load("checkpoints/best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    export_onnx(model, dc["num_jets"], device, best_val_acc)


def _run_epoch(model, loader, ce_loss_fn, mse_loss_fn, lambda_adv, device, optimizer=None):
    """Run one epoch of training or validation."""
    total_loss = 0.0
    total_correct = 0
    total_correct5 = 0
    total_samples = 0
    all_mass_pred = []
    all_mass_true = []

    for batch in loader:
        four_mom = batch["four_momenta"].to(device)
        labels = batch["label"].to(device)
        parent_mass = batch["parent_mass"].to(device)

        output = model(four_mom)
        logits = output["logits"]
        mass_pred = output["mass_pred"].squeeze(-1)

        # Assignment cross-entropy loss
        loss_ce = ce_loss_fn(logits, labels)

        # Adversarial mass loss (MSE on predicted vs true parent mass)
        # Only compute for events where we have truth mass > 0
        mass_mask = parent_mass > 0
        if mass_mask.any() and lambda_adv > 0:
            loss_adv = mse_loss_fn(mass_pred[mass_mask], parent_mass[mass_mask])
            loss_adv = torch.clamp(loss_adv, max=10.0)
        else:
            loss_adv = torch.tensor(0.0, device=device)

        # Combined loss: CE + adversarial (GRL handles gradient sign flip)
        loss = loss_ce + lambda_adv * loss_adv

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Metrics
        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()

        # Top-5 accuracy
        _, top5 = logits.topk(5, dim=-1)
        total_correct5 += (top5 == labels.unsqueeze(-1)).any(dim=-1).sum().item()

        # Track mass predictions for R² computation
        if mass_mask.any():
            all_mass_pred.append(mass_pred[mass_mask].detach().cpu())
            all_mass_true.append(parent_mass[mass_mask].detach().cpu())

    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)
    acc5 = total_correct5 / max(total_samples, 1)

    # Compute adversary R² (should stay low if decorrelation works)
    adv_r2 = 0.0
    if all_mass_pred:
        pred_cat = torch.cat(all_mass_pred)
        true_cat = torch.cat(all_mass_true)
        ss_res = ((pred_cat - true_cat) ** 2).sum()
        ss_tot = ((true_cat - true_cat.mean()) ** 2).sum()
        if ss_tot > 0:
            adv_r2 = 1.0 - (ss_res / ss_tot).item()

    return {"loss": avg_loss, "acc": acc, "acc5": acc5, "adv_r2": adv_r2}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train jet assignment model")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--data", type=str, default=None, help="Path to HDF5 data (glob pattern)")
    args = parser.parse_args()
    train(config_path=args.config, data_path=args.data)
