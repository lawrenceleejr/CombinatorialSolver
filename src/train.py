"""
Training loop for the jet assignment transformer.

Supports:
  - Factored loss: ISR cross-entropy + grouping cross-entropy (7-jet mode)
  - Combined cross-entropy (6-jet mode) + adversarial mass decorrelation
  - Cosine LR schedule with linear warmup and optional warm restarts
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
    input_dim = model.input_proj.in_features
    dummy = torch.randn(1, num_jets, input_dim, device=device)
    onnx_path = "checkpoints/best_model.onnx"

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


def cosine_with_warmup(optimizer, epoch, num_epochs, warmup_epochs, restart_period=0):
    """Adjust learning rate: linear warmup then cosine decay with optional warm restarts."""
    if epoch < warmup_epochs:
        lr_scale = (epoch + 1) / warmup_epochs
    else:
        post_warmup = epoch - warmup_epochs
        if restart_period > 0:
            cycle_pos = post_warmup % restart_period
            progress = cycle_pos / restart_period
        else:
            progress = post_warmup / max(num_epochs - warmup_epochs, 1)
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
        input_dim=4,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    if model.has_isr:
        print(f"Architecture: factored (ISR {model.num_jets}-way + grouping {model.num_groupings}-way)")
    else:
        print(f"Architecture: flat ({model.num_assignments}-way)")

    # Optimizer — separate param groups so ISR head gets a higher LR to avoid
    # being drowned out by grouping/GroupTransformer gradients.
    isr_lr_mult = tc.get("isr_lr_multiplier", 1.0)
    base_lr = tc["learning_rate"]

    if model.has_isr and isr_lr_mult != 1.0:
        isr_params = (
            list(model.isr_head.parameters())
            + list(model.grouping_summary_proj.parameters())
        )
        isr_param_ids = {id(p) for p in isr_params}
        other_params = [p for p in model.parameters() if id(p) not in isr_param_ids]

        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "lr": base_lr},
                {"params": isr_params, "lr": base_lr * isr_lr_mult},
            ],
            weight_decay=tc["weight_decay"],
        )
        print(f"Separate LR groups: base={base_lr:.1e}, ISR head={base_lr * isr_lr_mult:.1e}")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
            weight_decay=tc["weight_decay"],
        )
    for pg in optimizer.param_groups:
        pg["initial_lr"] = pg["lr"]

    # Loss functions
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    # Check if adversarial training is useful
    mass_std = dataset.parent_masses.std().item()
    use_adversary = tc["lambda_adv"] > 0 and mass_std > 0.01
    if tc["lambda_adv"] == 0:
        print("Adversary disabled: lambda_adv=0")
    elif not use_adversary:
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
            "val_loss", "val_acc", "val_acc5",
            "train_isr_acc", "train_grp_acc", "val_isr_acc", "val_grp_acc",
            "adv_r2", "lr",
        ])

    best_val_acc = 0.0
    best_epoch = 0
    patience = tc.get("patience", 25)
    no_improve = 0

    tf_start = tc.get("tf_start", 1.0)
    tf_end = tc.get("tf_end", 0.3)
    tf_decay_epochs = tc.get("tf_decay_epochs", 100)
    lambda_isr = tc.get("lambda_isr", 1.0)
    lambda_sym_max = tc.get("lambda_sym", 0.0)
    lambda_qcd_max = tc.get("lambda_qcd", 0.0)
    lambda_sym_rampup = tc.get("lambda_sym_rampup", 0)
    lambda_qcd_rampup = tc.get("lambda_qcd_rampup", 0)

    for epoch in range(tc["num_epochs"]):
        cosine_with_warmup(
            optimizer, epoch, tc["num_epochs"], tc["warmup_epochs"],
            restart_period=tc.get("restart_period", 0),
        )
        current_lr = optimizer.param_groups[0]["lr"]

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

        # Teacher forcing ratio: linearly decay from tf_start to tf_end over tf_decay_epochs
        if epoch < tf_decay_epochs:
            tf_ratio = tf_start + (tf_end - tf_start) * epoch / tf_decay_epochs
        else:
            tf_ratio = tf_end

        # Ramp up auxiliary losses: off at start, linearly reach full strength
        if lambda_sym_rampup > 0:
            lambda_sym = lambda_sym_max * min(1.0, epoch / lambda_sym_rampup)
        else:
            lambda_sym = lambda_sym_max

        if lambda_qcd_rampup > 0:
            lambda_qcd = lambda_qcd_max * min(1.0, epoch / lambda_qcd_rampup)
        else:
            lambda_qcd = lambda_qcd_max

        # Training
        model.train()
        train_metrics = _run_epoch(
            model, train_loader, ce_loss_fn, mse_loss_fn,
            lambda_adv, device, optimizer=optimizer,
            tf_ratio=tf_ratio, lambda_sym=lambda_sym, lambda_qcd=lambda_qcd,
            lambda_isr=lambda_isr,
        )

        # Validation (no augmentation, no teacher forcing: tf_ratio=0 = pure end-to-end)
        model.eval()
        with torch.no_grad():
            val_metrics = _run_epoch(
                model, val_loader, ce_loss_fn, mse_loss_fn,
                lambda_adv, device, optimizer=None,
                tf_ratio=0.0, lambda_sym=0.0, lambda_qcd=0.0,
                lambda_isr=lambda_isr,
            )

        # Log
        adv_str = f" | Adv R²={val_metrics['adv_r2']:.3f}" if use_adversary else ""
        isr_str = ""
        if "isr_acc" in val_metrics:
            isr_str = (
                f" | ISR={val_metrics['isr_acc']:.3f}"
                f" Grp={val_metrics['grp_acc']:.3f}"
            )

        print(
            f"Epoch {epoch+1:3d}/{tc['num_epochs']} | "
            f"Train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.3f} | "
            f"Val loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.3f}"
            f"{isr_str}{adv_str} | "
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
                f"{train_metrics.get('isr_acc', 0):.4f}",
                f"{train_metrics.get('grp_acc', 0):.4f}",
                f"{val_metrics.get('isr_acc', 0):.4f}",
                f"{val_metrics.get('grp_acc', 0):.4f}",
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


def _run_epoch(
    model, loader, ce_loss_fn, mse_loss_fn, lambda_adv, device, optimizer=None,
    tf_ratio=1.0, lambda_sym=0.0, lambda_qcd=0.0, lambda_isr=1.0,
):
    """Run one epoch of training or validation."""
    total_loss = 0.0
    total_correct = 0
    total_correct5 = 0
    total_isr_correct = 0
    total_grp_correct = 0
    total_samples = 0
    all_mass_pred = []
    all_mass_true = []
    factored = model.has_isr

    for batch in loader:
        four_mom = batch["four_momenta"].to(device)
        labels = batch["label"].to(device)
        parent_mass = batch["parent_mass"].to(device)

        # φ/η augmentation during training (hard symmetries of the problem)
        if optimizer is not None:
            four_mom = four_mom.clone()
            batch_size = four_mom.shape[0]

            theta = torch.rand(batch_size, device=device) * 2 * torch.pi
            cos_t = theta.cos().view(-1, 1)   # (batch, 1) for broadcasting over jets
            sin_t = theta.sin().view(-1, 1)
            px_orig = four_mom[:, :, 1].clone()
            py_orig = four_mom[:, :, 2].clone()
            four_mom[:, :, 1] = px_orig * cos_t - py_orig * sin_t
            four_mom[:, :, 2] = px_orig * sin_t + py_orig * cos_t

            flip = (torch.rand(batch_size, device=device) > 0.5).float().view(-1, 1)
            four_mom[:, :, 3] = four_mom[:, :, 3] * (1.0 - 2.0 * flip)

        output = model(four_mom)
        logits = output["logits"]
        mass_pred = output["mass_pred"].squeeze(-1)

        # Assignment loss
        if factored and "isr_logits" in output:
            isr_logits = output["isr_logits"]
            grouping_logits = output["grouping_logits"]

            isr_labels = model.flat_to_factored[labels, 0]
            grouping_labels = model.flat_to_factored[labels, 1]

            loss_isr = ce_loss_fn(isr_logits, isr_labels)

            batch_idx = torch.arange(labels.shape[0], device=device)
            gt_grp_logits = grouping_logits[batch_idx, isr_labels]
            loss_grp_tf = ce_loss_fn(gt_grp_logits, grouping_labels)

            # Blend teacher-forced factored loss with flat end-to-end loss
            # tf_ratio=1: fully teacher-forced (original); tf_ratio=0: flat CE only
            # lambda_isr upweights the ISR loss to compensate for a gradient imbalance:
            # each signal jet appears in all num_groupings groupings, so loss_grp_tf
            # produces num_groupings gradient paths per signal jet while the ISR jet
            # (excluded from every group) receives gradient only from loss_isr.
            # Scaling loss_isr by lambda_isr partially rebalances this asymmetry.
            # loss_flat: logits are log-probabilities (output of _combine_logits
            # which applies log_softmax to each factored component), so use
            # NLLLoss rather than CrossEntropyLoss which would incorrectly
            # re-apply softmax.
            loss_flat = torch.nn.functional.nll_loss(logits, labels)
            loss_ce = tf_ratio * (lambda_isr * loss_isr + loss_grp_tf) + (1.0 - tf_ratio) * loss_flat

            # Measure ISR and grouping accuracy from the combined flat prediction,
            # not from the raw factored heads independently.  The combined logit
            # logit(j,k) = isr_logit[j] + grouping_logit[j,k] lets the grouping
            # head correct the raw ISR head, so argmax(isr_logits) frequently
            # disagrees with the model's actual ISR choice.  The operationally
            # meaningful metric is the ISR/grouping encoded in the combined
            # prediction that the model actually outputs.
            flat_pred = logits.argmax(dim=-1)
            total_isr_correct += (model.flat_to_factored[flat_pred, 0] == isr_labels).sum().item()
            total_grp_correct += (model.flat_to_factored[flat_pred, 1] == grouping_labels).sum().item()
        else:
            loss_ce = ce_loss_fn(logits, labels)

        # Mass symmetry auxiliary loss: minimize expected |m1-m2|/(m1+m2) over assignments
        if lambda_sym > 0 and "mass_asym_flat" in output:
            mass_asym = output["mass_asym_flat"].detach()  # (batch, num_assignments)
            # logits are log-probabilities; convert to probabilities with .exp()
            probs = logits.exp()
            loss_sym = (probs * mass_asym).sum(dim=-1).mean()
            loss_ce = loss_ce + lambda_sym * loss_sym

        # QCD hierarchy penalty: background events with large pT hierarchies are pushed
        # to prefer high-mass-asymmetry assignments, making them self-select interpretations
        # that look maximally unlike a symmetric signal decay.
        # loss_qcd = -mean(H_i * expected_mass_asym_i), where H = log(pT_max/pT_min).
        # Minimising this negative quantity increases H-weighted expected asymmetry,
        # disfavouring signal-like (low-asymmetry) interpretations for QCD-dominated events.
        #
        # IMPORTANT: applied only to background events (parent_mass == 0).  On signal events
        # H > 0 but the correct assignment already has low asymmetry, so firing this penalty
        # on signal events would push the model toward wrong (high-asymmetry) assignments and
        # degrade training.  On a signal-only dataset the mask is always False and the term
        # never contributes; on a mixed dataset it fires exclusively on QCD background events.
        if lambda_qcd > 0 and "mass_asym_flat" in output:
            qcd_mask = parent_mass == 0
            if qcd_mask.any():
                px_all = four_mom[qcd_mask, :, 1]
                py_all = four_mom[qcd_mask, :, 2]
                pt_all = torch.sqrt(px_all**2 + py_all**2).clamp(min=1e-8)
                pt_max = pt_all.max(dim=-1).values
                pt_min = pt_all.min(dim=-1).values.clamp(min=1e-8)
                # Clamp H to prevent very large values from degenerate (near-zero pT_min) events
                H = torch.log(pt_max / pt_min).clamp(max=10.0)         # (n_bkg,) hierarchy score

                # Detach mass_asym: we only want to steer the assignment probabilities,
                # not back-propagate through the physics feature computation itself.
                mass_asym_qcd = output["mass_asym_flat"].detach()[qcd_mask]  # (n_bkg, num_assign)
                # logits are log-probabilities; convert to probabilities with .exp()
                probs_qcd = logits.exp()[qcd_mask]
                expected_asym = (probs_qcd * mass_asym_qcd).sum(dim=-1)     # (n_bkg,)
                # Negative sign: minimising drives H * expected_asym upward for high-H events
                loss_qcd_term = -(H * expected_asym).mean()
                loss_ce = loss_ce + lambda_qcd * loss_qcd_term

        # Adversarial mass loss
        mass_mask = parent_mass > 0
        if mass_mask.any() and lambda_adv > 0:
            loss_adv = mse_loss_fn(mass_pred[mass_mask], parent_mass[mass_mask])
            loss_adv = torch.clamp(loss_adv, max=10.0)
        else:
            loss_adv = torch.tensor(0.0, device=device)

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

        _, top5 = logits.topk(5, dim=-1)
        total_correct5 += (top5 == labels.unsqueeze(-1)).any(dim=-1).sum().item()

        if mass_mask.any():
            all_mass_pred.append(mass_pred[mass_mask].detach().cpu())
            all_mass_true.append(parent_mass[mass_mask].detach().cpu())

    avg_loss = total_loss / max(total_samples, 1)
    acc = total_correct / max(total_samples, 1)
    acc5 = total_correct5 / max(total_samples, 1)

    adv_r2 = 0.0
    if all_mass_pred:
        pred_cat = torch.cat(all_mass_pred)
        true_cat = torch.cat(all_mass_true)
        ss_res = ((pred_cat - true_cat) ** 2).sum()
        ss_tot = ((true_cat - true_cat.mean()) ** 2).sum()
        if ss_tot > 0:
            adv_r2 = 1.0 - (ss_res / ss_tot).item()

    result = {"loss": avg_loss, "acc": acc, "acc5": acc5, "adv_r2": adv_r2}
    if factored:
        result["isr_acc"] = total_isr_correct / max(total_samples, 1)
        result["grp_acc"] = total_grp_correct / max(total_samples, 1)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train jet assignment model")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--data", type=str, default=None, help="Path to HDF5 data (glob pattern)")
    args = parser.parse_args()
    train(config_path=args.config, data_path=args.data)
