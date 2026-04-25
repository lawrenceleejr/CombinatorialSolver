"""
Training loop for the Pair-Symmetric Slot Transformer (PSST).

Losses (all combined additively):

  - Cross-entropy on flat assignment logits (primary).
  - InfoNCE contrastive loss over single-swap neighbours (pushes truth above
    the 15 hardest confusables that differ by a single jet swap).
  - Pair-symmetric mass-asymmetry regression (weak auxiliary — steers the
    feature block toward assignments with balanced parent masses, but via
    regression rather than by reweighting softmax, which lets the scorer
    choose asymmetric hypotheses when the event evidence demands it).
  - ISR-auxiliary BCE (per-jet "is this the ISR?" head) so the ISR decision
    gets direct supervision even though it's not a separate scorer head.
  - Hard-case reweighting: events where the true ISR is *not* the lowest-pT
    jet are the ones the old architecture got wrong; upweight them so the
    model actually learns the hard direction rather than plateauing on the
    easy one.
  - Optional mass-adversarial regression (gradient reversal) to decorrelate
    features from parent mass — kept from the original design.
"""

import argparse
import csv
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from torch.utils.data import DataLoader, Subset, random_split

from .combinatorics import build_assignment_tensors
from .dataset import JetAssignmentDataset
from .model import JetAssignmentTransformer
from .utils import get_config, get_device


def export_onnx(model, num_jets, device, val_acc):
    model.eval()
    input_dim = 4  # PSST always takes raw 4-vectors
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


def _isr_is_hardest(four_momenta: torch.Tensor, isr_idx: torch.Tensor) -> torch.Tensor:
    """For each event, is the true ISR *not* the lowest-pT jet?  (B,) bool.

    Hard case for the combinatorial problem: pT rank alone cannot identify ISR,
    so the model must use angular structure.
    """
    # Compute pT from (E, px, py, pz)
    px = four_momenta[..., 1]
    py = four_momenta[..., 2]
    pt = torch.sqrt(px * px + py * py + 1e-8)
    # For invalid (zero-pad) jets, pt ≈ 0 — but they're never the true ISR.
    min_pt_idx = pt.argmin(dim=-1)
    return isr_idx != min_pt_idx


def _contrastive_loss(
    logits: torch.Tensor, labels: torch.Tensor, neighbor_idx: torch.Tensor
) -> torch.Tensor:
    """InfoNCE over {truth} ∪ {single-swap neighbours}.

    logits:        (B, N_assignments)
    labels:        (B,) long — truth class
    neighbor_idx:  (N_assignments, n_neighbours) — per-class neighbour table

    The denominator spans truth + up to 15 single-swap alternatives, which are
    the hardest negatives (differ by one jet identity).  This is much more
    sample-efficient than spreading gradient over all 69 non-truth classes.
    """
    B = logits.shape[0]
    # (B, n_neighbours)
    nbrs = neighbor_idx[labels]
    # Gather logits for truth and neighbours
    truth_logits = logits.gather(1, labels.unsqueeze(-1))                    # (B, 1)
    nbr_logits = logits.gather(1, nbrs)                                      # (B, n_neighbours)
    all_logits = torch.cat([truth_logits, nbr_logits], dim=-1)               # (B, 1 + n)
    # Truth is at index 0 in the concatenated logits
    target = torch.zeros(B, dtype=torch.long, device=logits.device)
    return F.cross_entropy(all_logits, target)


def train(config_path: str | None = None, data_path: str | None = None):
    config = get_config(config_path)
    device = get_device()
    print(f"Using device: {device}")

    mc = config["model"]
    tc = config["training"]
    dc = config["data"]

    if data_path is None:
        data_path = "data/*.h5"

    dataset = JetAssignmentDataset(
        data_paths=data_path,
        num_jets=dc["num_jets"],
        normalize_by_ht=dc["normalize_by_ht"],
        pt_smear_frac=dc.get("pt_smear_frac", 0.0),
    )
    print(f"Dataset size: {len(dataset)} events")

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

    # Hard-only curriculum warmup loader.  For the first
    # `hard_only_warmup_epochs` epochs the model trains only on events where
    # ISR is NOT the lowest-pT jet, removing the shortcut entirely from the
    # data so it can never be a local optimum.  Then the full training set
    # comes back online (with hard_case_alpha still upweighting hard cases).
    has_isr = dc["num_jets"] >= 7
    hard_only_warmup_epochs = tc.get("hard_only_warmup_epochs", 0) if has_isr else 0
    hard_train_loader = None
    if hard_only_warmup_epochs > 0:
        at = build_assignment_tensors(dc["num_jets"])
        isr_indices_assign = at["isr_indices"]                    # (N_assignments,)

        train_indices = list(train_set.indices)
        train_labels = dataset.labels[train_indices]              # (N_train,)
        train_fm = dataset.four_momenta[train_indices]            # (N_train, J, 4)

        train_isr = isr_indices_assign[train_labels]              # (N_train,)
        pt = (train_fm[..., 1] ** 2 + train_fm[..., 2] ** 2).sqrt()
        min_pt_idx = pt.argmin(dim=-1)                            # (N_train,)
        hard_mask = (train_isr != min_pt_idx).tolist()
        hard_indices = [train_indices[i] for i, h in enumerate(hard_mask) if h]

        hard_train_set = Subset(dataset, hard_indices)
        hard_train_loader = DataLoader(
            hard_train_set,
            batch_size=tc["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        print(
            f"Hard-only warmup: {len(hard_train_set):,}/{len(train_set):,} hard events "
            f"for the first {hard_only_warmup_epochs} epochs"
        )
    val_loader = DataLoader(
        val_set,
        batch_size=tc["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

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
    print(
        f"Architecture: PSST ({model.num_assignments}-way flat scorer, "
        f"{model.num_triplets} triplets, has_isr={model.has_isr})"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tc["learning_rate"],
        weight_decay=tc["weight_decay"],
    )
    for pg in optimizer.param_groups:
        pg["initial_lr"] = pg["lr"]

    mse_loss_fn = nn.MSELoss()

    mass_std = dataset.parent_masses.std().item()
    use_adversary = tc["lambda_adv"] > 0 and mass_std > 0.01
    if tc["lambda_adv"] == 0:
        print("Adversary disabled: lambda_adv=0")
    elif not use_adversary:
        print("Adversary disabled: single mass point detected")
    else:
        print(f"Adversary enabled: mass std = {mass_std:.3f} TeV")

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    log_path = Path("logs/training_log.csv")

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_acc", "train_acc5",
            "val_loss", "val_acc", "val_acc5",
            "val_hard_acc", "val_easy_acc",
            "adv_r2", "lr",
        ])

    best_val_acc = 0.0
    best_epoch = 0
    patience = tc.get("patience", 25)
    no_improve = 0

    lambda_contrast_max = tc.get("lambda_contrast", 1.0)
    lambda_contrast_rampup = tc.get("lambda_contrast_rampup", 0)
    lambda_pair_sym = tc.get("lambda_pair_sym", 0.05)
    lambda_isr_aux_max = tc.get("lambda_isr_aux", 0.5)
    lambda_isr_aux_rampup = tc.get("lambda_isr_aux_rampup", 0)
    hard_case_alpha = tc.get("hard_case_alpha", 2.0)   # weight for hard-ISR events

    for epoch in range(tc["num_epochs"]):
        cosine_with_warmup(
            optimizer, epoch, tc["num_epochs"], tc["warmup_epochs"],
            restart_period=tc.get("restart_period", 0),
        )
        current_lr = optimizer.param_groups[0]["lr"]

        if use_adversary:
            rampup = tc.get("lambda_adv_rampup", 10)
            adv_scale = min(1.0, epoch / rampup) if rampup > 0 else 1.0
            lambda_adv = tc["lambda_adv"] * adv_scale
            model.gradient_reversal.set_lambda(lambda_adv)
        else:
            lambda_adv = 0.0
            model.gradient_reversal.set_lambda(0.0)

        if lambda_contrast_rampup > 0:
            lambda_contrast = lambda_contrast_max * min(1.0, epoch / lambda_contrast_rampup)
        else:
            lambda_contrast = lambda_contrast_max

        if lambda_isr_aux_rampup > 0:
            lambda_isr_aux = lambda_isr_aux_max * min(1.0, epoch / lambda_isr_aux_rampup)
        else:
            lambda_isr_aux = lambda_isr_aux_max

        # Curriculum: hard-only loader during warmup, full loader after.
        if hard_train_loader is not None and epoch < hard_only_warmup_epochs:
            active_loader = hard_train_loader
            phase = "hard-only"
        else:
            active_loader = train_loader
            phase = "full"
            if hard_train_loader is not None and epoch == hard_only_warmup_epochs:
                print(f"  -> Switching to full training set (epoch {epoch + 1})")

        model.train()
        train_metrics = _run_epoch(
            model, active_loader, mse_loss_fn, lambda_adv, device,
            optimizer=optimizer,
            lambda_contrast=lambda_contrast,
            lambda_pair_sym=lambda_pair_sym,
            lambda_isr_aux=lambda_isr_aux,
            hard_case_alpha=hard_case_alpha,
            augment=True,
        )

        model.eval()
        with torch.no_grad():
            val_metrics = _run_epoch(
                model, val_loader, mse_loss_fn, lambda_adv, device,
                optimizer=None,
                lambda_contrast=0.0,
                lambda_pair_sym=0.0,
                lambda_isr_aux=0.0,
                hard_case_alpha=1.0,
                augment=False,
            )

        adv_str = f" | Adv R²={val_metrics['adv_r2']:.3f}" if use_adversary else ""
        hard_str = ""
        if "hard_acc" in val_metrics:
            hard_str = (
                f" | hard={val_metrics['hard_acc']:.3f}"
                f" easy={val_metrics['easy_acc']:.3f}"
            )
        phase_str = f" [{phase}]" if hard_train_loader is not None else ""
        print(
            f"Epoch {epoch+1:3d}/{tc['num_epochs']}{phase_str} | "
            f"Train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.3f} | "
            f"Val loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.3f}"
            f"{hard_str}{adv_str} | LR={current_lr:.2e}"
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
                f"{val_metrics.get('hard_acc', 0):.4f}",
                f"{val_metrics.get('easy_acc', 0):.4f}",
                f"{val_metrics['adv_r2']:.4f}",
                f"{current_lr:.6e}",
            ])

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

    ckpt = torch.load("checkpoints/best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    export_onnx(model, dc["num_jets"], device, best_val_acc)


def _run_epoch(
    model, loader, mse_loss_fn, lambda_adv, device,
    optimizer=None,
    lambda_contrast=0.0,
    lambda_pair_sym=0.0,
    lambda_isr_aux=0.0,
    hard_case_alpha=1.0,
    augment=False,
):
    total_loss = 0.0
    total_correct = 0
    total_correct5 = 0
    total_samples = 0
    total_hard = 0
    total_hard_correct = 0
    total_easy = 0
    total_easy_correct = 0
    all_mass_pred = []
    all_mass_true = []

    neighbor_idx = getattr(model, "neighbor_idx", None)                      # (N, n_nbrs)
    isr_indices_assign = getattr(model, "isr_indices_assign", None)           # (N,) -1 if 6-jet
    has_isr = model.has_isr
    num_jets = model.num_jets

    for batch in loader:
        four_mom = batch["four_momenta"].to(device)
        labels = batch["label"].to(device)
        parent_mass = batch["parent_mass"].to(device)

        # --- φ-rotation + η-flip augmentation (hard symmetries of the problem) ---
        if augment:
            four_mom = four_mom.clone()
            batch_size = four_mom.shape[0]
            theta = torch.rand(batch_size, device=device) * 2 * torch.pi
            cos_t = theta.cos().view(-1, 1)
            sin_t = theta.sin().view(-1, 1)
            px_orig = four_mom[:, :, 1].clone()
            py_orig = four_mom[:, :, 2].clone()
            four_mom[:, :, 1] = px_orig * cos_t - py_orig * sin_t
            four_mom[:, :, 2] = px_orig * sin_t + py_orig * cos_t
            flip = (torch.rand(batch_size, device=device) > 0.5).float().view(-1, 1)
            four_mom[:, :, 3] = four_mom[:, :, 3] * (1.0 - 2.0 * flip)

        output = model(four_mom)
        logits = output["logits"]                                            # (B, N)
        mass_pred = output["mass_pred"].squeeze(-1)

        # --- Hard-case reweighting ---
        # Per-event weight: hard_case_alpha if ISR is not the lowest-pT jet, else 1.
        if has_isr and hard_case_alpha != 1.0:
            true_isr = isr_indices_assign[labels]                            # (B,)
            hard = _isr_is_hardest(four_mom, true_isr)                       # (B,)
            weights = torch.where(
                hard,
                torch.full_like(hard, hard_case_alpha, dtype=torch.float32),
                torch.ones_like(hard, dtype=torch.float32),
            )
        else:
            weights = torch.ones(labels.shape[0], device=device)

        # --- Cross-entropy (weighted per sample) ---
        ce_per_sample = F.cross_entropy(logits, labels, reduction="none")    # (B,)
        w_norm = weights / weights.mean().clamp(min=1e-6)
        loss_ce = (w_norm * ce_per_sample).mean()

        # --- Contrastive single-swap InfoNCE ---
        if lambda_contrast > 0 and neighbor_idx is not None:
            loss_contrast = _contrastive_loss(logits, labels, neighbor_idx)
        else:
            loss_contrast = torch.tensor(0.0, device=device)

        # --- Pair-symmetric mass regression ---
        # Weak signal: the truth assignment tends to have small asymmetry, so
        # regressing predicted-truth asymmetry toward zero helps the scorer
        # build a representation that's sensitive to mass balance.  But it's
        # only applied to the truth class — we do NOT reweight softmax by
        # asymmetry, which would kill hard events where the truth assignment
        # has an asymmetric fluctuation.
        if lambda_pair_sym > 0 and "mass_asym_flat" in output:
            mass_asym = output["mass_asym_flat"]                             # (B, N)
            truth_asym = mass_asym.gather(1, labels.unsqueeze(-1)).squeeze(-1)
            # We don't want to back-prop through the physics feature computation
            # (it's not differentiable through the sqrt at m2 ≈ 0).  Use a
            # simple target-zero MSE on the detached truth asymmetry as a
            # *signal statistic*: encourage the scorer to put high logit mass
            # on hypotheses whose asymmetry is small.  We reweight CE gradient
            # by a margin against neighbours — achieved via a margin regressor.
            truth_logit = logits.gather(1, labels.unsqueeze(-1)).squeeze(-1)
            # Regression target: -asymmetry (so lower asymmetry → higher logit)
            loss_pair_sym = F.mse_loss(truth_logit.tanh(), (-truth_asym.detach()).tanh())
        else:
            loss_pair_sym = torch.tensor(0.0, device=device)

        # --- ISR auxiliary BCE ---
        if lambda_isr_aux > 0 and has_isr and "isr_aux_logits" in output:
            true_isr = isr_indices_assign[labels]                            # (B,)
            isr_target = F.one_hot(true_isr, num_classes=num_jets).float()   # (B, J)
            loss_isr_aux = F.binary_cross_entropy_with_logits(
                output["isr_aux_logits"], isr_target
            )
        else:
            loss_isr_aux = torch.tensor(0.0, device=device)

        # --- Mass adversarial ---
        mass_mask = parent_mass > 0
        if mass_mask.any() and lambda_adv > 0:
            loss_adv = mse_loss_fn(mass_pred[mass_mask], parent_mass[mass_mask])
            loss_adv = torch.clamp(loss_adv, max=10.0)
        else:
            loss_adv = torch.tensor(0.0, device=device)

        loss = (
            loss_ce
            + lambda_contrast * loss_contrast
            + lambda_pair_sym * loss_pair_sym
            + lambda_isr_aux * loss_isr_aux
            + lambda_adv * loss_adv
        )

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # --- Metrics ---
        batch_size = labels.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds = logits.argmax(dim=-1)
        correct = (preds == labels)
        total_correct += correct.sum().item()

        _, top5 = logits.topk(5, dim=-1)
        total_correct5 += (top5 == labels.unsqueeze(-1)).any(dim=-1).sum().item()

        if has_isr:
            true_isr = isr_indices_assign[labels]
            hard = _isr_is_hardest(four_mom, true_isr)
            total_hard += hard.sum().item()
            total_hard_correct += (correct & hard).sum().item()
            total_easy += (~hard).sum().item()
            total_easy_correct += (correct & ~hard).sum().item()

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
    if has_isr and total_hard + total_easy > 0:
        result["hard_acc"] = total_hard_correct / max(total_hard, 1)
        result["easy_acc"] = total_easy_correct / max(total_easy, 1)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train jet assignment model")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--data", type=str, default=None, help="Path to HDF5 data (glob pattern)")
    args = parser.parse_args()
    train(config_path=args.config, data_path=args.data)
