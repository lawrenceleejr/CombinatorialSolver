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
import torch.nn.functional as F
import torch.onnx
from torch.utils.data import DataLoader, random_split

from .dataset import JetAssignmentDataset
from .model import JetAssignmentTransformer
from .utils import get_config, get_device


def export_onnx(model, num_jets, device, val_acc):
    """Export model to ONNX format."""
    model.eval()
    input_dim = model.raw_input_dim  # raw 4-vector dim; model augments internally
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
    # Detect whether this is the new flat joint model or the old factored model.
    has_factored_architecture = hasattr(model, "isr_head")
    if model.has_isr:
        if has_factored_architecture:
            print(f"Architecture: factored (ISR {model.num_jets}-way + grouping {model.num_groupings}-way)")
        else:
            print(f"Architecture: joint flat ({model.num_assignments}-way, no factored ISR head)")
    else:
        print(f"Architecture: flat ({model.num_assignments}-way)")

    # Optimizer — use a single parameter group for the new flat architecture.
    # isr_lr_multiplier is only applied for the old factored architecture that
    # still has a separate isr_head; the new joint model uses a single group.
    isr_lr_mult = tc.get("isr_lr_multiplier", 1.0)
    base_lr = tc["learning_rate"]

    if model.has_isr and isr_lr_mult != 1.0 and has_factored_architecture:
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
    label_smoothing = tc.get("label_smoothing", 0.0)
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
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
    # New auxiliary loss weights (default to 0 for backward compatibility)
    lambda_isr_aux_max  = tc.get("lambda_isr_aux", 0.0)
    lambda_infonce_max  = tc.get("lambda_infonce", 0.0)
    lambda_hard         = tc.get("lambda_hard", 0.0)
    # Ramp-up periods for new losses
    lambda_isr_aux_rampup = tc.get("lambda_isr_aux_rampup", 10)
    lambda_infonce_rampup = tc.get("lambda_infonce_rampup", 10)

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

        if lambda_isr_aux_rampup > 0:
            lambda_isr_aux = lambda_isr_aux_max * min(1.0, epoch / lambda_isr_aux_rampup)
        else:
            lambda_isr_aux = lambda_isr_aux_max

        if lambda_infonce_rampup > 0:
            lambda_infonce = lambda_infonce_max * min(1.0, epoch / lambda_infonce_rampup)
        else:
            lambda_infonce = lambda_infonce_max

        # Training
        model.train()
        train_metrics = _run_epoch(
            model, train_loader, ce_loss_fn, mse_loss_fn,
            lambda_adv, device, optimizer=optimizer,
            tf_ratio=tf_ratio, lambda_sym=lambda_sym, lambda_qcd=lambda_qcd,
            lambda_isr=lambda_isr,
            lambda_isr_aux=lambda_isr_aux, lambda_infonce=lambda_infonce,
            lambda_hard=lambda_hard, label_smoothing=label_smoothing,
        )

        # Validation (no augmentation, no auxiliary losses)
        model.eval()
        with torch.no_grad():
            val_metrics = _run_epoch(
                model, val_loader, ce_loss_fn, mse_loss_fn,
                lambda_adv, device, optimizer=None,
                tf_ratio=0.0, lambda_sym=0.0, lambda_qcd=0.0,
                lambda_isr=lambda_isr,
                lambda_isr_aux=0.0, lambda_infonce=0.0,
                lambda_hard=0.0, label_smoothing=label_smoothing,
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


def permute_jets_and_remap_labels(four_mom, labels, model, device):
    """Randomly shuffle jet order per event and remap assignment labels accordingly.

    Jets are pT-sorted by the dataset, which gives the model a positional shortcut:
    "last jet (index 6) = lowest pT = likely ISR."  Randomly permuting the jet
    order during training forces the model to use kinematic features (relative
    pT ratios, ΔR, pairwise masses) rather than slot position to identify the
    ISR jet and signal groupings.

    Zero-padded ISR slots (when no ISR jet exists in the event) get mixed into
    random positions, so the model must learn to identify them by their all-zero
    feature vector rather than by being last.

    Args:
        four_mom: (batch, J, 4) input four-momenta (pT-sorted)
        labels:   (batch,) assignment label indices (in original sorted order)
        model:    JetAssignmentTransformer — provides isr_indices, group1/2_indices
        device:   torch device

    Returns:
        four_mom_perm: (batch, J, 4) permuted four-momenta
        labels_perm:   (batch,) remapped labels consistent with permuted ordering
    """
    batch_size, num_jets = four_mom.shape[:2]

    # Vectorised random permutation: argsort of uniform noise gives uniform perms.
    # perms[b, new_pos] = old_pos  →  four_mom_perm[b, new_pos] = four_mom[b, old_pos]
    perms = torch.rand(batch_size, num_jets, device=device).argsort(dim=1)

    # Permute four-momenta
    perm_expand = perms.unsqueeze(-1).expand(-1, -1, four_mom.shape[2])
    four_mom_perm = torch.gather(four_mom, 1, perm_expand)

    # Compute inverse permutations: inv_perms[b, old_pos] = new_pos
    inv_perms = torch.zeros_like(perms)
    inv_perms.scatter_(
        1, perms,
        torch.arange(num_jets, device=device).unsqueeze(0).expand(batch_size, -1),
    )

    # Original ISR and group jet positions (in the old pT-sorted ordering).
    # model.isr_indices[label] = ISR jet index for that assignment (-1 for 6-jet mode).
    isr_old = model.isr_indices[labels]          # (batch,)
    g1_old  = model.group1_indices[labels, :]    # (batch, 3)
    g2_old  = model.group2_indices[labels, :]    # (batch, 3)

    # Apply inverse permutation to get new jet positions.
    # For 6-jet mode (isr_old == -1): clamp to 0 for valid gather, restore -1 after.
    isr_valid = isr_old.clamp(min=0).unsqueeze(1)
    isr_new = inv_perms.gather(1, isr_valid).squeeze(1)
    isr_new = torch.where(isr_old < 0, isr_old, isr_new)  # restore -1 for 6-jet

    g1_new = inv_perms.gather(1, g1_old.long())   # (batch, 3)
    g2_new = inv_perms.gather(1, g2_old.long())   # (batch, 3)

    # Sort within each group to get canonical form (ascending jet index).
    g1_new_s, _ = g1_new.sort(dim=1)   # (batch, 3)
    g2_new_s, _ = g2_new.sort(dim=1)   # (batch, 3)

    # Find the new flat assignment label by matching (isr_new, g1_new, g2_new)
    # against all N canonical assignments.  We check both orderings of the two
    # groups since canonical form may have swapped them (g1 < g2 lexicographically).
    all_isr = model.isr_indices.unsqueeze(0)    # (1, N)
    all_g1  = model.group1_indices.unsqueeze(0) # (1, N, 3)
    all_g2  = model.group2_indices.unsqueeze(0) # (1, N, 3)

    isr_match = isr_new.unsqueeze(1) == all_isr                          # (batch, N)
    g1s_g1 = (g1_new_s.unsqueeze(1) == all_g1).all(dim=2)               # (batch, N)
    g2s_g2 = (g2_new_s.unsqueeze(1) == all_g2).all(dim=2)               # (batch, N)
    g1s_g2 = (g1_new_s.unsqueeze(1) == all_g2).all(dim=2)               # (batch, N)
    g2s_g1 = (g2_new_s.unsqueeze(1) == all_g1).all(dim=2)               # (batch, N)

    grp_match  = (g1s_g1 & g2s_g2) | (g1s_g2 & g2s_g1)                 # (batch, N)
    full_match = isr_match & grp_match                                   # (batch, N)

    labels_perm = full_match.long().argmax(dim=1)   # (batch,)
    return four_mom_perm, labels_perm


def _run_epoch(
    model, loader, ce_loss_fn, mse_loss_fn, lambda_adv, device, optimizer=None,
    tf_ratio=1.0, lambda_sym=0.0, lambda_qcd=0.0, lambda_isr=1.0,
    lambda_isr_aux=0.0, lambda_infonce=0.0, lambda_hard=0.0, label_smoothing=0.0,
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
        batch_size = labels.shape[0]

        # Training-time augmentations
        if optimizer is not None:
            four_mom = four_mom.clone()

            # φ rotation (azimuthal symmetry — exact) and η flip (parity — exact)
            theta = torch.rand(batch_size, device=device) * 2 * torch.pi
            cos_t = theta.cos().view(-1, 1)
            sin_t = theta.sin().view(-1, 1)
            px_orig = four_mom[:, :, 1].clone()
            py_orig = four_mom[:, :, 2].clone()
            four_mom[:, :, 1] = px_orig * cos_t - py_orig * sin_t
            four_mom[:, :, 2] = px_orig * sin_t + py_orig * cos_t

            flip = (torch.rand(batch_size, device=device) > 0.5).float().view(-1, 1)
            four_mom[:, :, 3] = four_mom[:, :, 3] * (1.0 - 2.0 * flip)

            # Random jet order permutation — breaks the pT-rank positional shortcut.
            # The dataset sorts jets by descending pT so "last slot = lowest pT = ISR"
            # is a cheap heuristic.  Shuffling forces the model to use kinematic
            # relationships (pT ratios, ΔR, m_ij) rather than position.
            # Labels are remapped to remain consistent with the shuffled ordering.
            if factored:
                four_mom, labels = permute_jets_and_remap_labels(
                    four_mom, labels, model, device
                )

        output = model(four_mom)
        logits = output["logits"]
        mass_pred = output["mass_pred"].squeeze(-1)

        # ── Assignment loss ────────────────────────────────────────────────────
        # Legacy factored path (old architecture with separate isr_logits).
        if factored and "isr_logits" in output:
            isr_logits = output["isr_logits"]
            grouping_logits = output["grouping_logits"]

            isr_labels = model.flat_to_factored[labels, 0]
            grouping_labels = model.flat_to_factored[labels, 1]

            loss_isr = ce_loss_fn(isr_logits, isr_labels)

            batch_idx = torch.arange(labels.shape[0], device=device)
            gt_grp_logits = grouping_logits[batch_idx, isr_labels]
            loss_grp_tf = ce_loss_fn(gt_grp_logits, grouping_labels)

            loss_flat = ce_loss_fn(logits, labels)
            loss_ce = tf_ratio * (lambda_isr * loss_isr + loss_grp_tf) + (1.0 - tf_ratio) * loss_flat

            eval_logits = output.get("logits_eval", logits)
            flat_pred = eval_logits.argmax(dim=-1)
            total_isr_correct += (model.flat_to_factored[flat_pred, 0] == isr_labels).sum().item()
            total_grp_correct += (model.flat_to_factored[flat_pred, 1] == grouping_labels).sum().item()
        else:
            # Flat joint model: single N-way CE, with optional hard-case weighting.
            #
            # Hard-case reweighting: events where the ISR jet is NOT the lowest-pT
            # jet get upweighted.  The model has an easy shortcut ("last jet =
            # lowest pT = ISR") that works ~50% of the time but fails on the hard
            # case.  Upweighting these events directs gradient toward learning the
            # genuine kinematic discriminants.
            if factored and lambda_hard > 0:
                with torch.no_grad():
                    px_j = four_mom[..., 1]
                    py_j = four_mom[..., 2]
                    pt_j = torch.sqrt(px_j ** 2 + py_j ** 2 + 1e-8)
                    min_pt_pos = pt_j.argmin(dim=1)                  # (batch,)
                    isr_pos = model.isr_indices[labels]              # (batch,)
                    is_hard = (isr_pos != min_pt_pos).float()        # 1 if hard case
                    event_weight = 1.0 + lambda_hard * is_hard       # (batch,)

                loss_ce_per = F.cross_entropy(
                    logits, labels, reduction="none", label_smoothing=label_smoothing
                )
                loss_ce = (loss_ce_per * event_weight).mean()
            else:
                loss_ce = ce_loss_fn(logits, labels)

            eval_logits = logits

        # ISR and grouping accuracy from flat predictions (flat model path).
        if factored and "isr_logits" not in output:
            flat_pred = eval_logits.argmax(dim=-1)
            isr_labels = model.flat_to_factored[labels, 0]
            grouping_labels = model.flat_to_factored[labels, 1]
            total_isr_correct += (model.flat_to_factored[flat_pred, 0] == isr_labels).sum().item()
            total_grp_correct += (model.flat_to_factored[flat_pred, 1] == grouping_labels).sum().item()

        # ── InfoNCE on single-swap neighbours ─────────────────────────────────
        # For each event, we compute two focused CE losses using the flat logits:
        #   1. Grouping InfoNCE (10-way): given the *correct* ISR choice, which of
        #      the 10 groupings is right?  Sharpens grouping gradients without the
        #      confounding ISR ambiguity.
        #   2. ISR InfoNCE (7-way): given the best grouping *per* ISR choice (max
        #      logit over 10 groupings), which ISR is right?  Sharpens the ISR
        #      gradient without the confounding grouping ambiguity.
        # Together these act as InfoNCE over the "single-swap" neighbourhood
        # (assignments differing by one decision from the correct one).
        if factored and lambda_infonce > 0 and hasattr(model, "factored_to_flat"):
            if "isr_logits" not in output:
                isr_labels_f = model.flat_to_factored[labels, 0]    # (batch,)
                grp_labels_f = model.flat_to_factored[labels, 1]    # (batch,)

                # 1. Grouping InfoNCE: 10-way CE over all groupings for correct ISR
                grp_flat_idxs = model.factored_to_flat[isr_labels_f, :]  # (batch, 10)
                grp_logits_nc = logits.gather(1, grp_flat_idxs)           # (batch, 10)
                loss_infonce_grp = F.cross_entropy(grp_logits_nc, grp_labels_f)

                # 2. ISR InfoNCE: 7-way CE using max logit per ISR choice
                # factored_to_flat: (J, 10) of flat indices
                # logits[:, factored_to_flat]: (batch, J*10) → reshape (batch, J, 10)
                logits_fac = logits[:, model.factored_to_flat.reshape(-1)].reshape(
                    batch_size, model.num_jets, model.num_groupings
                )
                isr_logits_nc = logits_fac.max(dim=2).values  # (batch, J)
                loss_infonce_isr = F.cross_entropy(isr_logits_nc, isr_labels_f)

                loss_ce = loss_ce + lambda_infonce * (loss_infonce_grp + loss_infonce_isr)

        # ── ISR auxiliary BCE loss ─────────────────────────────────────────────
        # Per-jet binary ISR classification.  Directly supervises the ISR head
        # with a one-hot binary target — the one jet that is the ISR gets label 1.
        # This loss provides a strong, clean gradient for the hard-case (high-pT ISR)
        # events that the main flat CE may not learn from quickly.
        if factored and lambda_isr_aux > 0 and "isr_aux_logits" in output:
            isr_labels_aux = model.flat_to_factored[labels, 0]        # (batch,)
            isr_binary = F.one_hot(
                isr_labels_aux, num_classes=model.num_jets
            ).float()                                                  # (batch, J)
            loss_isr_aux = F.binary_cross_entropy_with_logits(
                output["isr_aux_logits"], isr_binary
            )
            loss_ce = loss_ce + lambda_isr_aux * loss_isr_aux

        # Mass symmetry auxiliary loss: minimize expected |m1-m2|/(m1+m2) over assignments
        if lambda_sym > 0 and "mass_asym_flat" in output:
            mass_asym = output["mass_asym_flat"].detach()  # (batch, num_assignments)
            probs = logits.softmax(dim=-1)
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
                # logits are raw combined logits; softmax gives assignment probabilities.
                probs_qcd = logits.softmax(dim=-1)[qcd_mask]
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

        preds = eval_logits.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()

        _, top5 = eval_logits.topk(5, dim=-1)
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
