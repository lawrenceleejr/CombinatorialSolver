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
import datetime
import math
import os
import subprocess
import zipfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
from torch.utils.data import DataLoader, random_split

from .dataset import JetAssignmentDataset
from .export_onnx import export_classical_solver, export_ml_model
from .model import JetAssignmentTransformer
from .utils import get_config, get_device


def _get_git_commit_hash() -> str:
    """Return the short git commit hash of HEAD, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _export_phase1_snapshot(
    checkpoint_path: str,
    num_jets: int,
    val_acc: float,
) -> None:
    """Export phase-1 best model + classical solver as a timestamped ONNX bundle.

    Produces a zip archive at ``onnx_snapshots/phase1_<timestamp>_<commit>.zip``
    containing:
      - ``ml_model_phase1_<timestamp>_<commit>.onnx``   – the ML transformer
      - ``classical_mass_asymmetry_<timestamp>_<commit>.onnx`` – classical solver

    Args:
        checkpoint_path: Path to the phase-1 best-model checkpoint.
        num_jets: Number of jets per event (from the data config).
        val_acc: Best validation accuracy reached during phase 1 (for the log message).
    """
    ts = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    commit = _get_git_commit_hash()
    tag = f"phase1_{ts}_{commit}"

    snapshot_dir = Path("onnx_snapshots") / tag
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    ml_name = f"ml_model_{tag}.onnx"
    classical_name = f"classical_mass_asymmetry_{tag}.onnx"
    ml_path = str(snapshot_dir / ml_name)
    classical_path = str(snapshot_dir / classical_name)

    try:
        export_ml_model(checkpoint_path=checkpoint_path, output_path=ml_path)
    except Exception as exc:
        print(f"  Warning: ML model ONNX export failed: {exc}")
        ml_path = None

    try:
        export_classical_solver(output_path=classical_path, num_jets=num_jets)
    except Exception as exc:
        print(f"  Warning: Classical solver ONNX export failed: {exc}")
        classical_path = None

    zip_path = Path("onnx_snapshots") / f"{tag}.zip"
    exported = [(p, n) for p, n in [(ml_path, ml_name), (classical_path, classical_name)] if p is not None]
    if exported:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path, name in exported:
                zf.write(path, arcname=name)
    else:
        zip_path = None

    print(
        f"\n*** Phase 1 ONNX snapshot ***\n"
        f"  ML model      : {ml_path or 'export failed'}\n"
        f"  Classical     : {classical_path or 'export failed'}\n"
        f"  Bundle        : {zip_path or 'not created (no successful exports)'}\n"
        f"  (val_acc={val_acc:.4f}, commit={commit})\n"
    )


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
        use_mass_asymmetry_labels=dc.get("use_mass_asymmetry_labels", True),
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
            "adv_r2", "lr", "phase",
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
    lambda_distill_max = tc.get("lambda_distill", 2.0)
    lambda_distill_epochs = tc.get("lambda_distill_epochs", 20)
    distill_temperature = tc.get("distill_temperature", 4.0)
    if lambda_distill_max > 0 and lambda_distill_epochs > 0:
        print(
            f"Classical distillation: lambda={lambda_distill_max}, "
            f"decay_epochs={lambda_distill_epochs}, T={distill_temperature}"
        )
    else:
        print("Classical distillation disabled")

    # -------------------------------------------------------------------------
    # Two-phase training setup.
    #   Phase 1: grouping head trained with mass-asymmetry pseudolabels, ISR head
    #            frozen.  For each possible ISR choice the grouping with the lowest
    #            mass asymmetry is used as a CE pseudolabel.  This directly teaches
    #            the grouping scorer the classical heuristic without involving the
    #            frozen ISR head in the loss signal.
    #   Phase 2: full loss (CE + ISR + sym + qcd + decayed distillation).  ISR
    #            head is unfrozen.  Distillation decays from lambda_distill_max
    #            over lambda_distill_epochs measured from the Phase 2 start epoch.
    # Phase 1 is only active when phase1_patience > 0.
    # -------------------------------------------------------------------------
    phase1_patience = tc.get("phase1_patience", 0)
    phase1_active = phase1_patience > 0
    # Phase 1 LR cap: the normal cosine warmup ramps from 0 to base_lr in
    # warmup_epochs, which is calibrated for the Phase 2 multi-component loss.
    # The simple grouping-CE pseudolabel loss diverges at that scale, so we cap
    # Phase 1 LR at phase1_max_lr_fraction * initial_lr.
    phase1_lr_fraction = tc.get("phase1_max_lr_fraction", 0.1)
    training_phase = 1  # 1 or 2
    phase1_best_acc = 0.0
    phase1_no_improve = 0
    phase2_start_epoch = None   # absolute epoch index when Phase 2 begins

    if phase1_active:
        if model.has_isr:
            # Freeze the ISR head and the projection that feeds it.
            # The grouping scorer and main encoder continue to train,
            # learning to score groupings by mass asymmetry.
            _isr_freeze_params = (
                list(model.isr_head.parameters())
                + list(model.grouping_summary_proj.parameters())
            )
            for p in _isr_freeze_params:
                p.requires_grad_(False)
            print(
                f"Phase 1: ISR head frozen. Training grouping head with "
                f"mass-asymmetry pseudolabels "
                f"(patience={phase1_patience} epochs, "
                f"LR capped at {phase1_lr_fraction * base_lr:.1e} before Phase 2)."
            )
        else:
            print(
                f"Phase 1: Training with mass-asymmetry pseudolabels "
                f"(patience={phase1_patience} epochs, "
                f"LR capped at {phase1_lr_fraction * base_lr:.1e} before Phase 2)."
            )

    for epoch in range(tc["num_epochs"]):
        cosine_with_warmup(
            optimizer, epoch, tc["num_epochs"], tc["warmup_epochs"],
            restart_period=tc.get("restart_period", 0),
        )

        # During Phase 1, clamp LR to phase1_lr_fraction × initial_lr.
        # Phase 1 uses a pure 10-class grouping-CE pseudolabel loss (no
        # sym/qcd/adversary/distillation terms), which is stable at the full
        # cosine LR. phase1_max_lr_fraction defaults to 1.0 (no cap) so Phase 1
        # can converge in the ~15-20 epochs before phase1_patience fires.
        # Reduce if Phase 1 shows training-loss divergence.
        if training_phase == 1 and phase1_active:
            for pg in optimizer.param_groups:
                pg["lr"] = min(pg["lr"], pg["initial_lr"] * phase1_lr_fraction)

        current_lr = optimizer.param_groups[0]["lr"]

        if use_adversary and training_phase == 2:
            rampup = tc.get("lambda_adv_rampup", 10)
            phase2_epoch = epoch - (phase2_start_epoch or 0)
            if rampup > 0:
                adv_scale = min(1.0, phase2_epoch / rampup)
            else:
                adv_scale = 1.0
            lambda_adv = tc["lambda_adv"] * adv_scale
            model.gradient_reversal.set_lambda(lambda_adv)
        else:
            lambda_adv = 0.0
            model.gradient_reversal.set_lambda(0.0)

        if training_phase == 1:
            # Phase 1: full-strength distillation, no decay, no other losses
            lambda_distill = lambda_distill_max
            tf_ratio = 0.0          # irrelevant (CE loss is skipped)
            lambda_sym = 0.0
            lambda_qcd = 0.0
            phase1_only_train = True
        else:
            # Phase 2: teacher forcing, auxiliary losses, decaying distillation
            phase2_epoch = epoch - phase2_start_epoch

            # Teacher forcing ratio: linearly decay from tf_start to tf_end
            if phase2_epoch < tf_decay_epochs:
                tf_ratio = tf_start + (tf_end - tf_start) * phase2_epoch / tf_decay_epochs
            else:
                tf_ratio = tf_end

            # Ramp up auxiliary losses from Phase 2 start
            if lambda_sym_rampup > 0:
                lambda_sym = lambda_sym_max * min(1.0, phase2_epoch / lambda_sym_rampup)
            else:
                lambda_sym = lambda_sym_max

            if lambda_qcd_rampup > 0:
                lambda_qcd = lambda_qcd_max * min(1.0, phase2_epoch / lambda_qcd_rampup)
            else:
                lambda_qcd = lambda_qcd_max

            # Distillation decays from max to zero over lambda_distill_epochs
            if lambda_distill_epochs > 0:
                lambda_distill = lambda_distill_max * max(
                    0.0, 1.0 - phase2_epoch / lambda_distill_epochs
                )
            else:
                lambda_distill = 0.0

            phase1_only_train = False

        # Training
        model.train()
        train_metrics = _run_epoch(
            model, train_loader, ce_loss_fn, mse_loss_fn,
            lambda_adv, device, optimizer=optimizer,
            tf_ratio=tf_ratio, lambda_sym=lambda_sym, lambda_qcd=lambda_qcd,
            lambda_isr=lambda_isr,
            lambda_distill=lambda_distill, distill_temperature=distill_temperature,
            phase1_only=phase1_only_train,
        )

        # Validation (no augmentation, no teacher forcing: tf_ratio=0 = pure end-to-end)
        model.eval()
        with torch.no_grad():
            val_metrics = _run_epoch(
                model, val_loader, ce_loss_fn, mse_loss_fn,
                lambda_adv, device, optimizer=None,
                tf_ratio=0.0, lambda_sym=0.0, lambda_qcd=0.0,
                lambda_isr=lambda_isr,
                lambda_distill=0.0, distill_temperature=distill_temperature,
            )

        # Log
        phase_tag = f"[P{training_phase}]" if phase1_active else ""
        adv_str = f" | Adv R²={val_metrics['adv_r2']:.3f}" if use_adversary else ""
        isr_str = ""
        if "isr_acc" in val_metrics:
            isr_str = (
                f" | ISR={val_metrics['isr_acc']:.3f}"
                f" Grp={val_metrics['grp_acc']:.3f}"
            )
        asym_str = (
            f" | AvgAsym={val_metrics['avg_mass_asym']:.4f}"
            f"±{val_metrics['std_mass_asym']:.4f}"
            if "avg_mass_asym" in val_metrics
            else ""
        )

        print(
            f"Epoch {epoch+1:3d}/{tc['num_epochs']} {phase_tag} | "
            f"Train loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.3f} | "
            f"Val loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.3f}"
            f"{isr_str}{adv_str}{asym_str} | "
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
                training_phase,
            ])

        # ---------------------------------------------------------------
        # Phase 1 plateau detection → trigger Phase 2
        # ---------------------------------------------------------------
        if training_phase == 1 and phase1_active:
            # Use grouping accuracy (given truth ISR) as the Phase 1 plateau
            # signal.  During Phase 1 the ISR head is frozen at random initial
            # weights, so the *combined* assignment accuracy (acc) stays near
            # 1/70 ≈ 1.4% regardless of how well the grouping scorer is
            # learning — using acc would trigger Phase 2 prematurely after
            # just a few epochs.  grp_acc measures "given the truth ISR, how
            # often does the grouping head pick the right 3+3 split?", which
            # is exactly what Phase 1 is training.  For 6-jet (no-ISR) models,
            # grp_acc is not reported, so fall back to acc.
            phase1_monitor = val_metrics.get("grp_acc", val_metrics["acc"])
            if phase1_monitor > phase1_best_acc:
                phase1_best_acc = phase1_monitor
                phase1_no_improve = 0
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": phase1_best_acc,
                        "config": config,
                    },
                    "checkpoints/phase1_best_model.pt",
                )
            else:
                phase1_no_improve += 1

            if phase1_no_improve >= phase1_patience:
                training_phase = 2
                phase2_start_epoch = epoch + 1
                print(
                    f"\n*** Phase 1 plateau at epoch {epoch+1} "
                    f"(best grp_acc={phase1_best_acc:.4f}, "
                    f"no improvement for {phase1_patience} epochs). "
                    f"Entering Phase 2: full supervised training. ***\n"
                )
                _export_phase1_snapshot(
                    checkpoint_path="checkpoints/phase1_best_model.pt",
                    num_jets=dc["num_jets"],
                    val_acc=phase1_best_acc,
                )
                if model.has_isr:
                    for p in model.isr_head.parameters():
                        p.requires_grad_(True)
                    for p in model.grouping_summary_proj.parameters():
                        p.requires_grad_(True)
                    print("  ISR head unfrozen.")
                # Reset the Phase 2 early-stopping counter independently.
                no_improve = 0
                best_val_acc = 0.0  # let Phase 2 build its own best checkpoint
            # In Phase 1 we do not apply early stopping — only the plateau
            # detector (phase1_no_improve) controls the transition.
            continue

        # Checkpoint (Phase 2 or single-phase)
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
    lambda_distill=0.0, distill_temperature=4.0, phase1_only=False,
):
    """Run one epoch of training or validation.

    When *phase1_only* is True (Phase 1 training only), the loss is restricted
    to the classical-distillation term.  All other auxiliary losses (CE, ISR,
    sym, qcd, adversary) are skipped so that the model focuses exclusively on
    replicating the argmin-mass-asymmetry heuristic.  Accuracy is still
    measured the normal way (argmax logits vs ground-truth label) in both
    phases so that the plateau detector works correctly.
    """
    total_loss = 0.0
    total_correct = 0
    total_correct5 = 0
    total_isr_correct = 0
    total_grp_correct = 0
    total_samples = 0
    total_mass_asym = 0.0
    total_mass_asym_samples = 0
    all_pred_asym = []
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

        if phase1_only and optimizer is not None:
            # ---------------------------------------------------------------
            # Phase 1 training: teach the grouping head to minimise mass
            # asymmetry using per-ISR-block pseudolabels.
            #
            # Why not KL distillation?  mass_asym ∈ [0, 1], so with T=4 the
            # teacher softmax(-mass_asym / T) spans values in exp(-0.25)…1
            # — a ratio of only ~1.28 across 70 classes.  The resulting KL
            # divergence is ≈ 0, giving essentially zero gradient regardless
            # of temperature or lambda_distill.
            #
            # Instead we use a direct CE loss:
            #   7-jet (ISR): for each of the 7 ISR-block choices, find the
            #     grouping with the lowest mass asymmetry and train grouping_
            #     logits with CE against that per-block pseudolabel.  This
            #     completely bypasses the frozen (random) ISR head.
            #   6-jet (flat): the flat assignment with the lowest mass asym is
            #     used as the pseudolabel for the flat CE loss.
            # ---------------------------------------------------------------
            if "mass_asym_flat" in output:
                if factored and "grouping_logits" in output:
                    # Per-ISR-block pseudolabels.
                    # factored_to_flat[j, k] = flat index for (isr=j, grp=k).
                    # Gather mass_asym into (batch, num_jets, num_groupings).
                    f2flat = model.factored_to_flat           # (num_jets, 10)
                    mass_asym_per_block = output["mass_asym_flat"][:, f2flat]  # (B, J, 10)
                    pseudo_grp = mass_asym_per_block.argmin(dim=-1)            # (B, J)
                    grp_logits = output["grouping_logits"]    # (B, J, 10)
                    loss = ce_loss_fn(
                        grp_logits.reshape(-1, model.num_groupings),
                        pseudo_grp.reshape(-1),
                    )
                else:
                    # 6-jet flat mode: argmin across all 10 assignments
                    pseudo_label = output["mass_asym_flat"].argmin(dim=-1)
                    loss = ce_loss_fn(logits, pseudo_label)
            else:
                # Fallback (mass_asym_flat not available)
                loss = ce_loss_fn(logits, labels)
            loss_adv = torch.tensor(0.0, device=device)

            # Still track factored accuracy metrics for monitoring
            if factored and "isr_logits" in output:
                isr_logits = output["isr_logits"]
                grouping_logits = output["grouping_logits"]
                isr_labels = model.flat_to_factored[labels, 0]
                batch_idx = torch.arange(labels.shape[0], device=device)
                gt_grp_logits = grouping_logits[batch_idx, isr_labels]
                grouping_labels = model.flat_to_factored[labels, 1]
                total_isr_correct += (isr_logits.argmax(dim=-1) == isr_labels).sum().item()
                total_grp_correct += (gt_grp_logits.argmax(dim=-1) == grouping_labels).sum().item()
        else:
            # ---------------------------------------------------------------
            # Phase 2 (or legacy single-phase) training: full loss.
            # ---------------------------------------------------------------

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
                loss_flat = ce_loss_fn(logits, labels)
                loss_ce = tf_ratio * (lambda_isr * loss_isr + loss_grp_tf) + (1.0 - tf_ratio) * loss_flat

                total_isr_correct += (isr_logits.argmax(dim=-1) == isr_labels).sum().item()
                total_grp_correct += (gt_grp_logits.argmax(dim=-1) == grouping_labels).sum().item()
            else:
                loss_ce = ce_loss_fn(logits, labels)

            # Classical distillation loss: pull NN logits toward the classical
            # mass-asymmetry solver (argmin |m1-m2|/(m1+m2) = argmax -mass_asym).
            # mass_asym_flat is scale-invariant so it is unaffected by HT normalisation.
            # No T² rescaling (Hinton et al. 2015): T² is only correct when teacher
            # logits are NN outputs softened by T; here the teacher is -mass_asym
            # (a bounded physics quantity), so T² would over-amplify the KL gradient.
            # This loss decays to zero by lambda_distill_epochs (counted from the
            # Phase 2 start epoch when two-phase training is active).
            if lambda_distill > 0 and "mass_asym_flat" in output:
                T = distill_temperature
                teacher_logits = -output["mass_asym_flat"].detach()  # (batch, num_assignments)
                teacher_probs = F.softmax(teacher_logits / T, dim=-1)
                student_log_probs = F.log_softmax(logits / T, dim=-1)
                # NOTE: we intentionally omit Hinton's T² gradient-restoration factor.
                # T² is only valid when both teacher and student are NN logits scaled
                # by the same temperature T.  Here the teacher is -mass_asym ∈ [-1,0]
                # (a bounded physics quantity, not a NN output).  With T=4 and
                # lambda_distill=2, the T² factor would give an effective KL weight
                # of 32, amplifying the gradient enough to oppose the CE signal
                # (KL pushes student_prob toward uniform teacher ≈ 1/70) and cap
                # the model at ~1.6% accuracy for the entire 20-epoch decay period.
                loss_distill = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
                loss_ce = loss_ce + lambda_distill * loss_distill

            # Mass symmetry auxiliary loss: minimize expected |m1-m2|/(m1+m2) over assignments
            if lambda_sym > 0 and "mass_asym_flat" in output:
                mass_asym = output["mass_asym_flat"].detach()  # (batch, num_assignments)
                probs = logits.softmax(dim=-1)
                loss_sym = (probs * mass_asym).sum(dim=-1).mean()
                loss_ce = loss_ce + lambda_sym * loss_sym

            # QCD hierarchy penalty: events with large pT hierarchies (QCD-like) are pushed
            # to prefer high-mass-asymmetry assignments, making them self-select interpretations
            # that look maximally unlike a symmetric signal decay.
            # loss_qcd = -mean(H_i * expected_mass_asym_i), where H = log(pT_max/pT_min).
            # Minimising this negative quantity increases H-weighted expected asymmetry,
            # disfavouring signal-like (low-asymmetry) interpretations for QCD-dominated events.
            if lambda_qcd > 0 and "mass_asym_flat" in output:
                px_all = four_mom[..., 1]
                py_all = four_mom[..., 2]
                pt_all = torch.sqrt(px_all**2 + py_all**2).clamp(min=1e-8)
                pt_max = pt_all.max(dim=-1).values
                pt_min = pt_all.min(dim=-1).values.clamp(min=1e-8)
                # Clamp H to prevent very large values from degenerate (near-zero pT_min) events
                H = torch.log(pt_max / pt_min).clamp(max=10.0)             # (batch,) hierarchy score

                # Detach mass_asym: we only want to steer the assignment probabilities,
                # not back-propagate through the physics feature computation itself.
                mass_asym_qcd = output["mass_asym_flat"].detach()           # (batch, num_assignments)
                probs_qcd = logits.softmax(dim=-1)
                expected_asym = (probs_qcd * mass_asym_qcd).sum(dim=-1)    # (batch,)
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

        if "mass_asym_flat" in output:
            mass_asym_flat = output["mass_asym_flat"].detach()  # (batch, num_assignments)
            pred_asym = mass_asym_flat.gather(1, preds.unsqueeze(1)).squeeze(1)  # (batch,)
            total_mass_asym += pred_asym.sum().item()
            total_mass_asym_samples += batch_size
            all_pred_asym.append(pred_asym.cpu())

        mass_mask = parent_mass > 0
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
    if total_mass_asym_samples > 0:
        result["avg_mass_asym"] = total_mass_asym / total_mass_asym_samples
        result["std_mass_asym"] = torch.cat(all_pred_asym).std().item()
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
