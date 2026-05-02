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


def _export_onnx_snapshot(
    checkpoint_path: str,
    num_jets: int,
    val_acc: float,
    tag_prefix: str = "final",
    extra_files: "list[Path] | None" = None,
) -> None:
    """Export best model + classical solver as a timestamped ONNX bundle.

    Produces a zip archive at
    ``onnx_snapshots/<tag_prefix>_<timestamp>_<commit>.zip`` containing:
      - ``ml_model_<tag>.onnx``                     – the ML transformer
      - ``classical_mass_asymmetry_<tag>.onnx``      – classical solver
      - any paths listed in *extra_files* (e.g. training-curve PDFs / GIFs)

    Args:
        checkpoint_path: Path to the saved model checkpoint (``.pt`` file).
        num_jets: Number of jets per event (from the data config).
        val_acc: Best validation accuracy reached (used in the log message).
        tag_prefix: String prepended to the snapshot tag (e.g. ``"phase1"``
            or ``"final"``).
        extra_files: Optional list of additional file paths to copy into the
            snapshot directory and include in the zip bundle.
    """
    import shutil

    ts = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    commit = _get_git_commit_hash()
    tag = f"{tag_prefix}_{ts}_{commit}"

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

    # Copy extra files (e.g. training-curve plots) into the snapshot directory.
    extra_entries: list[tuple[str, str]] = []
    for src in (extra_files or []):
        src = Path(src)
        if src.exists():
            dest = snapshot_dir / src.name
            shutil.copy2(src, dest)
            extra_entries.append((str(dest), src.name))
        else:
            print(f"  Warning: extra file not found, skipping: {src}")

    zip_path = Path("onnx_snapshots") / f"{tag}.zip"
    exported = [(p, n) for p, n in [(ml_path, ml_name), (classical_path, classical_name)] if p is not None]
    exported.extend(extra_entries)
    if exported:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path, name in exported:
                zf.write(path, arcname=name)
    else:
        zip_path = None

    label = tag_prefix.replace("_", " ").title()
    extra_summary = "".join(f"  Plot          : {n}\n" for _, n in extra_entries)
    print(
        f"\n*** {label} ONNX snapshot ***\n"
        f"  ML model      : {ml_path or 'export failed'}\n"
        f"  Classical     : {classical_path or 'export failed'}\n"
        f"{extra_summary}"
        f"  Bundle        : {zip_path or 'not created (no successful exports)'}\n"
        f"  (val_acc={val_acc:.4f}, commit={commit})\n"
    )


def _export_phase1_snapshot(
    checkpoint_path: str,
    num_jets: int,
    val_acc: float,
) -> None:
    """Backward-compatible wrapper: export phase-1 snapshot bundle."""
    _export_onnx_snapshot(
        checkpoint_path=checkpoint_path,
        num_jets=num_jets,
        val_acc=val_acc,
        tag_prefix="phase1",
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


def _check_optional_deps() -> None:
    """Warn the user if optional plotting dependencies are missing.

    ``matplotlib`` and ``pillow`` are required for the training-curve PDFs and
    the animated mass-asymmetry GIF.  Neither is needed for the core training
    loop itself, so the user is given the choice to continue without them.
    """
    missing = []
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        missing.append("matplotlib")
    try:
        import PIL  # noqa: F401
    except ImportError:
        missing.append("pillow")

    if not missing:
        return

    pkg_str = " ".join(missing)
    print(
        f"\n  [WARNING] Optional plotting package(s) not installed: {pkg_str}\n"
        f"  Install with:  pip install {pkg_str}\n"
        f"  Without them, training-curve PDFs and the mass-asymmetry GIF\n"
        f"  will be skipped, but training itself will proceed normally.\n"
    )
    try:
        answer = input("  Continue without plotting? [Y/n]: ").strip().lower()
    except (EOFError, OSError):
        # Non-interactive environment (e.g. CI / script redirect) → continue.
        print("  Non-interactive environment detected; continuing without plots.")
        return

    if answer in ("n", "no"):
        raise SystemExit(
            f"Aborted. Install the missing packages and re-run:\n"
            f"  pip install {pkg_str}"
        )


def train(config_path: str | None = None, data_path: str | None = None):
    """Main training function."""
    _check_optional_deps()

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
        group_num_layers=mc.get("group_num_layers", 1),
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
    label_smoothing = tc.get("label_smoothing", 0.0)
    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if label_smoothing > 0:
        print(f"Label smoothing: {label_smoothing}")
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
            "train_avg_mass_asym", "train_std_mass_asym",
            "val_avg_mass_asym", "val_std_mass_asym",
            "train_avg_max_triplet_pt", "train_std_max_triplet_pt",
            "val_avg_max_triplet_pt", "val_std_max_triplet_pt",
            "train_avg_delta_phi", "train_std_delta_phi",
            "val_avg_delta_phi", "val_std_delta_phi",
            "train_avg_democracy", "train_std_democracy",
            "val_avg_democracy", "val_std_democracy",
        ])

    best_val_acc = 0.0
    best_epoch = 0
    patience = tc.get("patience", 25)
    no_improve = 0
    val_asym_history: list = []  # per-epoch list of numpy arrays (val pred_asym_values)
    val_mass_sum_history: list = []  # per-epoch list of numpy arrays (val pred_mass_sum_values)
    val_max_triplet_pt_history: list = []  # per-epoch list of numpy arrays (val max-triplet scalar pT)
    val_delta_phi_history: list = []  # per-epoch list of numpy arrays (val Δφ between parent candidates)
    val_democracy_history: list = []  # per-epoch list of numpy arrays (val avg pT democracy of triplets)

    tf_start = tc.get("tf_start", 1.0)
    tf_end = tc.get("tf_end", 0.3)
    tf_decay_epochs = tc.get("tf_decay_epochs", 100)
    lambda_isr = tc.get("lambda_isr", 1.0)
    lambda_sym_max = tc.get("lambda_sym", 0.0)
    lambda_qcd_max = tc.get("lambda_qcd", 0.0)
    lambda_sym_rampup = tc.get("lambda_sym_rampup", 0)
    lambda_qcd_rampup = tc.get("lambda_qcd_rampup", 0)
    lambda_isr_direct_max = tc.get("lambda_isr_direct", 0.0)
    lambda_isr_direct_rampup = tc.get("lambda_isr_direct_rampup", 0)
    lambda_distill_max = tc.get("lambda_distill", 2.0)
    lambda_distill_epochs = tc.get("lambda_distill_epochs", 20)
    distill_temperature = tc.get("distill_temperature", 4.0)
    # Entropy-weighted physics prior losses (push uncertain events toward
    # QCD-like interpretations with high mass asymmetry and low mass sum):
    #   lambda_entropy_asym: maximises entropy-weighted expected mass asymmetry.
    #   lambda_entropy_mass: minimises entropy-weighted expected mass sum (m1+m2).
    # Both are zero by default; ramp up from Phase 2 start like lambda_sym/qcd.
    lambda_entropy_asym_max = tc.get("lambda_entropy_asym", 0.0)
    lambda_entropy_mass_max = tc.get("lambda_entropy_mass", 0.0)
    lambda_entropy_asym_rampup = tc.get("lambda_entropy_asym_rampup", 0)
    lambda_entropy_mass_rampup = tc.get("lambda_entropy_mass_rampup", 0)
    if lambda_entropy_asym_max > 0 or lambda_entropy_mass_max > 0:
        print(
            f"Entropy-weighted physics prior: "
            f"lambda_entropy_asym={lambda_entropy_asym_max} "
            f"(rampup={lambda_entropy_asym_rampup}), "
            f"lambda_entropy_mass={lambda_entropy_mass_max} "
            f"(rampup={lambda_entropy_mass_rampup})"
        )
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

    # If restart_period is 0 (or not set), default to patience so the cosine
    # cycle length scales with the early-stopping window.  This ensures the LR
    # stays near its minimum for roughly one full patience window before the
    # next warm restart, preventing premature kicks out of a good minimum.
    _restart_period = tc.get("restart_period", 0) or patience

    try:
        for epoch in range(tc["num_epochs"]):
            cosine_with_warmup(
                optimizer, epoch, tc["num_epochs"], tc["warmup_epochs"],
                restart_period=_restart_period,
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
                lambda_isr_direct = 0.0
                lambda_entropy_asym = 0.0
                lambda_entropy_mass = 0.0
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

                if lambda_isr_direct_rampup > 0:
                    lambda_isr_direct = lambda_isr_direct_max * min(
                        1.0, phase2_epoch / lambda_isr_direct_rampup
                    )
                else:
                    lambda_isr_direct = lambda_isr_direct_max

                if lambda_entropy_asym_rampup > 0:
                    lambda_entropy_asym = lambda_entropy_asym_max * min(
                        1.0, phase2_epoch / lambda_entropy_asym_rampup
                    )
                else:
                    lambda_entropy_asym = lambda_entropy_asym_max

                if lambda_entropy_mass_rampup > 0:
                    lambda_entropy_mass = lambda_entropy_mass_max * min(
                        1.0, phase2_epoch / lambda_entropy_mass_rampup
                    )
                else:
                    lambda_entropy_mass = lambda_entropy_mass_max

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
                lambda_isr=lambda_isr, lambda_isr_direct=lambda_isr_direct,
                lambda_distill=lambda_distill, distill_temperature=distill_temperature,
                lambda_entropy_asym=lambda_entropy_asym,
                lambda_entropy_mass=lambda_entropy_mass,
                phase1_only=phase1_only_train,
                pt_smear_frac=dc.get("pt_smear_frac", 0.0),
            )

            # Validation (no φ/η augmentation, no teacher forcing: tf_ratio=0 = pure
            # end-to-end; pT smearing is applied if configured, matching training conditions)
            model.eval()
            with torch.no_grad():
                val_metrics = _run_epoch(
                    model, val_loader, ce_loss_fn, mse_loss_fn,
                    lambda_adv, device, optimizer=None,
                    tf_ratio=0.0, lambda_sym=0.0, lambda_qcd=0.0,
                    lambda_isr=lambda_isr, lambda_isr_direct=0.0,
                    lambda_distill=0.0, distill_temperature=distill_temperature,
                    lambda_entropy_asym=0.0, lambda_entropy_mass=0.0,
                    pt_smear_frac=dc.get("pt_smear_frac", 0.0),
                )

            # Accumulate per-event validation mass-asymmetry distribution for GIF
            if "pred_asym_values" in val_metrics:
                val_asym_history.append((
                    epoch + 1, training_phase,
                    val_metrics["pred_asym_values"],
                    val_metrics.get("pred_correct_values"),  # bool array or None
                ))

            # Accumulate per-event validation mass-sum distribution for GIF
            if "pred_mass_sum_values" in val_metrics:
                val_mass_sum_history.append((
                    epoch + 1, training_phase,
                    val_metrics["pred_mass_sum_values"],
                    val_metrics.get("pred_correct_values"),  # bool array or None
                ))

            # Accumulate per-event validation max-triplet scalar-pT distribution for GIF
            if "pred_max_triplet_pt_values" in val_metrics:
                val_max_triplet_pt_history.append((
                    epoch + 1, training_phase,
                    val_metrics["pred_max_triplet_pt_values"],
                    val_metrics.get("pred_correct_values"),  # bool array or None
                ))

            # Accumulate per-event validation ΔΦ distribution for GIF
            if "pred_delta_phi_values" in val_metrics:
                val_delta_phi_history.append((
                    epoch + 1, training_phase,
                    val_metrics["pred_delta_phi_values"],
                    val_metrics.get("pred_correct_values"),  # bool array or None
                ))

            # Accumulate per-event validation pT-democracy distribution for GIF
            if "pred_democracy_values" in val_metrics:
                val_democracy_history.append((
                    epoch + 1, training_phase,
                    val_metrics["pred_democracy_values"],
                    val_metrics.get("pred_correct_values"),  # bool array or None
                ))

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
                    f"{train_metrics['avg_mass_asym']:.6f}" if "avg_mass_asym" in train_metrics else "",
                    f"{train_metrics['std_mass_asym']:.6f}" if "std_mass_asym" in train_metrics else "",
                    f"{val_metrics['avg_mass_asym']:.6f}" if "avg_mass_asym" in val_metrics else "",
                    f"{val_metrics['std_mass_asym']:.6f}" if "std_mass_asym" in val_metrics else "",
                    f"{train_metrics['avg_max_triplet_pt']:.6f}" if "avg_max_triplet_pt" in train_metrics else "",
                    f"{train_metrics['std_max_triplet_pt']:.6f}" if "std_max_triplet_pt" in train_metrics else "",
                    f"{val_metrics['avg_max_triplet_pt']:.6f}" if "avg_max_triplet_pt" in val_metrics else "",
                    f"{val_metrics['std_max_triplet_pt']:.6f}" if "std_max_triplet_pt" in val_metrics else "",
                    f"{train_metrics['avg_delta_phi']:.6f}" if "avg_delta_phi" in train_metrics else "",
                    f"{train_metrics['std_delta_phi']:.6f}" if "std_delta_phi" in train_metrics else "",
                    f"{val_metrics['avg_delta_phi']:.6f}" if "avg_delta_phi" in val_metrics else "",
                    f"{val_metrics['std_delta_phi']:.6f}" if "std_delta_phi" in val_metrics else "",
                    f"{train_metrics['avg_democracy']:.6f}" if "avg_democracy" in train_metrics else "",
                    f"{train_metrics['std_democracy']:.6f}" if "std_democracy" in train_metrics else "",
                    f"{val_metrics['avg_democracy']:.6f}" if "avg_democracy" in val_metrics else "",
                    f"{val_metrics['std_democracy']:.6f}" if "std_democracy" in val_metrics else "",
                ])

            # Per-epoch live-monitoring plots (overwrite fixed "latest" files so a
            # viewer that auto-refreshes (e.g. an open PDF) always shows current progress).
            _plot_training_curves(log_path, phase2_start_epoch=phase2_start_epoch, tag="latest")
            if val_asym_history:
                _make_mass_asym_gif(
                    val_asym_history,
                    phase2_start_epoch=phase2_start_epoch,
                    gif_path=Path("plots") / "mass_asym_anim_latest.gif",
                )
            if val_mass_sum_history:
                _make_mass_sum_gif(
                    val_mass_sum_history,
                    phase2_start_epoch=phase2_start_epoch,
                    gif_path=Path("plots") / "mass_sum_anim_latest.gif",
                )
            if val_max_triplet_pt_history:
                _make_max_triplet_pt_gif(
                    val_max_triplet_pt_history,
                    phase2_start_epoch=phase2_start_epoch,
                    gif_path=Path("plots") / "max_triplet_pt_anim_latest.gif",
                )
            if val_delta_phi_history:
                _make_delta_phi_gif(
                    val_delta_phi_history,
                    phase2_start_epoch=phase2_start_epoch,
                    gif_path=Path("plots") / "delta_phi_anim_latest.gif",
                )
            if val_democracy_history:
                _make_democracy_gif(
                    val_democracy_history,
                    phase2_start_epoch=phase2_start_epoch,
                    gif_path=Path("plots") / "democracy_anim_latest.gif",
                )

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

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Exporting best model...")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    final_checkpoint = "checkpoints/best_model.pt"
    if not Path(final_checkpoint).exists():
        # This can happen when phase1_active=True and Phase 2 never ran or never
        # improved (so best_model.pt was never written).  Fall back to the Phase 1
        # best checkpoint so that at least some model is exported.
        fallback = "checkpoints/phase1_best_model.pt"
        if Path(fallback).exists():
            print(
                f"  Warning: {final_checkpoint} not found; "
                f"falling back to {fallback} for ONNX export."
            )
            final_checkpoint = fallback
        else:
            print(
                f"  Warning: neither {final_checkpoint} nor {fallback} found. "
                f"Skipping ONNX export."
            )
            return

    # Reload best checkpoint before ONNX export
    ckpt = torch.load(final_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Plain ONNX for quick access at the well-known path
    export_onnx(model, dc["num_jets"], device, best_val_acc)

    # Generate training-curve plots and the mass-asymmetry GIF first so that
    # they can be bundled into the ONNX snapshot zip below.
    plot_paths: list = []

    # Training-curve plots (loss and accuracy vs epoch) with phase transition markers.
    plot_paths.extend(
        _plot_training_curves(log_path, phase2_start_epoch=phase2_start_epoch) or []
    )

    # Animated GIF of the validation mass-asymmetry distribution.
    if val_asym_history:
        gif = _make_mass_asym_gif(val_asym_history, phase2_start_epoch=phase2_start_epoch)
        if gif is not None:
            plot_paths.append(gif)

    # Animated GIF of the validation average candidate mass distribution.
    if val_mass_sum_history:
        mass_sum_gif = _make_mass_sum_gif(val_mass_sum_history, phase2_start_epoch=phase2_start_epoch)
        if mass_sum_gif is not None:
            plot_paths.append(mass_sum_gif)

    # Animated GIF of the validation max-triplet scalar-sum pT distribution.
    if val_max_triplet_pt_history:
        mpt_gif = _make_max_triplet_pt_gif(val_max_triplet_pt_history, phase2_start_epoch=phase2_start_epoch)
        if mpt_gif is not None:
            plot_paths.append(mpt_gif)

    # Animated GIF of the validation ΔΦ between parent candidates.
    if val_delta_phi_history:
        dphi_gif = _make_delta_phi_gif(val_delta_phi_history, phase2_start_epoch=phase2_start_epoch)
        if dphi_gif is not None:
            plot_paths.append(dphi_gif)

    # Animated GIF of the validation pT democracy.
    if val_democracy_history:
        dem_gif = _make_democracy_gif(val_democracy_history, phase2_start_epoch=phase2_start_epoch)
        if dem_gif is not None:
            plot_paths.append(dem_gif)

    # Full timestamped snapshot bundle (ML model + classical solver + plots),
    # mirroring the Phase 1 snapshot produced by _export_phase1_snapshot.
    _export_onnx_snapshot(
        checkpoint_path=final_checkpoint,
        num_jets=dc["num_jets"],
        val_acc=best_val_acc,
        tag_prefix="final",
        extra_files=plot_paths,
    )


def _make_mass_asym_gif(
    val_asym_history: list,
    phase2_start_epoch: int | None = None,
    gif_path: str | Path | None = None,
) -> "Path | None":
    """Build an animated GIF of the validation mass-asymmetry distribution.

    Each frame shows a histogram of log₁₀(mass asymmetry) for the model's
    chosen interpretation across all validation events for that epoch.  When a
    per-event correctness mask is available (4-tuple history entries) the bars
    are stacked by correct vs incorrect network outputs.  A vertical line marks
    the per-epoch mean.  When two-phase training was used, frames from Phase 2
    onward carry a "Phase 2" annotation so the transition is immediately
    visible.

    When *gif_path* is ``None`` the file is written to
    ``plots/mass_asym_anim_{timestamp}_{commit}.gif``; pass an explicit path
    (e.g. ``plots/mass_asym_anim_latest.gif``) to overwrite a fixed file on
    every call and requires ``pillow`` (pip install pillow).

    Args:
        val_asym_history: List of ``(epoch, phase, values_array[, correct_mask])``
            tuples collected during training, one entry per epoch.  The optional
            fourth element is a boolean NumPy array aligned with *values_array*
            (True = model chose the correct assignment).
        phase2_start_epoch: 1-based epoch index at which Phase 2 began, or
            ``None`` for single-phase runs.
        gif_path: Destination file path.  When ``None`` a timestamped path
            inside ``plots/`` is generated automatically.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
    except ImportError:
        print("  Warning: matplotlib not available; skipping mass-asym GIF.")
        return None

    if not val_asym_history:
        return None

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    if gif_path is None:
        ts = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        commit = _get_git_commit_hash()
        gif_path = plots_dir / f"mass_asym_anim_{ts}_{commit}.gif"
    gif_path = Path(gif_path)

    import numpy as np

    # Unpack history: support both 3-tuple (legacy) and 4-tuple (with correct mask).
    def _unpack(entry):
        if len(entry) == 4:
            return entry
        return entry[0], entry[1], entry[2], None

    # x-axis in log10 space: mass_asym ∈ (0, 1] → log10 ∈ (-∞, 0].
    # Clip values below 1e-4 to avoid -inf.
    LOG_CLIP = 1e-4
    x_min, x_max = -4.0, 0.0
    n_bins = 50
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)

    # Pre-compute stacked counts for every frame to fix the y-axis.
    max_count = 0
    for entry in val_asym_history:
        _, _, values, correct_mask = _unpack(entry)
        log_vals = np.log10(np.clip(values, LOG_CLIP, 1.0))
        counts_total, _ = np.histogram(log_vals, bins=bin_edges)
        if counts_total.max() > max_count:
            max_count = int(counts_total.max())
    y_max = max_count * 1.1

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = (x_max - x_min) / n_bins
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    def _draw_frame(frame_idx):
        _, _, values, correct_mask = _unpack(val_asym_history[frame_idx])
        epoch, phase = val_asym_history[frame_idx][0], val_asym_history[frame_idx][1]
        ax.cla()
        log_vals = np.log10(np.clip(values, LOG_CLIP, 1.0))
        mean_val = float(log_vals.mean())

        if correct_mask is not None:
            log_correct   = log_vals[correct_mask]
            log_incorrect = log_vals[~correct_mask]
            counts_correct,   _ = np.histogram(log_correct,   bins=bin_edges)
            counts_incorrect, _ = np.histogram(log_incorrect, bins=bin_edges)
            ax.bar(centers, counts_correct,   width=bar_width,
                   color="steelblue", alpha=0.85, align="center", label="Correct")
            ax.bar(centers, counts_incorrect, width=bar_width,
                   color="coral",     alpha=0.85, align="center", label="Incorrect",
                   bottom=counts_correct)
        else:
            counts, _ = np.histogram(log_vals, bins=bin_edges)
            ax.bar(centers, counts, width=bar_width,
                   color="steelblue", alpha=0.75, align="center")

        ax.axvline(mean_val, color="darkorange", linewidth=2.0,
                   label=f"Mean = {mean_val:.2f}")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("log₁₀(mass asymmetry of chosen interpretation)")
        ax.set_ylabel("Validation events")
        phase_label = ""
        if phase2_start_epoch is not None:
            phase_label = " [Phase 2]" if epoch >= phase2_start_epoch else " [Phase 1]"
        elif phase == 2:
            phase_label = " [Phase 2]"
        ax.set_title(f"Epoch {epoch}{phase_label}")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    anim = animation.FuncAnimation(
        fig,
        _draw_frame,
        frames=[0] * 4 + list(range(len(val_asym_history))),
        interval=200,
        repeat=False,
    )

    try:
        anim.save(str(gif_path), writer="pillow", fps=5)
        print(f"  -> Saved mass asym GIF : {gif_path}")
        return gif_path
    except Exception as exc:
        print(f"  Warning: could not save mass-asym GIF ({exc}). "
              "Is pillow installed?  pip install pillow")
        return None
    finally:
        plt.close(fig)


def _make_mass_sum_gif(
    val_mass_sum_history: list,
    phase2_start_epoch: int | None = None,
    gif_path: str | Path | None = None,
) -> "Path | None":
    """Build an animated GIF of the validation average-candidate-mass distribution.

    Each frame shows a histogram of ``(m₁ + m₂) / 2`` for the model's chosen
    interpretation across all validation events for that epoch.  When a
    per-event correctness mask is available (4-tuple history entries) the bars
    are stacked by correct vs incorrect network outputs.  A vertical line marks
    the per-epoch mean.  When two-phase training was used, frames from Phase 2
    onward carry a "Phase 2" annotation so the transition is immediately
    visible.

    When *gif_path* is ``None`` the file is written to
    ``plots/mass_sum_anim_{timestamp}_{commit}.gif``; pass an explicit path
    (e.g. ``plots/mass_sum_anim_latest.gif``) to overwrite a fixed file on
    every call.  Requires ``pillow`` (pip install pillow).

    Args:
        val_mass_sum_history: List of ``(epoch, phase, values_array[, correct_mask])``
            tuples where *values_array* contains ``mass_sum_flat`` at the
            predicted assignment for each validation event.  The optional
            fourth element is a boolean NumPy array aligned with *values_array*
            (True = model chose the correct assignment).
        phase2_start_epoch: 1-based epoch index at which Phase 2 began, or
            ``None`` for single-phase runs.
        gif_path: Destination file path.  When ``None`` a timestamped path
            inside ``plots/`` is generated automatically.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
    except ImportError:
        print("  Warning: matplotlib not available; skipping mass-sum GIF.")
        return None

    if not val_mass_sum_history:
        return None

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    if gif_path is None:
        ts = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        commit = _get_git_commit_hash()
        gif_path = plots_dir / f"mass_sum_anim_{ts}_{commit}.gif"
    gif_path = Path(gif_path)

    import numpy as np

    # Unpack history: support both 3-tuple (legacy) and 4-tuple (with correct mask).
    def _unpack(entry):
        if len(entry) == 4:
            return entry
        return entry[0], entry[1], entry[2], None

    # The average mass per candidate is mass_sum / 2.
    all_avg_mass = [v / 2.0 for *_, v, _ in [_unpack(e) for e in val_mass_sum_history]]

    # Fixed x-axis determined from the global data range (1st–99th percentile).
    all_concat = np.concatenate(all_avg_mass)
    x_min = float(np.percentile(all_concat, 1))
    x_max = float(np.percentile(all_concat, 99))
    if x_min >= x_max:
        x_min, x_max = float(all_concat.min()), float(all_concat.max())
    n_bins = 50
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)

    # Pre-compute total counts for every frame to fix the y-axis.
    max_count = 0
    for avg_mass in all_avg_mass:
        counts, _ = np.histogram(avg_mass, bins=bin_edges)
        if counts.max() > max_count:
            max_count = int(counts.max())
    y_max = max_count * 1.1

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = (x_max - x_min) / n_bins
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    def _draw_frame(frame_idx):
        epoch, phase, values, correct_mask = _unpack(val_mass_sum_history[frame_idx])
        avg_mass = values / 2.0
        ax.cla()
        mean_val = float(avg_mass.mean())

        if correct_mask is not None:
            avg_correct   = avg_mass[correct_mask]
            avg_incorrect = avg_mass[~correct_mask]
            counts_correct,   _ = np.histogram(avg_correct,   bins=bin_edges)
            counts_incorrect, _ = np.histogram(avg_incorrect, bins=bin_edges)
            ax.bar(centers, counts_correct,   width=bar_width,
                   color="mediumseagreen", alpha=0.85, align="center", label="Correct")
            ax.bar(centers, counts_incorrect, width=bar_width,
                   color="coral",           alpha=0.85, align="center", label="Incorrect",
                   bottom=counts_correct)
        else:
            counts, _ = np.histogram(avg_mass, bins=bin_edges)
            ax.bar(centers, counts, width=bar_width,
                   color="mediumseagreen", alpha=0.75, align="center")

        ax.axvline(mean_val, color="darkorange", linewidth=2.0,
                   label=f"Mean = {mean_val:.3f}")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Average candidate mass (m₁+m₂)/2 of chosen interpretation")
        ax.set_ylabel("Validation events")
        phase_label = ""
        if phase2_start_epoch is not None:
            phase_label = " [Phase 2]" if epoch >= phase2_start_epoch else " [Phase 1]"
        elif phase == 2:
            phase_label = " [Phase 2]"
        ax.set_title(f"Epoch {epoch}{phase_label}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    anim = animation.FuncAnimation(
        fig,
        _draw_frame,
        frames=[0] * 4 + list(range(len(val_mass_sum_history))),
        interval=200,
        repeat=False,
    )

    try:
        anim.save(str(gif_path), writer="pillow", fps=5)
        print(f"  -> Saved mass sum GIF  : {gif_path}")
        return gif_path
    except Exception as exc:
        print(f"  Warning: could not save mass-sum GIF ({exc}). "
              "Is pillow installed?  pip install pillow")
        return None
    finally:
        plt.close(fig)


def _make_max_triplet_pt_gif(
    val_max_triplet_pt_history: list,
    phase2_start_epoch: int | None = None,
    gif_path: str | Path | None = None,
) -> "Path | None":
    """Build an animated GIF of the max-triplet scalar-sum-pT distribution.

    For each event the scalar sum pT of each of the two triplets in the
    predicted interpretation is computed and the larger of the two is
    recorded.  Each frame shows the histogram of this quantity over all
    validation events for that epoch.  When a per-event correctness mask is
    available (4-tuple history entries) the bars are stacked by correct vs
    incorrect network outputs.  A vertical line marks the per-epoch mean.
    When two-phase training was used, frames from Phase 2 onward carry a
    "Phase 2" annotation so the transition is immediately visible.

    When *gif_path* is ``None`` the file is written to
    ``plots/max_triplet_pt_anim_{timestamp}_{commit}.gif``; pass an explicit
    path (e.g. ``plots/max_triplet_pt_anim_latest.gif``) to overwrite a fixed
    file on every call.  Requires ``pillow`` (pip install pillow).

    Args:
        val_max_triplet_pt_history: List of ``(epoch, phase, values_array[, correct_mask])``
            tuples where *values_array* contains the max-triplet scalar-sum pT at the
            predicted assignment for each validation event.  The optional
            fourth element is a boolean NumPy array aligned with *values_array*
            (True = model chose the correct assignment).
        phase2_start_epoch: 1-based epoch index at which Phase 2 began, or
            ``None`` for single-phase runs.
        gif_path: Destination file path.  When ``None`` a timestamped path
            inside ``plots/`` is generated automatically.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
    except ImportError:
        print("  Warning: matplotlib not available; skipping max-triplet-pT GIF.")
        return None

    if not val_max_triplet_pt_history:
        return None

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    if gif_path is None:
        ts = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        commit = _get_git_commit_hash()
        gif_path = plots_dir / f"max_triplet_pt_anim_{ts}_{commit}.gif"
    gif_path = Path(gif_path)

    import numpy as np

    # Unpack history: support both 3-tuple (legacy) and 4-tuple (with correct mask).
    def _unpack(entry):
        if len(entry) == 4:
            return entry
        return entry[0], entry[1], entry[2], None

    all_values = [_unpack(e)[2] for e in val_max_triplet_pt_history]

    # Fixed x-axis determined from the global data range (1st–99th percentile).
    all_concat = np.concatenate(all_values)
    x_min = float(np.percentile(all_concat, 1))
    x_max = float(np.percentile(all_concat, 99))
    if x_min >= x_max:
        x_min, x_max = float(all_concat.min()), float(all_concat.max())
    n_bins = 50
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)

    # Pre-compute total counts for every frame to fix the y-axis.
    max_count = 0
    for vals in all_values:
        counts, _ = np.histogram(vals, bins=bin_edges)
        if counts.max() > max_count:
            max_count = int(counts.max())
    y_max = max_count * 1.1

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = (x_max - x_min) / n_bins
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    def _draw_frame(frame_idx):
        epoch, phase, values, correct_mask = _unpack(val_max_triplet_pt_history[frame_idx])
        ax.cla()
        mean_val = float(values.mean())

        if correct_mask is not None:
            vals_correct   = values[correct_mask]
            vals_incorrect = values[~correct_mask]
            counts_correct,   _ = np.histogram(vals_correct,   bins=bin_edges)
            counts_incorrect, _ = np.histogram(vals_incorrect, bins=bin_edges)
            ax.bar(centers, counts_correct,   width=bar_width,
                   color="mediumorchid", alpha=0.85, align="center", label="Correct")
            ax.bar(centers, counts_incorrect, width=bar_width,
                   color="coral",        alpha=0.85, align="center", label="Incorrect",
                   bottom=counts_correct)
        else:
            counts, _ = np.histogram(values, bins=bin_edges)
            ax.bar(centers, counts, width=bar_width,
                   color="mediumorchid", alpha=0.75, align="center")

        ax.axvline(mean_val, color="darkorange", linewidth=2.0,
                   label=f"Mean = {mean_val:.3f}")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Max-triplet scalar sum pT of chosen interpretation")
        ax.set_ylabel("Validation events")
        phase_label = ""
        if phase2_start_epoch is not None:
            phase_label = " [Phase 2]" if epoch >= phase2_start_epoch else " [Phase 1]"
        elif phase == 2:
            phase_label = " [Phase 2]"
        ax.set_title(f"Epoch {epoch}{phase_label}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    anim = animation.FuncAnimation(
        fig,
        _draw_frame,
        frames=[0] * 4 + list(range(len(val_max_triplet_pt_history))),
        interval=200,
        repeat=False,
    )

    try:
        anim.save(str(gif_path), writer="pillow", fps=5)
        print(f"  -> Saved max-triplet-pT GIF: {gif_path}")
        return gif_path
    except Exception as exc:
        print(f"  Warning: could not save max-triplet-pT GIF ({exc}). "
              "Is pillow installed?  pip install pillow")
        return None
    finally:
        plt.close(fig)


def _make_delta_phi_gif(
    val_delta_phi_history: list,
    phase2_start_epoch: int | None = None,
    gif_path: str | Path | None = None,
) -> "Path | None":
    """Build an animated GIF of the Δφ distribution between the two parent candidates.

    For each event Δφ = |φ(triplet1) − φ(triplet2)| is folded into [0, π],
    where φᵢ = atan2(ΣPy, ΣPx) over the jets assigned to candidate i.  Each
    frame shows a histogram over all validation events for that epoch.  When a
    per-event correctness mask is available (4-tuple history entries) the bars
    are stacked by correct vs incorrect network outputs.  A vertical line marks
    the per-epoch mean.  When two-phase training was used, frames from Phase 2
    onward carry a "Phase 2" annotation.

    When *gif_path* is ``None`` the file is written to
    ``plots/delta_phi_anim_{timestamp}_{commit}.gif``.  Requires ``pillow``
    (pip install pillow).

    Args:
        val_delta_phi_history: List of ``(epoch, phase, values_array[, correct_mask])``
            tuples where *values_array* contains the per-event Δφ in [0, π].
            The optional fourth element is a boolean NumPy array (True = correct).
        phase2_start_epoch: 1-based epoch index at which Phase 2 began, or
            ``None`` for single-phase runs.
        gif_path: Destination file path.  When ``None`` a timestamped path
            inside ``plots/`` is generated automatically.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
    except ImportError:
        print("  Warning: matplotlib not available; skipping Δφ GIF.")
        return None

    if not val_delta_phi_history:
        return None

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    if gif_path is None:
        ts = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        commit = _get_git_commit_hash()
        gif_path = plots_dir / f"delta_phi_anim_{ts}_{commit}.gif"
    gif_path = Path(gif_path)

    import numpy as np
    import math as _math_gif

    def _unpack(entry):
        if len(entry) == 4:
            return entry
        return entry[0], entry[1], entry[2], None

    x_min, x_max = 0.0, _math_gif.pi
    n_bins = 50
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)

    # Pre-compute total counts to fix the y-axis.
    max_count = 0
    for entry in val_delta_phi_history:
        _, _, values, _ = _unpack(entry)
        counts, _ = np.histogram(values, bins=bin_edges)
        if counts.max() > max_count:
            max_count = int(counts.max())
    y_max = max_count * 1.1

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = (x_max - x_min) / n_bins
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    def _draw_frame(frame_idx):
        epoch, phase, values, correct_mask = _unpack(val_delta_phi_history[frame_idx])
        ax.cla()
        mean_val = float(values.mean())

        if correct_mask is not None:
            vals_correct   = values[correct_mask]
            vals_incorrect = values[~correct_mask]
            counts_correct,   _ = np.histogram(vals_correct,   bins=bin_edges)
            counts_incorrect, _ = np.histogram(vals_incorrect, bins=bin_edges)
            ax.bar(centers, counts_correct,   width=bar_width,
                   color="steelblue", alpha=0.85, align="center", label="Correct")
            ax.bar(centers, counts_incorrect, width=bar_width,
                   color="coral",     alpha=0.85, align="center", label="Incorrect",
                   bottom=counts_correct)
        else:
            counts, _ = np.histogram(values, bins=bin_edges)
            ax.bar(centers, counts, width=bar_width,
                   color="steelblue", alpha=0.75, align="center")

        ax.axvline(mean_val, color="darkorange", linewidth=2.0,
                   label=f"Mean = {mean_val:.3f}")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Δφ between parent candidates (rad)")
        ax.set_ylabel("Validation events")
        phase_label = ""
        if phase2_start_epoch is not None:
            phase_label = " [Phase 2]" if epoch >= phase2_start_epoch else " [Phase 1]"
        elif phase == 2:
            phase_label = " [Phase 2]"
        ax.set_title(f"Epoch {epoch}{phase_label}")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    anim = animation.FuncAnimation(
        fig,
        _draw_frame,
        frames=[0] * 4 + list(range(len(val_delta_phi_history))),
        interval=200,
        repeat=False,
    )

    try:
        anim.save(str(gif_path), writer="pillow", fps=5)
        print(f"  -> Saved Δφ GIF        : {gif_path}")
        return gif_path
    except Exception as exc:
        print(f"  Warning: could not save Δφ GIF ({exc}). "
              "Is pillow installed?  pip install pillow")
        return None
    finally:
        plt.close(fig)


def _make_democracy_gif(
    val_democracy_history: list,
    phase2_start_epoch: int | None = None,
    gif_path: str | Path | None = None,
) -> "Path | None":
    """Build an animated GIF of the pT-democracy distribution.

    For each event the pT-democracy of a triplet is defined as
    ``min(pT) / max(pT)`` over its three jets, which equals 1 when all jets
    carry equal pT and approaches 0 when one jet dominates.  The per-event
    score is the average democracy across the two triplets in the predicted
    assignment.  Each frame shows a histogram over all validation events for
    that epoch.  When a per-event correctness mask is available (4-tuple
    history entries) the bars are stacked by correct vs incorrect outputs.
    A vertical line marks the per-epoch mean.  When two-phase training was
    used, frames from Phase 2 onward carry a "Phase 2" annotation.

    When *gif_path* is ``None`` the file is written to
    ``plots/democracy_anim_{timestamp}_{commit}.gif``.  Requires ``pillow``
    (pip install pillow).

    Args:
        val_democracy_history: List of ``(epoch, phase, values_array[, correct_mask])``
            tuples where *values_array* contains the per-event average pT
            democracy in (0, 1].  The optional fourth element is a boolean
            NumPy array (True = correct).
        phase2_start_epoch: 1-based epoch index at which Phase 2 began, or
            ``None`` for single-phase runs.
        gif_path: Destination file path.  When ``None`` a timestamped path
            inside ``plots/`` is generated automatically.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
    except ImportError:
        print("  Warning: matplotlib not available; skipping democracy GIF.")
        return None

    if not val_democracy_history:
        return None

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    if gif_path is None:
        ts = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        commit = _get_git_commit_hash()
        gif_path = plots_dir / f"democracy_anim_{ts}_{commit}.gif"
    gif_path = Path(gif_path)

    import numpy as np

    def _unpack(entry):
        if len(entry) == 4:
            return entry
        return entry[0], entry[1], entry[2], None

    x_min, x_max = 0.0, 1.0
    n_bins = 50
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)

    # Pre-compute total counts to fix the y-axis.
    max_count = 0
    for entry in val_democracy_history:
        _, _, values, _ = _unpack(entry)
        counts, _ = np.histogram(values, bins=bin_edges)
        if counts.max() > max_count:
            max_count = int(counts.max())
    y_max = max_count * 1.1

    fig, ax = plt.subplots(figsize=(8, 5))
    bar_width = (x_max - x_min) / n_bins
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    def _draw_frame(frame_idx):
        epoch, phase, values, correct_mask = _unpack(val_democracy_history[frame_idx])
        ax.cla()
        mean_val = float(values.mean())

        if correct_mask is not None:
            vals_correct   = values[correct_mask]
            vals_incorrect = values[~correct_mask]
            counts_correct,   _ = np.histogram(vals_correct,   bins=bin_edges)
            counts_incorrect, _ = np.histogram(vals_incorrect, bins=bin_edges)
            ax.bar(centers, counts_correct,   width=bar_width,
                   color="mediumseagreen", alpha=0.85, align="center", label="Correct")
            ax.bar(centers, counts_incorrect, width=bar_width,
                   color="coral",           alpha=0.85, align="center", label="Incorrect",
                   bottom=counts_correct)
        else:
            counts, _ = np.histogram(values, bins=bin_edges)
            ax.bar(centers, counts, width=bar_width,
                   color="mediumseagreen", alpha=0.75, align="center")

        ax.axvline(mean_val, color="darkorange", linewidth=2.0,
                   label=f"Mean = {mean_val:.3f}")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("pT democracy = avg(min pT / max pT) per triplet")
        ax.set_ylabel("Validation events")
        phase_label = ""
        if phase2_start_epoch is not None:
            phase_label = " [Phase 2]" if epoch >= phase2_start_epoch else " [Phase 1]"
        elif phase == 2:
            phase_label = " [Phase 2]"
        ax.set_title(f"Epoch {epoch}{phase_label}")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

    anim = animation.FuncAnimation(
        fig,
        _draw_frame,
        frames=[0] * 4 + list(range(len(val_democracy_history))),
        interval=200,
        repeat=False,
    )

    try:
        anim.save(str(gif_path), writer="pillow", fps=5)
        print(f"  -> Saved democracy GIF : {gif_path}")
        return gif_path
    except Exception as exc:
        print(f"  Warning: could not save democracy GIF ({exc}). "
              "Is pillow installed?  pip install pillow")
        return None
    finally:
        plt.close(fig)


def _plot_training_curves(
    log_path: str | Path,
    phase2_start_epoch: int | None = None,
    tag: str | None = None,
) -> "list[Path]":
    """Generate loss, accuracy, learning-rate, and mass-asymmetry plots from the training log CSV.

    Creates PDF files in a ``plots/`` directory:
      - ``loss_{tag}.pdf``         – train and validation loss vs epoch
      - ``lr_{tag}.pdf``           – learning rate vs epoch
      - ``accuracy_{tag}.pdf``     – train and validation accuracy vs epoch
      - ``mass_asym_{tag}.pdf``    – mean ± 1σ mass asymmetry of the chosen
                                     interpretation vs epoch (train and val)

    Vertical dashed lines mark phase transitions (Phase 1 → Phase 2) when
    two-phase training was used.  Files are named with a UTC timestamp and
    the short git commit hash for traceability.

    Args:
        log_path: Path to the training log CSV file written during training.
        phase2_start_epoch: The (1-based) epoch at which Phase 2 began, or
            ``None`` if single-phase training was used.
        tag: File-name suffix used for the output PDFs.  When ``None`` a
            timestamp + commit hash is generated automatically.  Pass a fixed
            string (e.g. ``"latest"``) to overwrite the same files on every
            call, which is useful for live monitoring during training.

    Returns:
        List of :class:`~pathlib.Path` objects for the PDF files that were
        successfully saved.  Empty list when plots could not be generated.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend, safe in all environments
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Warning: matplotlib not available; skipping training curve plots.")
        return []

    log_path = Path(log_path)
    if not log_path.exists():
        print(f"  Warning: training log not found at {log_path}; skipping plots.")
        return []

    # --- Read CSV ---
    epochs, train_loss, val_loss, train_acc, val_acc, phases = [], [], [], [], [], []
    train_avg_asym, train_std_asym, val_avg_asym, val_std_asym = [], [], [], []
    train_grp_acc, val_grp_acc = [], []
    train_avg_mpt, train_std_mpt, val_avg_mpt, val_std_mpt = [], [], [], []
    train_avg_dphi, train_std_dphi, val_avg_dphi, val_std_dphi = [], [], [], []
    train_avg_dem, train_std_dem, val_avg_dem, val_std_dem = [], [], [], []
    lr_values: list[float] = []
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            train_acc.append(float(row["train_acc"]))
            val_acc.append(float(row["val_acc"]))
            phases.append(int(row["phase"]))
            # LR column – present in all current logs; guard against very old
            # CSV files (pre-dating this column) where the field may be absent.
            def _parse_float(s):
                return float(s) if s else float("nan")
            lr_values.append(_parse_float(row.get("lr", "")))
            train_avg_asym.append(_parse_float(row.get("train_avg_mass_asym", "")))
            train_std_asym.append(_parse_float(row.get("train_std_mass_asym", "")))
            val_avg_asym.append(_parse_float(row.get("val_avg_mass_asym", "")))
            val_std_asym.append(_parse_float(row.get("val_std_mass_asym", "")))
            train_grp_acc.append(_parse_float(row.get("train_grp_acc", "")))
            val_grp_acc.append(_parse_float(row.get("val_grp_acc", "")))
            train_avg_mpt.append(_parse_float(row.get("train_avg_max_triplet_pt", "")))
            train_std_mpt.append(_parse_float(row.get("train_std_max_triplet_pt", "")))
            val_avg_mpt.append(_parse_float(row.get("val_avg_max_triplet_pt", "")))
            val_std_mpt.append(_parse_float(row.get("val_std_max_triplet_pt", "")))
            train_avg_dphi.append(_parse_float(row.get("train_avg_delta_phi", "")))
            train_std_dphi.append(_parse_float(row.get("train_std_delta_phi", "")))
            val_avg_dphi.append(_parse_float(row.get("val_avg_delta_phi", "")))
            val_std_dphi.append(_parse_float(row.get("val_std_delta_phi", "")))
            train_avg_dem.append(_parse_float(row.get("train_avg_democracy", "")))
            train_std_dem.append(_parse_float(row.get("train_std_democracy", "")))
            val_avg_dem.append(_parse_float(row.get("val_avg_democracy", "")))
            val_std_dem.append(_parse_float(row.get("val_std_democracy", "")))

    if not epochs:
        print("  Warning: empty training log; skipping plots.")
        return []

    # --- Output directory and file tag ---
    if tag is None:
        ts = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        commit = _get_git_commit_hash()
        tag = f"{ts}_{commit}"
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # --- Phase-transition x-positions (between last Phase-1 and first Phase-2 epoch) ---
    phase_transitions: list[float] = []
    if phase2_start_epoch is not None:
        # Passed explicitly: transition happens just before this epoch starts.
        phase_transitions.append(phase2_start_epoch - 0.5)
    else:
        # Infer from the phase column in case the caller didn't provide it.
        for i in range(1, len(phases)):
            if phases[i] != phases[i - 1]:
                phase_transitions.append(epochs[i - 1] + 0.5)

    def _add_phase_lines(ax):
        for idx, x in enumerate(phase_transitions):
            ax.axvline(
                x=x,
                color="gray",
                linestyle="--",
                linewidth=1.2,
                label="Phase 1 → 2" if idx == 0 else None,
            )

    saved_paths: list[Path] = []

    # --- Loss plot ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_loss, label="Train loss", color="steelblue")
    ax.plot(epochs, val_loss, label="Val loss", color="darkorange")
    _add_phase_lines(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title("Loss vs Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    loss_path = plots_dir / f"loss_{tag}.pdf"
    fig.savefig(loss_path)
    plt.close(fig)
    print(f"  -> Saved loss plot     : {loss_path}")
    saved_paths.append(loss_path)

    # --- Learning-rate plot ---
    lr_epochs = [e for e, v in zip(epochs, lr_values) if not math.isnan(v)]
    if lr_epochs:
        lr_vals = [v for v in lr_values if not math.isnan(v)]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(lr_epochs, lr_vals, color="mediumorchid", label="Learning rate")
        _add_phase_lines(ax)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Learning rate")
        ax.set_yscale("log")
        ax.set_title("Learning Rate vs Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")
        fig.tight_layout()
        lr_path = plots_dir / f"lr_{tag}.pdf"
        fig.savefig(lr_path)
        plt.close(fig)
        print(f"  -> Saved LR plot       : {lr_path}")
        saved_paths.append(lr_path)

    # --- Accuracy plot ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_acc, label="Train acc", color="steelblue")
    ax.plot(epochs, val_acc, label="Val acc", color="darkorange")
    _add_phase_lines(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Epoch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    acc_path = plots_dir / f"accuracy_{tag}.pdf"
    fig.savefig(acc_path)
    plt.close(fig)
    print(f"  -> Saved accuracy plot : {acc_path}")
    saved_paths.append(acc_path)

    # --- Mass asymmetry plot (only when data are available) ---
    import math as _math
    # Filter to rows where at least the val mean is a real number.
    asym_epochs = [e for e, v in zip(epochs, val_avg_asym) if not _math.isnan(v)]
    if asym_epochs:
        asym_train_avg = [v for v in train_avg_asym if not _math.isnan(v)]
        asym_train_std = [v for v in train_std_asym if not _math.isnan(v)]
        asym_val_avg   = [v for v in val_avg_asym   if not _math.isnan(v)]
        asym_val_std   = [v for v in val_std_asym   if not _math.isnan(v)]

        fig, ax = plt.subplots(figsize=(9, 5))

        # Train: line + ±1σ shaded band
        ax.plot(asym_epochs, asym_train_avg, label="Train mean asym", color="steelblue")
        ax.fill_between(
            asym_epochs,
            [m - s for m, s in zip(asym_train_avg, asym_train_std)],
            [m + s for m, s in zip(asym_train_avg, asym_train_std)],
            color="steelblue", alpha=0.2, label="Train ±1σ",
        )

        # Val: line + ±1σ shaded band
        ax.plot(asym_epochs, asym_val_avg, label="Val mean asym", color="darkorange")
        ax.fill_between(
            asym_epochs,
            [m - s for m, s in zip(asym_val_avg, asym_val_std)],
            [m + s for m, s in zip(asym_val_avg, asym_val_std)],
            color="darkorange", alpha=0.2, label="Val ±1σ",
        )

        _add_phase_lines(ax)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mass asymmetry of chosen interpretation")
        ax.set_title("Mass Asymmetry of Chosen Interpretation vs Epoch")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")
        fig.tight_layout()
        asym_path = plots_dir / f"mass_asym_{tag}.pdf"
        fig.savefig(asym_path)
        plt.close(fig)
        print(f"  -> Saved mass asym plot: {asym_path}")
        saved_paths.append(asym_path)

    # --- GRP score (grouping accuracy) plot – only for factored (ISR) models ---
    import math as _math2
    grp_epochs = [e for e, v in zip(epochs, val_grp_acc) if not _math2.isnan(v) and v > 0]
    if grp_epochs:
        grp_train = [v for e, v in zip(epochs, train_grp_acc)
                     if e in set(grp_epochs) and not _math2.isnan(v)]
        grp_val   = [v for e, v in zip(epochs, val_grp_acc)
                     if e in set(grp_epochs) and not _math2.isnan(v)]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(grp_epochs, grp_train, label="Train grp acc", color="steelblue")
        ax.plot(grp_epochs, grp_val, label="Val grp acc", color="darkorange")
        _add_phase_lines(ax)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Grouping accuracy")
        ax.set_title("GRP Score (Grouping Accuracy) vs Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        grp_path = plots_dir / f"grp_acc_{tag}.pdf"
        fig.savefig(grp_path)
        plt.close(fig)
        print(f"  -> Saved GRP acc plot  : {grp_path}")
        saved_paths.append(grp_path)

    # --- Max-triplet scalar-sum pT plot (only when data are available) ---
    mpt_epochs = [e for e, v in zip(epochs, val_avg_mpt) if not _math2.isnan(v)]
    if mpt_epochs:
        mpt_train_avg = [v for e, v in zip(epochs, train_avg_mpt) if e in set(mpt_epochs) and not _math2.isnan(v)]
        mpt_train_std = [v for e, v in zip(epochs, train_std_mpt) if e in set(mpt_epochs) and not _math2.isnan(v)]
        mpt_val_avg   = [v for e, v in zip(epochs, val_avg_mpt)   if e in set(mpt_epochs) and not _math2.isnan(v)]
        mpt_val_std   = [v for e, v in zip(epochs, val_std_mpt)   if e in set(mpt_epochs) and not _math2.isnan(v)]

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(mpt_epochs, mpt_train_avg, label="Train mean max-triplet pT", color="steelblue")
        ax.fill_between(
            mpt_epochs,
            [m - s for m, s in zip(mpt_train_avg, mpt_train_std)],
            [m + s for m, s in zip(mpt_train_avg, mpt_train_std)],
            color="steelblue", alpha=0.2, label="Train ±1σ",
        )

        ax.plot(mpt_epochs, mpt_val_avg, label="Val mean max-triplet pT", color="darkorange")
        ax.fill_between(
            mpt_epochs,
            [m - s for m, s in zip(mpt_val_avg, mpt_val_std)],
            [m + s for m, s in zip(mpt_val_avg, mpt_val_std)],
            color="darkorange", alpha=0.2, label="Val ±1σ",
        )

        _add_phase_lines(ax)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Max-triplet scalar sum pT of chosen interpretation")
        ax.set_title("Max-Triplet Scalar Sum pT of Chosen Interpretation vs Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        mpt_path = plots_dir / f"max_triplet_pt_{tag}.pdf"
        fig.savefig(mpt_path)
        plt.close(fig)
        print(f"  -> Saved max-triplet-pT plot: {mpt_path}")
        saved_paths.append(mpt_path)

    # --- Δφ between parent candidates plot ---
    dphi_epochs = [e for e, v in zip(epochs, val_avg_dphi) if not _math2.isnan(v)]
    if dphi_epochs:
        dphi_train_avg = [v for e, v in zip(epochs, train_avg_dphi) if e in set(dphi_epochs) and not _math2.isnan(v)]
        dphi_train_std = [v for e, v in zip(epochs, train_std_dphi) if e in set(dphi_epochs) and not _math2.isnan(v)]
        dphi_val_avg   = [v for e, v in zip(epochs, val_avg_dphi)   if e in set(dphi_epochs) and not _math2.isnan(v)]
        dphi_val_std   = [v for e, v in zip(epochs, val_std_dphi)   if e in set(dphi_epochs) and not _math2.isnan(v)]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(dphi_epochs, dphi_train_avg, label="Train mean Δφ", color="steelblue")
        ax.fill_between(
            dphi_epochs,
            [m - s for m, s in zip(dphi_train_avg, dphi_train_std)],
            [m + s for m, s in zip(dphi_train_avg, dphi_train_std)],
            color="steelblue", alpha=0.2, label="Train ±1σ",
        )
        ax.plot(dphi_epochs, dphi_val_avg, label="Val mean Δφ", color="darkorange")
        ax.fill_between(
            dphi_epochs,
            [m - s for m, s in zip(dphi_val_avg, dphi_val_std)],
            [m + s for m, s in zip(dphi_val_avg, dphi_val_std)],
            color="darkorange", alpha=0.2, label="Val ±1σ",
        )
        _add_phase_lines(ax)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Δφ between parent candidates (rad)")
        ax.set_title("Δφ Between Parent Candidates of Chosen Interpretation vs Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        dphi_path = plots_dir / f"delta_phi_{tag}.pdf"
        fig.savefig(dphi_path)
        plt.close(fig)
        print(f"  -> Saved Δφ plot       : {dphi_path}")
        saved_paths.append(dphi_path)

    # --- pT democracy plot ---
    dem_epochs = [e for e, v in zip(epochs, val_avg_dem) if not _math2.isnan(v)]
    if dem_epochs:
        dem_train_avg = [v for e, v in zip(epochs, train_avg_dem) if e in set(dem_epochs) and not _math2.isnan(v)]
        dem_train_std = [v for e, v in zip(epochs, train_std_dem) if e in set(dem_epochs) and not _math2.isnan(v)]
        dem_val_avg   = [v for e, v in zip(epochs, val_avg_dem)   if e in set(dem_epochs) and not _math2.isnan(v)]
        dem_val_std   = [v for e, v in zip(epochs, val_std_dem)   if e in set(dem_epochs) and not _math2.isnan(v)]

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(dem_epochs, dem_train_avg, label="Train mean democracy", color="mediumseagreen")
        ax.fill_between(
            dem_epochs,
            [m - s for m, s in zip(dem_train_avg, dem_train_std)],
            [m + s for m, s in zip(dem_train_avg, dem_train_std)],
            color="mediumseagreen", alpha=0.2, label="Train ±1σ",
        )
        ax.plot(dem_epochs, dem_val_avg, label="Val mean democracy", color="darkorange")
        ax.fill_between(
            dem_epochs,
            [m - s for m, s in zip(dem_val_avg, dem_val_std)],
            [m + s for m, s in zip(dem_val_avg, dem_val_std)],
            color="darkorange", alpha=0.2, label="Val ±1σ",
        )
        _add_phase_lines(ax)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("pT democracy = avg(min pT / max pT) per triplet")
        ax.set_title("pT Democracy of Chosen Interpretation vs Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        dem_path = plots_dir / f"democracy_{tag}.pdf"
        fig.savefig(dem_path)
        plt.close(fig)
        print(f"  -> Saved democracy plot: {dem_path}")
        saved_paths.append(dem_path)

    return saved_paths


def _run_epoch(
    model, loader, ce_loss_fn, mse_loss_fn, lambda_adv, device, optimizer=None,
    tf_ratio=1.0, lambda_sym=0.0, lambda_qcd=0.0, lambda_isr=1.0, lambda_isr_direct=0.0,
    lambda_distill=0.0, distill_temperature=4.0,
    lambda_entropy_asym=0.0, lambda_entropy_mass=0.0,
    phase1_only=False,
    pt_smear_frac=0.0,
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
    all_pred_correct = []
    all_pred_mass_sum = []
    all_pred_max_triplet_pt = []
    all_pred_delta_phi = []
    all_pred_democracy = []
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

        # Dynamic pT smearing: scale each jet's 4-vector by a random per-jet
        # factor (massless approximation — η preserved means all components
        # scale proportionally with pT).  Applied per batch so each epoch
        # sees a fresh random realization.  Applied during both training and
        # validation so that evaluation conditions match training conditions.
        # smear_factor = 1 + σ·N(0,1) where σ = pt_smear_frac (std deviation).
        if pt_smear_frac > 0:
            four_mom = four_mom.clone()
            batch_size = four_mom.shape[0]
            num_jets = four_mom.shape[1]
            smear = (
                1.0 + pt_smear_frac * torch.randn(batch_size, num_jets, device=device)
            ).clamp(0.5, 1.5).unsqueeze(-1)  # (batch, jets, 1)
            four_mom = four_mom * smear
            # Re-normalize by the new event HT (= sum of new per-jet pT magnitudes)
            # so that HT-normalized scale invariance is preserved post-smearing.
            new_ht = torch.sqrt(
                four_mom[:, :, 1] ** 2 + four_mom[:, :, 2] ** 2
            ).sum(dim=1, keepdim=True).clamp(min=1e-6).unsqueeze(-1)  # (batch, 1, 1)
            four_mom = four_mom / new_ht

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

                # Direct ISR supervision from flat logits: for each ISR candidate j, take
                # the max grouping score assuming jet j is ISR.  This marginalises out the
                # grouping choice and gives a per-jet ISR score derived from the final flat
                # logits, providing a gradient path directly through the combined logits
                # rather than only through the isr_head auxiliary branch.
                if lambda_isr_direct > 0:
                    f2f_flat = model.factored_to_flat.reshape(-1)  # (num_jets * num_groupings,)
                    logits_fac = logits[:, f2f_flat].reshape(
                        labels.shape[0], model.num_jets, model.num_groupings
                    )
                    isr_logits_direct = logits_fac.max(dim=2).values   # (batch, num_jets)
                    loss_isr_direct = ce_loss_fn(isr_logits_direct, isr_labels)
                    loss_ce = loss_ce + lambda_isr_direct * loss_isr_direct
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

            # Entropy-weighted physics prior losses.
            #
            # When the network is uncertain (high output entropy), it receives a
            # gradient that steers its assignment probabilities toward interpretations
            # that look like QCD multijet background: high mass asymmetry (|m1-m2|
            # large relative to m1+m2) and low total mass (m1+m2 small).  This is
            # complementary to lambda_qcd (which uses pT hierarchy as the QCD proxy)
            # because it directly uses the network's own uncertainty as the signal.
            #
            # Entropy is detached so it acts purely as a per-event weight, not as a
            # quantity being minimised.  Entropy is normalised by log(N_assignments)
            # so it lies in [0, 1] regardless of the number of candidate assignments.
            #
            # lambda_entropy_asym: coefficient for the asymmetry term.
            #   loss = -mean(norm_entropy * expected_mass_asym)
            #   Minimising this drives uncertain events toward high-asymmetry choices.
            # lambda_entropy_mass: coefficient for the mass-sum term.
            #   loss = mean(norm_entropy * expected_mass_sum)
            #   Minimising this drives uncertain events toward low-mass choices.
            if (lambda_entropy_asym > 0 or lambda_entropy_mass > 0) and "mass_asym_flat" in output:
                probs_ent = logits.softmax(dim=-1)
                # Shannon entropy, normalised to [0, 1].
                # Clamp probabilities away from 0 to avoid log(0).
                entropy = -(probs_ent * (probs_ent.clamp(min=1e-10)).log()).sum(dim=-1)  # (batch,)
                max_entropy = math.log(logits.shape[-1])
                norm_entropy = (entropy / max_entropy).detach()                          # (batch,)

                if lambda_entropy_asym > 0:
                    mass_asym_ent = output["mass_asym_flat"].detach()                    # (B, N)
                    expected_asym_ent = (probs_ent * mass_asym_ent).sum(dim=-1)         # (batch,)
                    loss_entropy_asym = -(norm_entropy * expected_asym_ent).mean()
                    loss_ce = loss_ce + lambda_entropy_asym * loss_entropy_asym

                if lambda_entropy_mass > 0 and "mass_sum_flat" in output:
                    mass_sum_ent = output["mass_sum_flat"].detach()                      # (B, N)
                    expected_mass_sum = (probs_ent * mass_sum_ent).sum(dim=-1)          # (batch,)
                    loss_entropy_mass_term = (norm_entropy * expected_mass_sum).mean()
                    loss_ce = loss_ce + lambda_entropy_mass * loss_entropy_mass_term

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
            all_pred_correct.append((preds == labels).cpu())  # aligned with pred_asym

        if "mass_sum_flat" in output:
            mass_sum_flat = output["mass_sum_flat"].detach()  # (batch, num_assignments)
            pred_mass_sum = mass_sum_flat.gather(1, preds.unsqueeze(1)).squeeze(1)  # (batch,)
            all_pred_mass_sum.append(pred_mass_sum.cpu())

        # Max-triplet scalar-sum pT: look up the two triplets for each event's
        # predicted assignment and take the larger of the two per-triplet pT sums.
        # Also compute Δφ between the two parent 4-vector sums and average pT democracy.
        if factored and hasattr(model, "f_group1"):
            f2f = model.flat_to_factored              # (num_assignments, 2)
            pred_isr_i = f2f[preds, 0]                # (batch,)
            pred_grp_i = f2f[preds, 1]                # (batch,)
            g1_jets = model.f_group1[pred_isr_i, pred_grp_i]  # (batch, 3)
            g2_jets = model.f_group2[pred_isr_i, pred_grp_i]  # (batch, 3)
        elif not factored and hasattr(model, "group1_indices"):
            g1_jets = model.group1_indices[preds]     # (batch, 3)
            g2_jets = model.group2_indices[preds]     # (batch, 3)
        else:
            g1_jets = None
        if g1_jets is not None:
            px_b = four_mom[:, :, 1]
            py_b = four_mom[:, :, 2]
            pt_b = torch.sqrt(px_b**2 + py_b**2)     # (batch, num_jets)
            pt_g1 = pt_b.gather(1, g1_jets).sum(dim=1)   # (batch,)
            pt_g2 = pt_b.gather(1, g2_jets).sum(dim=1)   # (batch,)
            max_triplet_pt = torch.maximum(pt_g1, pt_g2)  # (batch,)
            all_pred_max_triplet_pt.append(max_triplet_pt.detach().cpu())

            # Δφ: azimuthal angle between the two parent 4-vector sums, in [0, π].
            sum_px_g1 = px_b.gather(1, g1_jets).sum(dim=1)  # (batch,)
            sum_py_g1 = py_b.gather(1, g1_jets).sum(dim=1)
            sum_px_g2 = px_b.gather(1, g2_jets).sum(dim=1)
            sum_py_g2 = py_b.gather(1, g2_jets).sum(dim=1)
            phi1 = torch.atan2(sum_py_g1, sum_px_g1)  # (batch,)
            phi2 = torch.atan2(sum_py_g2, sum_px_g2)
            dphi = (phi1 - phi2).abs()
            # Fold into [0, π]: if dphi > π, use 2π − dphi.
            dphi = torch.where(dphi > torch.pi, 2.0 * torch.pi - dphi, dphi)
            all_pred_delta_phi.append(dphi.detach().cpu())

            # pT democracy: min(pT)/max(pT) per triplet, averaged across the two.
            pt_jets_g1 = pt_b.gather(1, g1_jets)       # (batch, 3)
            pt_jets_g2 = pt_b.gather(1, g2_jets)       # (batch, 3)
            dem_g1 = pt_jets_g1.min(dim=1).values / (pt_jets_g1.max(dim=1).values.clamp(min=1e-8))
            dem_g2 = pt_jets_g2.min(dim=1).values / (pt_jets_g2.max(dim=1).values.clamp(min=1e-8))
            democracy = (dem_g1 + dem_g2) / 2.0        # (batch,)
            all_pred_democracy.append(democracy.detach().cpu())

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
        pred_asym_cat = torch.cat(all_pred_asym)
        result["avg_mass_asym"] = total_mass_asym / total_mass_asym_samples
        result["std_mass_asym"] = pred_asym_cat.std().item()
        result["pred_asym_values"] = pred_asym_cat.numpy()  # full per-event array
        result["pred_correct_values"] = torch.cat(all_pred_correct).numpy()  # bool per-event
    if all_pred_mass_sum:
        result["pred_mass_sum_values"] = torch.cat(all_pred_mass_sum).numpy()  # full per-event array
    if all_pred_max_triplet_pt:
        mpt_cat = torch.cat(all_pred_max_triplet_pt)
        result["pred_max_triplet_pt_values"] = mpt_cat.numpy()
        result["avg_max_triplet_pt"] = mpt_cat.mean().item()
        result["std_max_triplet_pt"] = mpt_cat.std().item()
    if all_pred_delta_phi:
        dphi_cat = torch.cat(all_pred_delta_phi)
        result["pred_delta_phi_values"] = dphi_cat.numpy()
        result["avg_delta_phi"] = dphi_cat.mean().item()
        result["std_delta_phi"] = dphi_cat.std().item()
    if all_pred_democracy:
        dem_cat = torch.cat(all_pred_democracy)
        result["pred_democracy_values"] = dem_cat.numpy()
        result["avg_democracy"] = dem_cat.mean().item()
        result["std_democracy"] = dem_cat.std().item()
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
