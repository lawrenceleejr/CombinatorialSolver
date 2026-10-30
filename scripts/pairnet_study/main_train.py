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
from torch.utils.data import ConcatDataset, DataLoader, random_split

from .dataset import JetAssignmentDataset
from .export_onnx import export_classical_solver, export_ml_model
from .model import JetAssignmentTransformer
from .utils import get_config, get_device


# Consistent colour palette shared by all training animations and trend plots.
# Uses the Okabe–Ito palette — a mature, colourblind-safe scheme widely adopted
# for scientific figures (https://jfly.uni-koeln.de/color/).
#   signal_correct/signal_wrong : the two parts of the signal stack
#   qcd                         : QCD background (filled when alone, unfilled
#                                 outline when overlaid on the signal stack)
#   mean                        : per-epoch mean reference line
_HIST_COLORS = {
    "signal_correct": "#009E73",  # Okabe–Ito bluish green — correct interpretation
    "signal_wrong":   "#CFCFCF",  # light grey — combinatorial bkg (wrong interpretation)
    "qcd":            "#D55E00",  # Okabe–Ito vermillion — QCD background
    "mean":           "#2A2A2A",  # near-black grey — mean reference line
}


def _rgba(color, alpha: float):
    """Translucent face colour for filled areas."""
    from matplotlib.colors import to_rgba
    return to_rgba(color, alpha)


def _edge(color, factor: float = 0.55):
    """A darker shade of *color* for a crisp thin outline around a filled area."""
    from matplotlib.colors import to_rgba
    r, g, b = to_rgba(color)[:3]
    return (r * factor, g * factor, b * factor)


def _style_axis(ax, grid_axis: str = "both") -> None:
    """Apply a clean, Tufte-inspired style: drop the top/right spines, lighten the
    remaining spines and ticks, and lay faint gridlines behind the data so the ink
    serves the data, not the frame.  Legends are drawn frameless by the callers."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#777777")
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors="#555555", length=3, width=0.8)
    ax.grid(True, axis=grid_axis, color="#E9E9E9", linewidth=0.6)
    ax.set_axisbelow(True)


_FONTS_REGISTERED = False


def _ensure_serif_fonts() -> None:
    """Make the EB Garamond serif available without a system install or network.

    If no Garamond is already known to matplotlib, the OFL-licensed copies vendored
    in ``assets/fonts/`` are registered with the font manager for this process (so
    plots use Garamond immediately) and also copied into the user font directory so
    they persist for future runs and other tools.  Best-effort and idempotent;
    never raises — if anything fails, plotting simply falls back to Times/DejaVu.
    """
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return
    _FONTS_REGISTERED = True
    try:
        import matplotlib.font_manager as fm
        have = {f.name for f in fm.fontManager.ttflist}
        if {"EB Garamond", "Garamond"} & have:
            return  # a Garamond is already installed
        font_dir = Path(__file__).resolve().parent.parent / "assets" / "fonts"
        bundled = sorted(font_dir.rglob("*.otf")) + sorted(font_dir.rglob("*.ttf"))
        if not bundled:
            return
        for p in bundled:
            try:
                fm.fontManager.addfont(str(p))
            except Exception:
                pass
        # Persist to the user font directory so the font "installs" for next time.
        try:
            import shutil
            user_dir = Path.home() / ".local" / "share" / "fonts" / "ebgaramond"
            user_dir.mkdir(parents=True, exist_ok=True)
            for p in bundled:
                dest = user_dir / p.name
                if not dest.exists():
                    shutil.copy2(p, dest)
        except Exception:
            pass
        if "EB Garamond" in {f.name for f in fm.fontManager.ttflist}:
            print("  Registered vendored EB Garamond serif for plots.")
    except Exception:
        pass


def _init_plot_style(plt) -> None:
    """Tufte-flavoured global style: a serif font family, larger base/legend text,
    serif math, and extra title padding so titles sit a little higher.  Idempotent;
    called at the top of every plotting helper before any text is drawn."""
    _ensure_serif_fonts()
    plt.rcParams.update({
        "font.family": "serif",
        # Prefer a refined serif; fall back gracefully to whatever is installed.
        "font.serif": ["EB Garamond", "Garamond", "Adobe Garamond Pro",
                       "Times New Roman", "Times", "Nimbus Roman No9 L",
                       "Liberation Serif", "DejaVu Serif"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "mathtext.fontset": "cm",
        "axes.titlepad": 16,
        # Garamond/Times lack the Unicode minus (U+2212); use ASCII hyphen on ticks.
        "axes.unicode_minus": False,
    })


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


def train(config_path: str | None = None, data_path: str | None = None,
          qcd_data_path: str | None = None):
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
        data_path = dc.get("data_path", "data/*.h5")

    sig_dataset = JetAssignmentDataset(
        data_paths=data_path,
        num_jets=dc["num_jets"],
        normalize_by_ht=dc["normalize_by_ht"],
        pt_smear_frac=dc.get("pt_smear_frac", 0.0),
        use_mass_asymmetry_labels=dc.get("use_mass_asymmetry_labels", True),
        is_background=False,
    )
    print(f"Signal dataset size: {len(sig_dataset)} events")

    # Optional QCD/background sample.  Tagged is_background=True so its events
    # drive the background-rejection loss (push to low average mass OR high
    # asymmetry) instead of the supervised assignment loss.  The glob comes from
    # the --qcd-data CLI flag (preferred) or data.qcd_data_path; if unset or no
    # files match, training is signal-only and behaves exactly as before.
    qcd_path = qcd_data_path if qcd_data_path is not None else dc.get("qcd_data_path")
    qcd_dataset = None
    if qcd_path:
        try:
            qcd_dataset = JetAssignmentDataset(
                data_paths=qcd_path,
                num_jets=dc["num_jets"],
                normalize_by_ht=dc["normalize_by_ht"],
                pt_smear_frac=dc.get("pt_smear_frac", 0.0),
                use_mass_asymmetry_labels=dc.get("use_mass_asymmetry_labels", True),
                is_background=True,
            )
            print(f"QCD background dataset size: {len(qcd_dataset)} events")
        except (FileNotFoundError, OSError, ValueError) as exc:
            print(f"QCD background sample not loaded ({qcd_path}): {exc}")
            qcd_dataset = None

    qcd_present = qcd_dataset is not None and len(qcd_dataset) > 0
    if qcd_present:
        dataset = ConcatDataset([sig_dataset, qcd_dataset])
        n_bkg = len(qcd_dataset)
        print(
            f"Combined dataset size: {len(dataset)} events "
            f"({len(sig_dataset)} signal + {n_bkg} background, "
            f"{100.0 * n_bkg / max(len(dataset), 1):.1f}% background)"
        )
    else:
        dataset = sig_dataset
        print(f"Dataset size: {len(dataset)} events (signal only)")

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

    # Check if adversarial training is useful (signal mass spread only;
    # background events carry zeroed parent_mass and are excluded).
    mass_std = sig_dataset.parent_masses.std().item()
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
    val_max_boost_history: list = []  # per-epoch list (val larger triplet Lorentz boost γ=E/m)
    val_avg_boost_history: list = []  # per-epoch list (val average triplet Lorentz boost γ=E/m)
    val_dalitz_history: list = []     # per-epoch list of (epoch, phase, x, y, is_bkg) for Dalitz

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
    # Background-rejection loss (requires a QCD sample loaded with is_background=True).
    # For each background event, supervise the assignment toward the one that
    # maximises bg = mass_asym - beta_bg * mass_sum — the most QCD-like (low average
    # mass OR high asymmetry) interpretation.  This drives the QCD average-mass
    # distribution to fall at least as steeply as the classical mass-asymmetry
    # solver, while leaving signal untouched (the term is masked to background
    # events only).  beta_bg -> inf is pure average-mass minimisation (steepest
    # background); beta_bg -> 0 is pure asymmetry maximisation.  Ramps up from
    # Phase 2 start like lambda_sym/qcd.
    lambda_bg_max = tc.get("lambda_bg", 0.0)
    lambda_bg_rampup = tc.get("lambda_bg_rampup", 0)
    beta_bg = tc.get("beta_bg", 0.5)
    bg_soft_weight = tc.get("bg_soft_weight", 2.0)
    bg_asym_cut = tc.get("bg_asym_cut", 0.0)
    # Handing the trainer a QCD sample turns on the background-rejection objective
    # automatically: if lambda_bg was left at its default 0, enable it so the QCD
    # events actually penalise high-average-mass interpretations.  Set
    # training.lambda_bg explicitly in the config to override (e.g. back to 0).
    if qcd_present and lambda_bg_max <= 0:
        lambda_bg_max = 2.0
        print("QCD sample provided -> auto-enabling background-rejection loss "
              "(lambda_bg=2.0; set training.lambda_bg to override).")
    if lambda_bg_max > 0:
        print(
            f"Background-rejection loss: lambda_bg={lambda_bg_max} "
            f"(rampup={lambda_bg_rampup}), beta_bg={beta_bg}, "
            f"bg_soft_weight={bg_soft_weight}, bg_asym_cut={bg_asym_cut}"
        )
        if qcd_present and (
            lambda_qcd_max > 0 or lambda_entropy_asym_max > 0 or lambda_entropy_mass_max > 0
        ):
            print("  Note: lambda_qcd / lambda_entropy_* are signal-side QCD proxies; "
                  "with a real QCD sample they are redundant — consider setting them to 0.")
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

    try:
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
                lambda_isr_direct = 0.0
                lambda_entropy_asym = 0.0
                lambda_entropy_mass = 0.0
                lambda_bg = 0.0
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

                if lambda_bg_rampup > 0:
                    lambda_bg = lambda_bg_max * min(1.0, phase2_epoch / lambda_bg_rampup)
                else:
                    lambda_bg = lambda_bg_max

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
                lambda_bg=lambda_bg, beta_bg=beta_bg,
                bg_soft_weight=bg_soft_weight, bg_asym_cut=bg_asym_cut,
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
                    lambda_bg=0.0, beta_bg=beta_bg,
                    bg_soft_weight=bg_soft_weight, bg_asym_cut=bg_asym_cut,
                    pt_smear_frac=dc.get("pt_smear_frac", 0.0),
                )

            # Accumulate per-event validation mass-asymmetry distribution for GIF
            if "pred_asym_values" in val_metrics:
                val_asym_history.append((
                    epoch + 1, training_phase,
                    val_metrics["pred_asym_values"],
                    val_metrics.get("pred_correct_values"),       # bool array or None
                    val_metrics.get("pred_is_bkg_values"),        # bool array or None
                    val_metrics.get("pred_asym_achievable_values"),  # best achievable, or None
                ))

            # Accumulate per-event validation mass-sum distribution for GIF
            if "pred_mass_sum_values" in val_metrics:
                val_mass_sum_history.append((
                    epoch + 1, training_phase,
                    val_metrics["pred_mass_sum_values"],
                    val_metrics.get("pred_correct_values"),           # bool array or None
                    val_metrics.get("pred_is_bkg_values"),            # bool array or None
                    val_metrics.get("pred_mass_sum_achievable_values"),  # best achievable, or None
                ))

            # Accumulate per-event validation max-triplet scalar-pT distribution for GIF
            if "pred_max_triplet_pt_values" in val_metrics:
                val_max_triplet_pt_history.append((
                    epoch + 1, training_phase,
                    val_metrics["pred_max_triplet_pt_values"],
                    val_metrics.get("pred_correct_values"),  # bool array or None
                    val_metrics.get("pred_is_bkg_values"),   # bool array or None
                ))

            # Accumulate per-event validation ΔΦ distribution for GIF
            if "pred_delta_phi_values" in val_metrics:
                val_delta_phi_history.append((
                    epoch + 1, training_phase,
                    val_metrics["pred_delta_phi_values"],
                    val_metrics.get("pred_correct_values"),  # bool array or None
                    val_metrics.get("pred_is_bkg_values"),   # bool array or None
                ))

            # Accumulate per-event validation pT-democracy distribution for GIF
            if "pred_democracy_values" in val_metrics:
                val_democracy_history.append((
                    epoch + 1, training_phase,
                    val_metrics["pred_democracy_values"],
                    val_metrics.get("pred_correct_values"),  # bool array or None
                    val_metrics.get("pred_is_bkg_values"),   # bool array or None
                ))

            # Accumulate per-event validation triplet Lorentz-boost distributions for GIF
            if "pred_max_boost_values" in val_metrics:
                val_max_boost_history.append((
                    epoch + 1, training_phase,
                    val_metrics["pred_max_boost_values"],
                    val_metrics.get("pred_correct_values"),
                    val_metrics.get("pred_is_bkg_values"),
                ))
            if "pred_avg_boost_values" in val_metrics:
                val_avg_boost_history.append((
                    epoch + 1, training_phase,
                    val_metrics["pred_avg_boost_values"],
                    val_metrics.get("pred_correct_values"),
                    val_metrics.get("pred_is_bkg_values"),
                ))
            # Per-epoch validation Dalitz coordinates (for the animated 2-D Dalitz
            # plot).  Cap the stored points per epoch so the GIF stays light.
            if "pred_dalitz_x_values" in val_metrics:
                _dx = val_metrics["pred_dalitz_x_values"]
                _dy = val_metrics["pred_dalitz_y_values"]
                _db = val_metrics.get("pred_is_bkg_values")
                _cap = 4000
                if _dx.shape[0] > _cap:
                    _dx, _dy = _dx[:_cap], _dy[:_cap]
                    _db = _db[:_cap] if _db is not None else None
                val_dalitz_history.append((epoch + 1, training_phase, _dx, _dy, _db))

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
            # Distribution animations always show signal AND QCD together.
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
            # One combined trend overlaying signal and QCD means vs epoch
            # (mass asymmetry and average candidate mass).
            if val_asym_history or val_mass_sum_history:
                _make_trend_plot(
                    _trend_panels(val_asym_history, val_mass_sum_history),
                    phase2_start_epoch=phase2_start_epoch,
                    out_path=Path("plots") / "trends_latest.pdf",
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
            if val_max_boost_history:
                _make_max_boost_gif(
                    val_max_boost_history,
                    phase2_start_epoch=phase2_start_epoch,
                    gif_path=Path("plots") / "max_boost_anim_latest.gif",
                )
            if val_avg_boost_history:
                _make_avg_boost_gif(
                    val_avg_boost_history,
                    phase2_start_epoch=phase2_start_epoch,
                    gif_path=Path("plots") / "avg_boost_anim_latest.gif",
                )
            if val_dalitz_history:
                _make_dalitz_gif(
                    val_dalitz_history,
                    phase2_start_epoch=phase2_start_epoch,
                    gif_path=Path("plots") / "dalitz_anim_latest.gif",
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

    # One combined trend overlaying signal and QCD means vs epoch (mass asymmetry
    # and average candidate mass), with the QCD "best achievable" ceiling.
    trend = _make_trend_plot(
        _trend_panels(val_asym_history, val_mass_sum_history),
        phase2_start_epoch=phase2_start_epoch,
    )
    if trend is not None:
        plot_paths.append(trend)

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

    # Triplet Lorentz-boost animations and the 2-D Dalitz plot.
    if val_max_boost_history:
        mb_gif = _make_max_boost_gif(val_max_boost_history, phase2_start_epoch=phase2_start_epoch)
        if mb_gif is not None:
            plot_paths.append(mb_gif)
    if val_avg_boost_history:
        ab_gif = _make_avg_boost_gif(val_avg_boost_history, phase2_start_epoch=phase2_start_epoch)
        if ab_gif is not None:
            plot_paths.append(ab_gif)
    if val_dalitz_history:
        dalitz = _make_dalitz_gif(val_dalitz_history, phase2_start_epoch=phase2_start_epoch)
        if dalitz is not None:
            plot_paths.append(dalitz)

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
    """Combined mass-asymmetry GIF: signal correct/wrong stack + normalized QCD outline."""
    import numpy as np
    return _make_distribution_gif(
        val_asym_history,
        value_fn=lambda v: np.log10(np.clip(v, 1e-4, 1.0)),
        xlabel=r"$\log_{10}$(mass asymmetry of chosen interpretation)",
        title_prefix="Mass asymmetry",
        short_name="mass_asym_anim",
        subset="combined",
        x_range=(-4.0, 0.0),
        gif_path=gif_path,
        phase2_start_epoch=phase2_start_epoch,
    )


def _make_mass_sum_gif(
    val_mass_sum_history: list,
    phase2_start_epoch: int | None = None,
    gif_path: str | Path | None = None,
) -> "Path | None":
    """Combined average-mass GIF: signal correct/wrong stack + normalized QCD outline."""
    return _make_distribution_gif(
        val_mass_sum_history,
        value_fn=lambda v: v / 2.0,
        xlabel=r"Average candidate mass $(m_1{+}m_2)/2$ of chosen interpretation",
        title_prefix="Average candidate mass",
        short_name="mass_sum_anim",
        subset="combined",
        x_range=None,
        gif_path=gif_path,
        phase2_start_epoch=phase2_start_epoch,
    )


def _make_distribution_gif(
    history: list,
    value_fn,
    xlabel: str,
    title_prefix: str,
    short_name: str,
    *,
    subset: str = "combined",
    gif_path: str | Path | None = None,
    phase2_start_epoch: int | None = None,
    x_range: tuple[float, float] | None = None,
    n_bins: int = 50,
    ylabel: str | None = None,
) -> "Path | None":
    """Animated per-epoch histogram of a per-event quantity, with consistent colours.

    History entries are ``(epoch, phase, values, correct_mask, is_bkg[, achievable])``.
    *subset* selects what each frame shows:

      - ``"signal"``  : signal events only, a filled stack of correct (green) on
                        the bottom and wrong (orange) on top.
      - ``"qcd"``     : QCD/background events only, a single filled histogram (red).
      - ``"combined"``: the signal correct/wrong stack (filled) PLUS the QCD
                        distribution drawn as a separate **unfilled step outline**,
                        area-normalised to the signal stack so the shapes compare.

    *value_fn* maps the raw per-event values to the plotted quantity.  Returns
    ``None`` when the chosen subset has no events in any frame (e.g. "qcd"/"combined"
    on a signal-only run produce no QCD content; "qcd" returns None entirely).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        _init_plot_style(plt)
    except ImportError:
        print(f"  Warning: matplotlib not available; skipping {title_prefix} GIF.")
        return None
    if not history:
        return None

    import numpy as np
    C = _HIST_COLORS

    def _parts(entry):
        vals = value_fn(np.asarray(entry[2]))
        n = len(vals)
        correct = entry[3] if len(entry) >= 4 else None
        correct = np.asarray(correct, dtype=bool) if correct is not None else None
        is_bkg = entry[4] if len(entry) >= 5 else None
        is_bkg = np.asarray(is_bkg, dtype=bool) if is_bkg is not None else np.zeros(n, dtype=bool)
        return entry[0], entry[1], vals, correct, is_bkg

    parsed = [_parts(e) for e in history]

    def _sel(vals, is_bkg):
        if subset == "qcd":
            return vals[is_bkg]
        if subset == "signal":
            return vals[~is_bkg]
        return vals  # combined

    usable = [p for p in parsed if len(_sel(p[2], p[4])) > 0]
    if not usable:
        return None

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    if gif_path is None:
        ts = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        commit = _get_git_commit_hash()
        gif_path = plots_dir / f"{short_name}_{ts}_{commit}.gif"
    gif_path = Path(gif_path)

    all_concat = np.concatenate([_sel(p[2], p[4]) for p in usable])
    if x_range is not None:
        x_min, x_max = x_range
    else:
        x_min = float(np.percentile(all_concat, 1))
        x_max = float(np.percentile(all_concat, 99))
        if x_min >= x_max:
            x_min, x_max = float(all_concat.min()), float(all_concat.max()) + 1e-6
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_width = (x_max - x_min) / n_bins

    def _frame_hist(vals, correct, is_bkg):
        sig = ~is_bkg
        sig_corr = np.zeros(n_bins)
        sig_wrong = np.zeros(n_bins)
        qcd_out = np.zeros(n_bins)
        if subset in ("signal", "combined"):
            if correct is not None:
                sig_corr = np.histogram(vals[sig & correct], bins=bin_edges)[0]
                sig_wrong = np.histogram(vals[sig & ~correct], bins=bin_edges)[0]
            else:
                sig_corr = np.histogram(vals[sig], bins=bin_edges)[0]
        qc_raw = np.histogram(vals[is_bkg], bins=bin_edges)[0]
        if subset == "qcd":
            qcd_out = qc_raw.astype(float)
        elif subset == "combined":
            sig_total = float(sig_corr.sum() + sig_wrong.sum())
            qc_total = float(qc_raw.sum())
            qcd_out = qc_raw * (sig_total / qc_total) if (qc_total > 0 and sig_total > 0) else qc_raw.astype(float)
        return sig_corr, sig_wrong, qcd_out

    y_max = 1.0
    for _, _, vals, correct, is_bkg in usable:
        sc, sw, qo = _frame_hist(vals, correct, is_bkg)
        top = max(float((sc + sw).max()), float(qo.max()) if qo.size else 0.0)
        y_max = max(y_max, top)
    y_max *= 1.15

    fig, ax = plt.subplots(figsize=(8, 5))

    def _draw_frame(frame_idx):
        epoch, phase, vals, correct, is_bkg = usable[frame_idx]
        ax.cla()
        sc, sw, qo = _frame_hist(vals, correct, is_bkg)
        # Filled areas have NO per-bar edge; the histogram envelope is drawn once
        # as an unfilled step outline on top, for a clean ROOT-like look.
        if subset in ("signal", "combined"):
            ax.bar(centers, sc, width=bar_width, align="center", linewidth=0,
                   facecolor=_rgba(C["signal_correct"], 0.82), label="Signal correct")
            ax.bar(centers, sw, width=bar_width, align="center", bottom=sc, linewidth=0,
                   facecolor=_rgba(C["signal_wrong"], 0.82), label="Signal wrong")
            ax.stairs(sc, bin_edges, color=_edge(C["signal_correct"]), linewidth=1.1)
            ax.stairs(sc + sw, bin_edges, color=_edge(C["signal_wrong"]), linewidth=1.1)
        if subset == "qcd":
            ax.bar(centers, qo, width=bar_width, align="center", linewidth=0,
                   facecolor=_rgba(C["qcd"], 0.82), label="QCD background")
            ax.stairs(qo, bin_edges, color=_edge(C["qcd"]), linewidth=1.1)
        if subset == "combined" and qo.sum() > 0:
            ax.stairs(qo, bin_edges, color=C["qcd"], linewidth=1.6,
                      label="QCD (area-normalized)")
        # Mean reference line(s) — thin, understated.
        sig_vals = vals[~is_bkg]
        qcd_vals = vals[is_bkg]
        if subset in ("signal", "combined") and len(sig_vals):
            m = float(sig_vals.mean())
            ax.axvline(m, color=C["mean"], linewidth=1.3,
                       label=f"{'Signal ' if subset == 'combined' else ''}mean = {m:.3f}")
        if subset == "qcd" and len(qcd_vals):
            m = float(qcd_vals.mean())
            ax.axvline(m, color=C["mean"], linewidth=1.3, label=f"Mean = {m:.3f}")
        if subset == "combined" and len(qcd_vals):
            m = float(qcd_vals.mean())
            ax.axvline(m, color=C["qcd"], linewidth=1.3, linestyle="--",
                       label=f"QCD mean = {m:.3f}")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel or "Validation events")
        phase_label = ""
        if phase2_start_epoch is not None:
            phase_label = " [Phase 2]" if epoch >= phase2_start_epoch else " [Phase 1]"
        elif phase == 2:
            phase_label = " [Phase 2]"
        ax.set_title(f"{title_prefix} — Epoch {epoch}{phase_label}", loc="left", fontsize=11)
        ax.legend(loc="upper right", fontsize=11, frameon=False)
        _style_axis(ax, grid_axis="y")

    anim = animation.FuncAnimation(
        fig, _draw_frame, frames=len(usable), interval=200, repeat=False
    )
    try:
        anim.save(str(gif_path), writer="pillow", fps=5)
        print(f"  -> Saved {title_prefix} GIF: {gif_path}")
        return gif_path
    except Exception as exc:
        print(f"  Warning: could not save {title_prefix} GIF ({exc}).")
        return None
    finally:
        plt.close(fig)


def _trend_panels(asym_hist, mass_hist):
    """Build the (history, transform, ylabel, title, show_achievable) panel list
    for the combined trend plot: mass asymmetry and average candidate mass."""
    return [
        (asym_hist, lambda v: v,
         r"Mass asymmetry $|m_1{-}m_2|/(m_1{+}m_2)$", "Mass asymmetry vs epoch", True),
        (mass_hist, lambda v: v / 2.0,
         r"Average candidate mass $(m_1{+}m_2)/2$", "Average candidate mass vs epoch", True),
    ]


def _make_trend_plot(
    panels: list,
    phase2_start_epoch: int | None = None,
    out_path: str | Path | None = None,
) -> "Path | None":
    """Grid of mean(±1σ)-vs-epoch trends, each panel overlaying the signal and
    QCD-background means (with markers), laid out two per row.

    *panels* is a list of ``(history, transform, ylabel, title, show_achievable)``
    tuples.  ``show_achievable`` adds the dashed QCD "best achievable" ceiling
    (for the asymmetry / average-mass panels).  Returns ``None`` when there are
    no validation events.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _init_plot_style(plt)
    except ImportError:
        return None

    import numpy as np

    def _series(history, transform, subset):
        xs, means, stds, ach = [], [], [], []
        for entry in history:
            vals = np.asarray(entry[2])
            is_bkg = entry[4] if len(entry) >= 5 else None
            is_bkg = np.asarray(is_bkg, dtype=bool) if is_bkg is not None else np.zeros(len(vals), dtype=bool)
            sel = is_bkg if subset == "qcd" else ~is_bkg
            if not sel.any():
                continue
            v = transform(vals[sel])
            xs.append(entry[0])
            means.append(float(v.mean()))
            stds.append(float(v.std()))
            ach_arr = entry[5] if len(entry) >= 6 else None
            ach.append(float(transform(np.asarray(ach_arr)[sel]).mean())
                       if ach_arr is not None else float("nan"))
        return np.array(xs), np.array(means), np.array(stds), np.array(ach)

    subsets = [("signal", "Signal", _HIST_COLORS["signal_correct"]),
               ("qcd", "QCD background", _HIST_COLORS["qcd"])]

    n = len(panels)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.6 * nrows), squeeze=False)
    axes_flat = list(axes.ravel())
    any_data = False
    for ax, (history, transform, ylabel, title, show_ach) in zip(axes_flat, panels):
        for subset, label, color in subsets:
            e, m, s, a = _series(history, transform, subset)
            if not len(e):
                continue
            any_data = True
            ax.plot(e, m, label=f"{label} mean", marker="o", markersize=4,
                    markeredgecolor="white", markeredgewidth=0.6, linewidth=1.6, color=color)
            ax.fill_between(e, m - s, m + s, facecolor=_rgba(color, 0.15),
                            edgecolor=_rgba(color, 0.45), linewidth=0.6)
            # The "best achievable" ceiling is only meaningful for QCD (how far it
            # could be pushed out); the signal max-asymmetry ceiling is not useful.
            if show_ach and subset == "qcd" and np.isfinite(a).any():
                ax.plot(e, a, "--", marker="o", markersize=3, color=color, alpha=0.55,
                        label=f"{label} best achievable")
        if phase2_start_epoch is not None:
            ax.axvline(phase2_start_epoch, color="#777777", linestyle=":", linewidth=1.0,
                       label="Phase 2 start")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        _style_axis(ax, grid_axis="both")
        ax.legend(loc="best", frameon=False, fontsize=10)
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    if not any_data:
        plt.close(fig)
        return None
    fig.tight_layout()

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    if out_path is None:
        ts = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        commit = _get_git_commit_hash()
        out_path = plots_dir / f"trends_{ts}_{commit}.pdf"
    out_path = Path(out_path)
    try:
        fig.savefig(str(out_path))
        print(f"  -> Saved trend plot: {out_path}")
        return out_path
    except Exception as exc:
        print(f"  Warning: could not save trend plot ({exc}).")
        return None
    finally:
        plt.close(fig)


def _make_max_triplet_pt_gif(
    val_max_triplet_pt_history: list,
    phase2_start_epoch: int | None = None,
    gif_path: str | Path | None = None,
) -> "Path | None":
    """Combined max-triplet scalar-sum-pT animation: signal correct/wrong stack
    plus the area-normalised QCD outline (signal AND QCD shown together)."""
    return _make_distribution_gif(
        val_max_triplet_pt_history,
        value_fn=lambda v: v,
        xlabel="Max-triplet scalar sum pT of chosen interpretation",
        title_prefix="Max-triplet scalar-sum pT",
        short_name="max_triplet_pt_anim",
        subset="combined",
        x_range=None,
        gif_path=gif_path,
        phase2_start_epoch=phase2_start_epoch,
    )


def _make_delta_phi_gif(
    val_delta_phi_history: list,
    phase2_start_epoch: int | None = None,
    gif_path: str | Path | None = None,
) -> "Path | None":
    """Combined Δφ animation: signal correct/wrong stack + area-normalised QCD outline."""
    return _make_distribution_gif(
        val_delta_phi_history,
        value_fn=lambda v: v,
        xlabel=r"$\Delta\phi$ between parent candidates (rad)",
        title_prefix=r"$\Delta\phi$ between parent candidates",
        short_name="delta_phi_anim",
        subset="combined",
        x_range=(0.0, math.pi),
        gif_path=gif_path,
        phase2_start_epoch=phase2_start_epoch,
    )


def _make_democracy_gif(
    val_democracy_history: list,
    phase2_start_epoch: int | None = None,
    gif_path: str | Path | None = None,
) -> "Path | None":
    """Combined pT-democracy animation: signal correct/wrong stack + area-normalised QCD outline."""
    return _make_distribution_gif(
        val_democracy_history,
        value_fn=lambda v: v,
        xlabel="pT democracy = avg(min pT / max pT) per triplet",
        title_prefix="pT democracy",
        short_name="democracy_anim",
        subset="combined",
        x_range=(0.0, 1.0),
        gif_path=gif_path,
        phase2_start_epoch=phase2_start_epoch,
    )


def _make_max_boost_gif(
    val_max_boost_history: list,
    phase2_start_epoch: int | None = None,
    gif_path: str | Path | None = None,
) -> "Path | None":
    """Combined animation of the larger triplet Lorentz boost γ=E/m (signal + QCD)."""
    return _make_distribution_gif(
        val_max_boost_history,
        value_fn=lambda v: v,
        xlabel="Max triplet Lorentz boost  γ = E/m",
        title_prefix="Max triplet boost",
        short_name="max_boost_anim",
        subset="combined",
        x_range=None,
        gif_path=gif_path,
        phase2_start_epoch=phase2_start_epoch,
    )


def _make_avg_boost_gif(
    val_avg_boost_history: list,
    phase2_start_epoch: int | None = None,
    gif_path: str | Path | None = None,
) -> "Path | None":
    """Combined animation of the average triplet Lorentz boost γ=E/m (signal + QCD)."""
    return _make_distribution_gif(
        val_avg_boost_history,
        value_fn=lambda v: v,
        xlabel="Average triplet Lorentz boost  γ = E/m",
        title_prefix="Average triplet boost",
        short_name="avg_boost_anim",
        subset="combined",
        x_range=None,
        gif_path=gif_path,
        phase2_start_epoch=phase2_start_epoch,
    )


def _make_dalitz_gif(
    val_dalitz_history: list,
    phase2_start_epoch: int | None = None,
    gif_path: str | Path | None = None,
) -> "Path | None":
    """Animated two-panel Dalitz plot (Signal | QCD) over epochs.

    For each parent-candidate triplet the jets are pT-ordered and the normalised
    pairwise invariant-mass-squared are plotted: x = m²(lead,sub)/M²,
    y = m²(lead,third)/M².  Each event contributes its two triplets.  A genuine
    3-body decay fills the Dalitz interior, while combinatorial/QCD triplets
    cluster near the low-mass edges — so the two panels (and their evolution over
    epochs) reveal whether the network's chosen groupings have real 3-body
    structure.  Each panel's colour scale is fixed across frames for comparability.

    *val_dalitz_history* entries are ``(epoch, phase, x, y, is_bkg)`` with x, y of
    shape (N, 2) (the two triplets per event) and is_bkg of shape (N,).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        _init_plot_style(plt)
    except ImportError:
        return None
    if not val_dalitz_history:
        return None

    import numpy as np

    lim = 1.05
    nb = 44
    edges = np.linspace(0.0, lim, nb + 1)

    def _counts(entry):
        epoch, phase, x, y, bk = entry
        x = np.asarray(x, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
        if bk is not None:
            b = np.asarray(bk, dtype=bool).reshape(-1)
            bkg = np.repeat(b, 2) if b.size * 2 == x.size else np.zeros(x.size, dtype=bool)
        else:
            bkg = np.zeros(x.size, dtype=bool)
        hs, _, _ = np.histogram2d(x[~bkg], y[~bkg], bins=[edges, edges])
        hq, _, _ = np.histogram2d(x[bkg], y[bkg], bins=[edges, edges])
        return epoch, phase, hs.T, hq.T  # transpose -> rows = y for imshow

    frames = [_counts(e) for e in val_dalitz_history]
    vmax_s = max((float(f[2].max()) for f in frames if f[2].size), default=1.0) or 1.0
    vmax_q = max((float(f[3].max()) for f in frames if f[3].size), default=1.0) or 1.0

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    if gif_path is None:
        ts = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        commit = _get_git_commit_hash()
        gif_path = plots_dir / f"dalitz_anim_{ts}_{commit}.gif"
    gif_path = Path(gif_path)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5.4))

    def _draw_frame(i):
        epoch, phase, hs, hq = frames[i]
        for ax, h, cmap, label, vmax in (
            (a1, hs, "Greens", "Signal", vmax_s),
            (a2, hq, "OrRd", "QCD background", vmax_q),
        ):
            ax.cla()
            ax.imshow(h, origin="lower", extent=(0, lim, 0, lim), cmap=cmap,
                      vmin=0.0, vmax=vmax, aspect="auto", interpolation="nearest")
            phase_label = ""
            if phase2_start_epoch is not None:
                phase_label = " [Phase 2]" if epoch >= phase2_start_epoch else " [Phase 1]"
            elif phase == 2:
                phase_label = " [Phase 2]"
            ax.set_xlabel(r"$m^2(\mathrm{lead,sub})\,/\,M^2$")
            ax.set_ylabel(r"$m^2(\mathrm{lead,third})\,/\,M^2$")
            ax.set_title(f"{label} Dalitz — Epoch {epoch}{phase_label}", loc="left")
            _style_axis(ax, grid_axis="both")
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)

    anim = animation.FuncAnimation(fig, _draw_frame, frames=len(frames), interval=250, repeat=False)
    try:
        anim.save(str(gif_path), writer="pillow", fps=4)
        print(f"  -> Saved Dalitz GIF: {gif_path}")
        return gif_path
    except Exception as exc:
        print(f"  Warning: could not save Dalitz GIF ({exc}).")
        return None
    finally:
        plt.close(fig)


def _plot_training_curves(
    log_path: str | Path,
    phase2_start_epoch: int | None = None,
    tag: str | None = None,
) -> "list[Path]":
    """Generate loss, accuracy, and mass-asymmetry plots from the training log CSV.

    Creates three PDF files in a ``plots/`` directory:
      - ``loss_{tag}.pdf``         – train and validation loss vs epoch
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
        _init_plot_style(plt)
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
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            val_loss.append(float(row["val_loss"]))
            train_acc.append(float(row["train_acc"]))
            val_acc.append(float(row["val_acc"]))
            phases.append(int(row["phase"]))
            # Mass-asymmetry columns are present only in logs from this version onward;
            # older logs will have an empty string which we convert to NaN.
            def _parse_float(s):
                return float(s) if s else float("nan")
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

    # Draw every trend curve below with point markers, consistent with the
    # QCD/signal trend plots.  Saved and restored at the end so the marker style
    # does not leak into the GIF mean lines drawn later in the same epoch.
    _marker_keys = ("lines.marker", "lines.markersize", "lines.markeredgecolor",
                    "lines.markeredgewidth")
    _saved_rc = {k: plt.rcParams[k] for k in _marker_keys}
    plt.rcParams.update({
        "lines.marker": "o",
        "lines.markersize": 3.5,
        "lines.markeredgecolor": "white",
        "lines.markeredgewidth": 0.5,
    })

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
                label=r"Phase 1 $\rightarrow$ 2" if idx == 0 else None,
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
    # Plain-decimal log ticks (avoid mathtext 10^{-n} minus missing from serif fonts).
    from matplotlib import ticker as _mticker
    ax.yaxis.set_major_formatter(_mticker.FuncFormatter(lambda y, _pos: f"{y:g}"))
    ax.yaxis.set_minor_formatter(_mticker.NullFormatter())
    ax.legend(frameon=False)
    _style_axis(ax)
    fig.tight_layout()
    loss_path = plots_dir / f"loss_{tag}.pdf"
    fig.savefig(loss_path)
    plt.close(fig)
    print(f"  -> Saved loss plot     : {loss_path}")
    saved_paths.append(loss_path)

    # --- Accuracy plot ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_acc, label="Train acc", color="steelblue")
    ax.plot(epochs, val_acc, label="Val acc", color="darkorange")
    _add_phase_lines(ax)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs Epoch")
    ax.legend(frameon=False)
    _style_axis(ax)
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
            color="steelblue", alpha=0.2, label=r"Train $\pm1\sigma$",
        )

        # Val: line + ±1σ shaded band
        ax.plot(asym_epochs, asym_val_avg, label="Val mean asym", color="darkorange")
        ax.fill_between(
            asym_epochs,
            [m - s for m, s in zip(asym_val_avg, asym_val_std)],
            [m + s for m, s in zip(asym_val_avg, asym_val_std)],
            color="darkorange", alpha=0.2, label=r"Val $\pm1\sigma$",
        )

        _add_phase_lines(ax)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mass asymmetry of chosen interpretation")
        ax.set_title("Mass Asymmetry of Chosen Interpretation vs Epoch")
        ax.set_yscale("log")
        # Render log-axis ticks as plain decimals (avoids mathtext 10^{-n} whose
        # minus sign is missing from refined serif fonts like Garamond).
        from matplotlib import ticker as _mticker
        ax.yaxis.set_major_formatter(_mticker.FuncFormatter(lambda y, _pos: f"{y:g}"))
        ax.yaxis.set_minor_formatter(_mticker.NullFormatter())
        ax.legend(frameon=False)
        _style_axis(ax)
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
        ax.legend(frameon=False)
        _style_axis(ax)
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
            color="steelblue", alpha=0.2, label=r"Train $\pm1\sigma$",
        )

        ax.plot(mpt_epochs, mpt_val_avg, label="Val mean max-triplet pT", color="darkorange")
        ax.fill_between(
            mpt_epochs,
            [m - s for m, s in zip(mpt_val_avg, mpt_val_std)],
            [m + s for m, s in zip(mpt_val_avg, mpt_val_std)],
            color="darkorange", alpha=0.2, label=r"Val $\pm1\sigma$",
        )

        _add_phase_lines(ax)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Max-triplet scalar sum pT of chosen interpretation")
        ax.set_title("Max-Triplet Scalar Sum pT of Chosen Interpretation vs Epoch")
        ax.legend(frameon=False)
        _style_axis(ax)
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
        ax.plot(dphi_epochs, dphi_train_avg, label=r"Train mean $\Delta\phi$", color="steelblue")
        ax.fill_between(
            dphi_epochs,
            [m - s for m, s in zip(dphi_train_avg, dphi_train_std)],
            [m + s for m, s in zip(dphi_train_avg, dphi_train_std)],
            color="steelblue", alpha=0.2, label=r"Train $\pm1\sigma$",
        )
        ax.plot(dphi_epochs, dphi_val_avg, label=r"Val mean $\Delta\phi$", color="darkorange")
        ax.fill_between(
            dphi_epochs,
            [m - s for m, s in zip(dphi_val_avg, dphi_val_std)],
            [m + s for m, s in zip(dphi_val_avg, dphi_val_std)],
            color="darkorange", alpha=0.2, label=r"Val $\pm1\sigma$",
        )
        _add_phase_lines(ax)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"$\Delta\phi$ between parent candidates (rad)")
        ax.set_title(r"$\Delta\phi$ Between Parent Candidates of Chosen Interpretation vs Epoch")
        ax.legend(frameon=False)
        _style_axis(ax)
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
            color="mediumseagreen", alpha=0.2, label=r"Train $\pm1\sigma$",
        )
        ax.plot(dem_epochs, dem_val_avg, label="Val mean democracy", color="darkorange")
        ax.fill_between(
            dem_epochs,
            [m - s for m, s in zip(dem_val_avg, dem_val_std)],
            [m + s for m, s in zip(dem_val_avg, dem_val_std)],
            color="darkorange", alpha=0.2, label=r"Val $\pm1\sigma$",
        )
        _add_phase_lines(ax)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("pT democracy = avg(min pT / max pT) per triplet")
        ax.set_title("pT Democracy of Chosen Interpretation vs Epoch")
        ax.legend(frameon=False)
        _style_axis(ax)
        fig.tight_layout()
        dem_path = plots_dir / f"democracy_{tag}.pdf"
        fig.savefig(dem_path)
        plt.close(fig)
        print(f"  -> Saved democracy plot: {dem_path}")
        saved_paths.append(dem_path)

    plt.rcParams.update(_saved_rc)
    return saved_paths


def _run_epoch(
    model, loader, ce_loss_fn, mse_loss_fn, lambda_adv, device, optimizer=None,
    tf_ratio=1.0, lambda_sym=0.0, lambda_qcd=0.0, lambda_isr=1.0, lambda_isr_direct=0.0,
    lambda_distill=0.0, distill_temperature=4.0,
    lambda_entropy_asym=0.0, lambda_entropy_mass=0.0,
    lambda_bg=0.0, beta_bg=1.0, bg_soft_weight=1.0, bg_asym_cut=0.0,
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
    all_pred_max_boost = []
    all_pred_avg_boost = []
    all_pred_dalitz_x = []   # validation only; per-event (B, 2) for the two triplets
    all_pred_dalitz_y = []
    all_mass_pred = []
    all_mass_true = []
    all_pred_is_bkg = []
    all_pred_asym_max = []   # per-event achievable max asymmetry (over assignments)
    all_pred_mass_min = []   # per-event achievable min mass_sum (over assignments)
    total_sig_samples = 0
    factored = model.has_isr
    # Label smoothing used by ce_loss_fn, replicated here so the per-event
    # cross-entropies (needed for signal/background masking) match the
    # scalar ce_loss_fn exactly when no background events are present.
    label_smoothing = getattr(ce_loss_fn, "label_smoothing", 0.0)

    def _masked_mean(per_event, mask):
        """Mean of a per-event (B,) tensor over the events selected by *mask*.

        Returns 0 (no loss contribution, no gradient) when the mask is empty,
        so an all-signal or all-background batch is handled gracefully.
        """
        mask_f = mask.to(per_event.dtype)
        return (per_event * mask_f).sum() / mask_f.sum().clamp(min=1.0)

    for batch in loader:
        four_mom = batch["four_momenta"].to(device)
        labels = batch["label"].to(device)
        parent_mass = batch["parent_mass"].to(device)
        # Per-event signal/background tag.  Background (QCD) events have no truth
        # assignment: they are excluded from the supervised loss and instead
        # drive the background-rejection term.  Default all-signal when absent.
        if "is_background" in batch:
            is_bkg = batch["is_background"].to(device).bool()
        else:
            is_bkg = torch.zeros(labels.shape[0], dtype=torch.bool, device=device)
        sig_mask = ~is_bkg

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
                total_isr_correct += ((isr_logits.argmax(dim=-1) == isr_labels) & sig_mask).sum().item()
                total_grp_correct += ((gt_grp_logits.argmax(dim=-1) == grouping_labels) & sig_mask).sum().item()
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

                # Per-event cross-entropies (reduction="none") so signal and
                # background events can be combined with different objectives;
                # reduced to a scalar with the signal mask below.  Equivalent to
                # the original ce_loss_fn(...) when every event is signal.
                loss_isr = F.cross_entropy(
                    isr_logits, isr_labels, label_smoothing=label_smoothing, reduction="none"
                )

                batch_idx = torch.arange(labels.shape[0], device=device)
                gt_grp_logits = grouping_logits[batch_idx, isr_labels]
                loss_grp_tf = F.cross_entropy(
                    gt_grp_logits, grouping_labels, label_smoothing=label_smoothing, reduction="none"
                )

                # Blend teacher-forced factored loss with flat end-to-end loss
                # tf_ratio=1: fully teacher-forced (original); tf_ratio=0: flat CE only
                # lambda_isr upweights the ISR loss to compensate for a gradient imbalance:
                # each signal jet appears in all num_groupings groupings, so loss_grp_tf
                # produces num_groupings gradient paths per signal jet while the ISR jet
                # (excluded from every group) receives gradient only from loss_isr.
                # Scaling loss_isr by lambda_isr partially rebalances this asymmetry.
                loss_flat = F.cross_entropy(
                    logits, labels, label_smoothing=label_smoothing, reduction="none"
                )
                # loss_ce is kept per-event (B,) and reduced with the signal mask
                # below; background events feed only the bg-rejection term.
                loss_ce = tf_ratio * (lambda_isr * loss_isr + loss_grp_tf) + (1.0 - tf_ratio) * loss_flat

                total_isr_correct += ((isr_logits.argmax(dim=-1) == isr_labels) & sig_mask).sum().item()
                total_grp_correct += ((gt_grp_logits.argmax(dim=-1) == grouping_labels) & sig_mask).sum().item()

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
                    loss_isr_direct = F.cross_entropy(
                        isr_logits_direct, isr_labels, label_smoothing=label_smoothing, reduction="none"
                    )
                    loss_ce = loss_ce + lambda_isr_direct * loss_isr_direct
            else:
                loss_ce = F.cross_entropy(
                    logits, labels, label_smoothing=label_smoothing, reduction="none"
                )

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
                # Per-event KL (summed over assignments); reduced with the signal
                # mask below.  Equals reduction="batchmean" when all events are signal.
                loss_distill = F.kl_div(
                    student_log_probs, teacher_probs, reduction="none"
                ).sum(dim=-1)                                  # (batch,)
                loss_ce = loss_ce + lambda_distill * loss_distill

            # Mass symmetry auxiliary loss: minimize expected |m1-m2|/(m1+m2) over assignments
            if lambda_sym > 0 and "mass_asym_flat" in output:
                mass_asym = output["mass_asym_flat"].detach()  # (batch, num_assignments)
                probs = logits.softmax(dim=-1)
                loss_sym = (probs * mass_asym).sum(dim=-1)     # (batch,) per-event
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
                loss_qcd_term = -(H * expected_asym)                       # (batch,) per-event
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
                    loss_entropy_asym = -(norm_entropy * expected_asym_ent)             # (batch,)
                    loss_ce = loss_ce + lambda_entropy_asym * loss_entropy_asym

                if lambda_entropy_mass > 0 and "mass_sum_flat" in output:
                    mass_sum_ent = output["mass_sum_flat"].detach()                      # (B, N)
                    expected_mass_sum = (probs_ent * mass_sum_ent).sum(dim=-1)          # (batch,)
                    loss_entropy_mass_term = (norm_entropy * expected_mass_sum)         # (batch,)
                    loss_ce = loss_ce + lambda_entropy_mass * loss_entropy_mass_term

            # Reduce the per-event supervised loss over SIGNAL events only.
            # Background (QCD) events contribute nothing here; they are handled
            # by the background-rejection term below.  When every event is signal
            # this is identical to the previous batch-mean reductions.
            loss_ce = _masked_mean(loss_ce, sig_mask)

            # Background-rejection loss.  For each QCD/background event we want the
            # network to COMMIT to the interpretation that is most obviously not
            # signal-like and duck out of the signal region, rather than settling
            # on a mildly-asymmetric / mildly-low-mass compromise.
            #
            # Anti-signal score per assignment is an OR of two normalised extremes
            # (each in [0,1]): high mass asymmetry, or low average mass relative to
            # what is achievable for that event.  Taking the max (not a sum) means
            # the target is the single interpretation that is extreme in EITHER
            # dimension — the blatantly-non-signal choice the event affords.
            #   (a) hard CE toward argmax(bg_score) commits to that choice;
            #   (b) a soft term maximises the expected anti-signal score so the
            #       bulk of the QCD probability (not just the argmax) is pushed out;
            #   (c) an optional asymmetry cut explicitly evacuates the signal region
            #       by penalising probability left on low-asymmetry interpretations.
            # Masked to background events, so signal reconstruction is untouched.
            if (
                lambda_bg > 0
                and is_bkg.any()
                and "mass_sum_flat" in output
                and "mass_asym_flat" in output
            ):
                asym = output["mass_asym_flat"].detach()       # (batch, N) in [0, 1]
                msum = output["mass_sum_flat"].detach()        # (batch, N), HT units
                probs = logits.softmax(dim=-1)

                # Per-event mass "lowness" in [0,1] (1 = lowest-mass interpretation
                # available), so the low-mass route is comparable to the asymmetry
                # route regardless of the event's absolute mass scale.
                msum_min = msum.min(dim=-1, keepdim=True).values
                msum_max = msum.max(dim=-1, keepdim=True).values
                mass_low = (msum_max - msum) / (msum_max - msum_min).clamp(min=1e-6)

                # OR of the two extremes; beta_bg tilts toward the low-mass route.
                bg_score = torch.maximum(asym, beta_bg * mass_low)          # (batch, N)

                # (a) commit to the single most anti-signal interpretation.
                bg_target = bg_score.argmax(dim=-1)                          # (batch,)
                loss_bg = F.cross_entropy(logits, bg_target, reduction="none")   # (batch,)

                # (b) push the whole distribution toward high anti-signal score.
                if bg_soft_weight > 0:
                    loss_bg = loss_bg + bg_soft_weight * (1.0 - (probs * bg_score).sum(dim=-1))

                # (c) explicit signal-region exit: penalise probability on
                #     low-asymmetry (signal-like) interpretations below the cut.
                if bg_asym_cut > 0:
                    tau = max(0.25 * bg_asym_cut, 1e-3)
                    in_sr = torch.sigmoid((bg_asym_cut - asym) / tau)        # ~1 below cut
                    loss_bg = loss_bg + (probs * in_sr).sum(dim=-1)

                loss_ce = loss_ce + lambda_bg * _masked_mean(loss_bg, is_bkg)

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
        total_sig_samples += int(sig_mask.sum().item())

        preds = logits.argmax(dim=-1)
        # Assignment accuracy is only defined for signal events (background/QCD
        # events have no truth assignment), so restrict the counters to signal.
        total_correct += ((preds == labels) & sig_mask).sum().item()

        _, top5 = logits.topk(5, dim=-1)
        total_correct5 += ((top5 == labels.unsqueeze(-1)).any(dim=-1) & sig_mask).sum().item()

        # Per-event background flag, aligned with the per-event distribution
        # arrays below so downstream plotting can split signal vs QCD.
        all_pred_is_bkg.append(is_bkg.detach().cpu())

        if "mass_asym_flat" in output:
            mass_asym_flat = output["mass_asym_flat"].detach()  # (batch, num_assignments)
            pred_asym = mass_asym_flat.gather(1, preds.unsqueeze(1)).squeeze(1)  # (batch,)
            total_mass_asym += pred_asym.sum().item()
            total_mass_asym_samples += batch_size
            all_pred_asym.append(pred_asym.cpu())
            all_pred_correct.append((preds == labels).cpu())  # aligned with pred_asym
            # Best achievable asymmetry for this event (ceiling for the bg push).
            all_pred_asym_max.append(mass_asym_flat.max(dim=-1).values.cpu())

        if "mass_sum_flat" in output:
            mass_sum_flat = output["mass_sum_flat"].detach()  # (batch, num_assignments)
            pred_mass_sum = mass_sum_flat.gather(1, preds.unsqueeze(1)).squeeze(1)  # (batch,)
            all_pred_mass_sum.append(pred_mass_sum.cpu())
            # Lowest achievable mass_sum for this event (floor for the bg push).
            all_pred_mass_min.append(mass_sum_flat.min(dim=-1).values.cpu())

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

            # Lorentz boost γ = E/m of each parent candidate (E/m is a ratio, so it
            # is invariant under the HT normalisation of the inputs).  Track the
            # larger of the two and the average across the two triplets.
            E_b = four_mom[:, :, 0]
            pz_b = four_mom[:, :, 3]
            E_g1 = E_b.gather(1, g1_jets).sum(dim=1)
            E_g2 = E_b.gather(1, g2_jets).sum(dim=1)
            pz_g1 = pz_b.gather(1, g1_jets).sum(dim=1)
            pz_g2 = pz_b.gather(1, g2_jets).sum(dim=1)
            m2_g1 = (E_g1**2 - sum_px_g1**2 - sum_py_g1**2 - pz_g1**2).clamp(min=1e-8)
            m2_g2 = (E_g2**2 - sum_px_g2**2 - sum_py_g2**2 - pz_g2**2).clamp(min=1e-8)
            gamma1 = (E_g1 / m2_g1.sqrt()).clamp(max=200.0)
            gamma2 = (E_g2 / m2_g2.sqrt()).clamp(max=200.0)
            all_pred_max_boost.append(torch.maximum(gamma1, gamma2).detach().cpu())
            all_pred_avg_boost.append(((gamma1 + gamma2) / 2.0).detach().cpu())

            # Dalitz coordinates (validation only, used for the 2-D Dalitz plot):
            # for each triplet, the pairwise invariant-mass-squared of its pT-ordered
            # jets normalised by the triplet m²:  x = m²(lead,sub)/M², y = m²(lead,third)/M².
            if optimizer is None:
                def _dalitz_xy(idx, m2_parent):
                    tri = torch.gather(four_mom, 1, idx.unsqueeze(-1).expand(-1, -1, 4))  # (B,3,4)
                    ptt = torch.sqrt(tri[..., 1] ** 2 + tri[..., 2] ** 2)
                    order = torch.argsort(ptt, dim=1, descending=True)
                    tri = torch.gather(tri, 1, order.unsqueeze(-1).expand(-1, -1, 4))
                    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]

                    def _m2(p, q):
                        s = p + q
                        return s[:, 0] ** 2 - s[:, 1] ** 2 - s[:, 2] ** 2 - s[:, 3] ** 2
                    denom = m2_parent.clamp(min=1e-8)
                    return _m2(a, b) / denom, _m2(a, c) / denom

                x1, y1 = _dalitz_xy(g1_jets, m2_g1)
                x2, y2 = _dalitz_xy(g2_jets, m2_g2)
                all_pred_dalitz_x.append(torch.stack([x1, x2], dim=1).detach().cpu())  # (B, 2)
                all_pred_dalitz_y.append(torch.stack([y1, y2], dim=1).detach().cpu())

        mass_mask = parent_mass > 0
        if mass_mask.any():
            all_mass_pred.append(mass_pred[mass_mask].detach().cpu())
            all_mass_true.append(parent_mass[mass_mask].detach().cpu())

    avg_loss = total_loss / max(total_samples, 1)
    # Accuracy is over signal events only (background events have no truth label).
    acc = total_correct / max(total_sig_samples, 1)
    acc5 = total_correct5 / max(total_sig_samples, 1)

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
    if all_pred_asym_max:
        result["pred_asym_achievable_values"] = torch.cat(all_pred_asym_max).numpy()
    if all_pred_mass_sum:
        result["pred_mass_sum_values"] = torch.cat(all_pred_mass_sum).numpy()  # full per-event array
    if all_pred_mass_min:
        result["pred_mass_sum_achievable_values"] = torch.cat(all_pred_mass_min).numpy()
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
    if all_pred_max_boost:
        result["pred_max_boost_values"] = torch.cat(all_pred_max_boost).numpy()
    if all_pred_avg_boost:
        result["pred_avg_boost_values"] = torch.cat(all_pred_avg_boost).numpy()
    if all_pred_dalitz_x:
        result["pred_dalitz_x_values"] = torch.cat(all_pred_dalitz_x).numpy()  # (N, 2)
        result["pred_dalitz_y_values"] = torch.cat(all_pred_dalitz_y).numpy()  # (N, 2)
    if all_pred_is_bkg:
        result["pred_is_bkg_values"] = torch.cat(all_pred_is_bkg).numpy()
    if factored:
        result["isr_acc"] = total_isr_correct / max(total_sig_samples, 1)
        result["grp_acc"] = total_grp_correct / max(total_sig_samples, 1)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train jet assignment model")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--data", type=str, default=None, help="Path to signal HDF5 data (glob pattern)")
    parser.add_argument(
        "--qcd-data",
        type=str,
        default=None,
        help=(
            "Path to QCD/background HDF5 data (glob pattern).  Providing this "
            "automatically turns on the background-rejection loss (lambda_bg), "
            "training signal and QCD jointly in a single step.  Overrides "
            "data.qcd_data_path in the config."
        ),
    )
    args = parser.parse_args()
    train(config_path=args.config, data_path=args.data, qcd_data_path=args.qcd_data)
