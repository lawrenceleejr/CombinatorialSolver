"""
Standalone ONNX diagnostic script for the jet assignment model.

Takes an ONNX model file (exported from export_onnx.py) and one or more HDF5
data files, runs inference, then prints a kinematic breakdown of mis-classified
events grouped by failure mode.

Does NOT require the src/ package — only numpy, h5py, and onnxruntime.

Usage
-----
    python scripts/diagnose_onnx.py \\
        --model  onnx_models/ml_model.onnx \\
        --data   data/test_sample.h5 \\
        --num-jets 7

Optional flags:
    --no-normalize      Pass raw (un-normalised) four-momenta to the model.
                        Use this for the classical solver ONNX; the ML model
                        always needs HT-normalised inputs.
    --n-examples 30     How many most-confident wrong events to print in full.
    --output results/   Directory for failure_diagnostics.csv (default: .).
    --batch-size 2048   ONNX inference batch size (default: 1024).
    --num-jets 7        Jets per event: 6 (no ISR) or 7 (with ISR).

Output
------
Printed to stdout:
  * Overall accuracy (top-1, top-5)
  * Failure mode counts (ISR-wrong / grp-wrong / both-wrong)
  * Per-mode mean ± std table for key kinematic features
  * Detailed snapshot for the N most-confidently wrong events

Written to disk:
  * <output>/failure_diagnostics.csv  (one row per event)
"""

import argparse
import csv
import glob as globmod
import math
import sys
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np

# ── Graceful onnxruntime import ─────────────────────────────────────────────
try:
    import onnxruntime as ort
except ImportError:
    print(
        "ERROR: onnxruntime is not installed.\n"
        "Install it with:  pip install onnxruntime\n"
        "or add it to requirements.txt and reinstall.",
        file=sys.stderr,
    )
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Combinatorics  (replicated from src/combinatorics.py)
# ═══════════════════════════════════════════════════════════════════════════

def enumerate_assignments(num_jets: int = 7):
    """Return list of (isr_or_None, group1_tuple, group2_tuple)."""
    assignments = []
    if num_jets == 6:
        all_jets = list(range(6))
        seen = set()
        for g1 in combinations(all_jets, 3):
            g2 = tuple(j for j in all_jets if j not in g1)
            canon = (min(g1, g2), max(g1, g2))
            if canon not in seen:
                seen.add(canon)
                assignments.append((None, canon[0], canon[1]))
        return assignments
    for isr in range(num_jets):
        remaining = [j for j in range(num_jets) if j != isr]
        seen = set()
        for g1 in combinations(remaining, 3):
            g2 = tuple(j for j in remaining if j not in g1)
            canon = (min(g1, g2), max(g1, g2))
            if canon not in seen:
                seen.add(canon)
                assignments.append((isr, canon[0], canon[1]))
    return assignments


def build_group_index_arrays(num_jets: int):
    """Return (g1_idx, g2_idx) numpy arrays of shape (N_assignments, 3)."""
    asgn = enumerate_assignments(num_jets)
    g1 = np.array([list(a[1]) for a in asgn], dtype=np.int64)
    g2 = np.array([list(a[2]) for a in asgn], dtype=np.int64)
    return g1, g2


def build_flat_to_factored(num_jets: int):
    """Return (N_assignments, 2) array mapping flat → (isr_idx, grp_idx).

    Only meaningful for num_jets == 7 (10-way grouping × 7 ISR jets).
    For 6-jet mode returns None.
    """
    if num_jets != 7:
        return None
    asgn = enumerate_assignments(num_jets)
    # assignment_idx = isr_idx * 10 + grouping_idx  (by construction)
    # We rebuild the grouping index as the position within the ISR's 10-subset.
    result = np.zeros((len(asgn), 2), dtype=np.int64)
    per_isr_count = {}
    for flat_idx, (isr, g1, g2) in enumerate(asgn):
        cnt = per_isr_count.get(isr, 0)
        result[flat_idx, 0] = isr
        result[flat_idx, 1] = cnt
        per_isr_count[isr] = cnt + 1
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Data loading  (replicated from src/dataset.py)
# ═══════════════════════════════════════════════════════════════════════════

def _inv_mass(v: np.ndarray) -> np.ndarray:
    """Invariant mass from (E, px, py, pz) last axis."""
    m2 = v[..., 0]**2 - v[..., 1]**2 - v[..., 2]**2 - v[..., 3]**2
    return np.sqrt(np.maximum(m2, 0.0))


def _pt_eta_phi_mass_to_epxpypz(pt, eta, phi, mass):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E  = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return np.stack([E, px, py, pz], axis=-1)


def _compute_mass_asym_labels(four_mom_raw: np.ndarray, num_jets: int):
    """Return (labels, parent_masses) using argmin |m1-m2| criterion."""
    g1_idx, g2_idx = build_group_index_arrays(num_jets)
    # four_mom_raw: (N, num_jets, 4)
    g1_sum = four_mom_raw[:, g1_idx, :].sum(axis=2)  # (N, N_assign, 4)
    g2_sum = four_mom_raw[:, g2_idx, :].sum(axis=2)
    m1 = _inv_mass(g1_sum)  # (N, N_assign)
    m2 = _inv_mass(g2_sum)
    labels = np.abs(m1 - m2).argmin(axis=1).astype(np.int64)
    n = len(labels)
    parent_masses = ((m1[np.arange(n), labels] + m2[np.arange(n), labels]) / 2.0).astype(np.float32)
    return labels, parent_masses


def load_hdf5(path: str, num_jets: int):
    """Load one HDF5 file.

    Returns:
        four_mom_raw  : (N, num_jets, 4) float32, un-normalised (E, px, py, pz)
        four_mom_norm : (N, num_jets, 4) float32, HT-normalised
        labels        : (N,) int64, truth assignment indices (≥ 0)
        ht            : (N,) float32, scalar HT
    """
    with h5py.File(path, "r") as f:
        # Detect layout
        if "INPUTS" in f and "Source" in f["INPUTS"]:
            pt   = f["INPUTS/Source/pt"][:]
            eta  = f["INPUTS/Source/eta"][:]
            phi  = f["INPUTS/Source/phi"][:]
            mass = f["INPUTS/Source/mass"][:]
            mask = f["INPUTS/Source/MASK"][:].astype(bool)
        elif "jet_features" in f:
            jf   = f["jet_features"][:]
            pt   = jf[:, :, 0]
            eta  = jf[:, :, 1]
            phi  = jf[:, :, 2]
            mass = jf[:, :, 3]
            mask = f["jet_mask"][:].astype(bool) if "jet_mask" in f else (pt > 0)
        else:
            raise ValueError(f"Unrecognised HDF5 layout in {path}. "
                             "Expected INPUTS/Source/* or jet_features.")

    n_events = pt.shape[0]
    max_jets  = pt.shape[1]
    effective_num_jets = min(num_jets, max_jets)

    four_mom_raw  = np.zeros((n_events, effective_num_jets, 4), dtype=np.float32)
    sort_order    = np.full((n_events, effective_num_jets), -1, dtype=np.int64)
    ht_arr        = np.ones(n_events, dtype=np.float32)

    for i in range(n_events):
        valid = np.where(mask[i])[0]
        if len(valid) == 0:
            continue
        order = np.argsort(-pt[i, valid])
        sel = valid[order[:effective_num_jets]]
        n_sel = len(sel)
        sort_order[i, :n_sel] = sel
        four_mom_raw[i, :n_sel] = _pt_eta_phi_mass_to_epxpypz(
            pt[i, sel], eta[i, sel], phi[i, sel], mass[i, sel]
        )
        ht_arr[i] = pt[i, valid].sum()

    labels, _ = _compute_mass_asym_labels(four_mom_raw, effective_num_jets)

    # Filter events with invalid labels (shouldn't happen for mass-asym labels, but be safe)
    valid_mask = labels >= 0
    four_mom_raw = four_mom_raw[valid_mask]
    labels       = labels[valid_mask]
    ht_arr       = ht_arr[valid_mask]

    four_mom_norm = four_mom_raw / ht_arr[:, None, None].clip(1e-6)

    return four_mom_raw, four_mom_norm, labels, ht_arr


def load_all_hdf5(paths: list[str], num_jets: int):
    """Load and concatenate multiple HDF5 files."""
    raws, norms, lbls, hts = [], [], [], []
    for p in paths:
        print(f"  Loading {p} …", flush=True)
        r, n, l, h = load_hdf5(p, num_jets)
        raws.append(r); norms.append(n); lbls.append(l); hts.append(h)
        print(f"    {len(l)} valid events loaded.")
    return (
        np.concatenate(raws),
        np.concatenate(norms),
        np.concatenate(lbls),
        np.concatenate(hts),
    )


# ═══════════════════════════════════════════════════════════════════════════
# 3.  ONNX inference
# ═══════════════════════════════════════════════════════════════════════════

def run_onnx(model_path: str, four_mom: np.ndarray, batch_size: int = 1024):
    """Run ONNX model in batches, return (preds, logits) numpy arrays."""
    sess = ort.InferenceSession(
        model_path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name   # 'four_momenta'

    all_logits = []
    n = len(four_mom)
    for start in range(0, n, batch_size):
        batch = four_mom[start : start + batch_size].astype(np.float32)
        (logits,) = sess.run(None, {input_name: batch})
        all_logits.append(logits)
    all_logits = np.concatenate(all_logits, axis=0)
    preds = all_logits.argmax(axis=-1)
    return preds, all_logits


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Per-event kinematic features
# ═══════════════════════════════════════════════════════════════════════════

def _group_kinematics(raw_4mom: np.ndarray, jet_idx) -> dict:
    """Compute kinematic features for one triplet.

    Args:
        raw_4mom : (num_jets, 4) un-normalised (E, px, py, pz)
        jet_idx  : 3 jet indices

    Returns dict with:
        m_group, pt_max_ratio, pt_cv, d2,
        dalitz_min, dalitz_mid, dalitz_max
    """
    jets = raw_4mom[list(jet_idx)]            # (3, 4)
    E    = np.maximum(jets[:, 0], 1e-8)
    px, py, pz = jets[:, 1], jets[:, 2], jets[:, 3]
    pt = np.sqrt(px**2 + py**2).clip(1e-8)

    pt_sorted = np.sort(pt)[::-1]
    pt_max_ratio = pt_sorted[0] / max(pt_sorted[2], 1e-8)
    pt_cv = pt.std() / max(pt.mean(), 1e-8)

    p_grp = jets.sum(axis=0)
    m2 = p_grp[0]**2 - p_grp[1]**2 - p_grp[2]**2 - p_grp[3]**2
    m_group = math.sqrt(max(m2, 0.0))

    pairs = [(0, 1), (0, 2), (1, 2)]
    sub_masses = []
    dr_list = []
    eta = np.arcsinh(pz / pt)
    phi = np.arctan2(py, px)
    for i, j in pairs:
        p_ij = jets[i] + jets[j]
        m2_ij = p_ij[0]**2 - p_ij[1]**2 - p_ij[2]**2 - p_ij[3]**2
        sub_masses.append(math.sqrt(max(m2_ij, 0.0)))
        deta = eta[i] - eta[j]
        dphi_raw = phi[i] - phi[j]
        dphi = (dphi_raw + math.pi) % (2 * math.pi) - math.pi
        dr_list.append(math.sqrt(deta**2 + dphi**2))

    denom = max(m_group, 1e-8)
    sub_sorted = sorted(s / denom for s in sub_masses)

    # Energy Correlation Functions for D₂
    E_sum = max(E.sum(), 1e-8)
    z = E / E_sum
    ecf2 = sum(z[i] * z[j] * dr_list[k] for k, (i, j) in enumerate(pairs))
    ecf3 = z[0] * z[1] * z[2] * dr_list[0] * dr_list[1] * dr_list[2]
    d2 = ecf3 / max(ecf2, 1e-8) ** 2

    return {
        "m_group":       m_group,
        "pt_max_ratio":  float(pt_max_ratio),
        "pt_cv":         float(pt_cv),
        "d2":            float(d2),
        "dalitz_min":    sub_sorted[0],
        "dalitz_mid":    sub_sorted[1],
        "dalitz_max":    sub_sorted[2],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Diagnostics
# ═══════════════════════════════════════════════════════════════════════════

def diagnose(
    four_mom_raw: np.ndarray,
    preds: np.ndarray,
    labels: np.ndarray,
    logits: np.ndarray,
    num_jets: int,
    n_examples: int = 20,
    output_dir: str = ".",
) -> None:
    assignments   = enumerate_assignments(num_jets)
    flat2factored = build_flat_to_factored(num_jets)   # None for 6-jet
    has_isr       = (num_jets == 7)

    N = len(preds)
    correct_mask = (preds == labels)

    # Softmax confidence
    exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    confidence = probs.max(axis=-1)

    # top-5 accuracy
    num_assign = logits.shape[1]
    topk = min(5, num_assign)
    top5_idx = np.argpartition(logits, -topk, axis=-1)[:, -topk:]
    acc5 = np.any(top5_idx == labels[:, None], axis=-1).mean()

    # ── failure modes ───────────────────────────────────────────────────────
    if has_isr and flat2factored is not None:
        truth_isr = flat2factored[labels, 0]
        pred_isr  = flat2factored[preds,  0]
        truth_grp = flat2factored[labels, 1]
        pred_grp  = flat2factored[preds,  1]
        isr_wrong = (pred_isr != truth_isr)
        grp_wrong = (pred_grp != truth_grp)
        mode = np.full(N, "correct", dtype=object)
        wrong = ~correct_mask
        mode[wrong &  isr_wrong & ~grp_wrong] = "ISR-wrong"
        mode[wrong & ~isr_wrong &  grp_wrong] = "grp-wrong"
        mode[wrong &  isr_wrong &  grp_wrong] = "both-wrong"
    else:
        mode = np.where(correct_mask, "correct", "wrong")

    # ── per-event feature computation ───────────────────────────────────────
    feature_keys = [
        "m_pred_g1", "m_pred_g2", "mass_asym_pred",
        "pt_max_ratio_g1", "pt_max_ratio_g2",
        "pt_cv_g1", "pt_cv_g2",
        "d2_g1", "d2_g2",
        "dalitz_min_g1", "dalitz_mid_g1", "dalitz_max_g1",
        "dalitz_min_g2", "dalitz_mid_g2", "dalitz_max_g2",
        "confidence",
    ]
    if has_isr:
        feature_keys.insert(0, "isr_pt")

    rows = []
    feat_mat = {k: np.full(N, np.nan) for k in feature_keys}

    for i in range(N):
        p_idx = int(preds[i])
        isr_p, g1_p, g2_p = assignments[p_idx]
        raw = four_mom_raw[i]

        gk1 = _group_kinematics(raw, g1_p)
        gk2 = _group_kinematics(raw, g2_p)
        denom = max(gk1["m_group"] + gk2["m_group"], 1e-8)
        mass_asym = abs(gk1["m_group"] - gk2["m_group"]) / denom

        row = {"event_idx": i, "mode": mode[i]}

        if has_isr and isr_p is not None:
            isr_jet = raw[isr_p]
            isr_pt  = math.sqrt(isr_jet[1]**2 + isr_jet[2]**2)
            row["isr_pt"] = isr_pt
            feat_mat["isr_pt"][i] = isr_pt

        vals = {
            "m_pred_g1":       gk1["m_group"],
            "m_pred_g2":       gk2["m_group"],
            "mass_asym_pred":  mass_asym,
            "pt_max_ratio_g1": gk1["pt_max_ratio"],
            "pt_max_ratio_g2": gk2["pt_max_ratio"],
            "pt_cv_g1":        gk1["pt_cv"],
            "pt_cv_g2":        gk2["pt_cv"],
            "d2_g1":           gk1["d2"],
            "d2_g2":           gk2["d2"],
            "dalitz_min_g1":   gk1["dalitz_min"],
            "dalitz_mid_g1":   gk1["dalitz_mid"],
            "dalitz_max_g1":   gk1["dalitz_max"],
            "dalitz_min_g2":   gk2["dalitz_min"],
            "dalitz_mid_g2":   gk2["dalitz_mid"],
            "dalitz_max_g2":   gk2["dalitz_max"],
            "confidence":      float(confidence[i]),
        }
        row.update(vals)
        rows.append(row)
        for k, v in vals.items():
            feat_mat[k][i] = v

    # ── header ──────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("ONNX FAILURE DIAGNOSTICS")
    print("=" * 72)
    print(f"Events : {N}")
    print(f"Top-1  : {correct_mask.mean():.4f}  ({correct_mask.sum()}/{N})")
    print(f"Top-{topk}  : {acc5:.4f}")
    print()

    unique_modes = ["correct"] + sorted(m for m in set(mode) if m != "correct")
    for m in unique_modes:
        n_m = (mode == m).sum()
        print(f"  {m:<15}: {n_m:>6} events  ({n_m / N:.1%})")

    # ── summary table ───────────────────────────────────────────────────────
    print()
    print(f"{'Feature':<22}", end="")
    for m in unique_modes:
        print(f"  {m:>26}", end="")
    print()
    print("-" * (22 + 28 * len(unique_modes)))

    for k in feature_keys:
        print(f"{k:<22}", end="")
        for m in unique_modes:
            vals = feat_mat[k][mode == m]
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                mu, sigma = vals.mean(), vals.std()
                print(f"  {mu:>10.3f} ± {sigma:<8.3f} (n={len(vals)})", end="")
            else:
                print(f"  {'n/a':>26}", end="")
        print()

    # ── most-confidently wrong ───────────────────────────────────────────────
    wrong_idx = np.where(mode != "correct")[0]
    if len(wrong_idx) == 0:
        print("\nNo wrong events — model is perfectly correct on this dataset.")
    else:
        order = confidence[wrong_idx].argsort()[::-1]
        top_n_idx = wrong_idx[order[:n_examples]]

        print(f"\n--- {min(n_examples, len(wrong_idx))} most-confident wrong predictions ---")
        for rank, i in enumerate(top_n_idx, 1):
            p_idx = int(preds[i])
            t_idx = int(labels[i])
            isr_p, g1_p, g2_p = assignments[p_idx]
            isr_t, g1_t, g2_t = assignments[t_idx]
            r = rows[i]
            print(f"\n[{rank}] event {i:>6}  mode={mode[i]:<12}  conf={confidence[i]:.3f}")
            print(f"     pred : ISR={isr_p}  g1={list(g1_p)}  g2={list(g2_p)}")
            print(f"     truth: ISR={isr_t}  g1={list(g1_t)}  g2={list(g2_t)}")
            if "isr_pt" in r:
                print(f"     ISR pT                 = {r['isr_pt']:.1f} GeV")
            print(f"     pred masses            = {r['m_pred_g1']:.1f} / {r['m_pred_g2']:.1f} GeV"
                  f"  (asym={r['mass_asym_pred']:.3f})")
            print(f"     pT hierarchy (g1/g2)   = {r['pt_max_ratio_g1']:.2f} / {r['pt_max_ratio_g2']:.2f}")
            print(f"     pT CoV       (g1/g2)   = {r['pt_cv_g1']:.3f} / {r['pt_cv_g2']:.3f}")
            print(f"     D₂           (g1/g2)   = {r['d2_g1']:.4f} / {r['d2_g2']:.4f}")
            print(f"     Dalitz g1 (min/mid/max) = "
                  f"{r['dalitz_min_g1']:.3f} / {r['dalitz_mid_g1']:.3f} / {r['dalitz_max_g1']:.3f}")
            print(f"     Dalitz g2 (min/mid/max) = "
                  f"{r['dalitz_min_g2']:.3f} / {r['dalitz_mid_g2']:.3f} / {r['dalitz_max_g2']:.3f}")

    # ── CSV export ───────────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = Path(output_dir) / "failure_diagnostics.csv"
    fieldnames = ["event_idx", "mode"] + (["isr_pt"] if has_isr else []) + [
        k for k in feature_keys if k != "isr_pt"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()}
            )
    print(f"\nFailure diagnostics written to: {csv_path}")
    print("=" * 72)


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose jet-assignment ONNX model failures on HDF5 data.\n"
            "Prints a kinematic breakdown of wrong predictions and writes\n"
            "failure_diagnostics.csv to the output directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to the ONNX model (e.g. onnx_models/ml_model.onnx).",
    )
    parser.add_argument(
        "--data", required=True, nargs="+",
        help="HDF5 file(s) or glob pattern(s) to evaluate on.",
    )
    parser.add_argument(
        "--num-jets", type=int, default=7, choices=[6, 7],
        help="Jets per event: 6 (no ISR) or 7 (with ISR).  Default: 7.",
    )
    parser.add_argument(
        "--no-normalize", action="store_true",
        help=(
            "Pass raw (un-normalised) four-momenta to the model.  "
            "Use this for the classical solver ONNX.  "
            "The ML model always expects HT-normalised inputs."
        ),
    )
    parser.add_argument(
        "--n-examples", type=int, default=20,
        help="Number of most-confident wrong events to print in full.  Default: 20.",
    )
    parser.add_argument(
        "--output", default=".",
        help="Output directory for failure_diagnostics.csv.  Default: current dir.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1024,
        help="ONNX inference batch size.  Default: 1024.",
    )
    args = parser.parse_args()

    # Resolve data files (support glob patterns)
    data_files = []
    for pattern in args.data:
        matches = sorted(globmod.glob(pattern))
        data_files.extend(matches if matches else [pattern])
    if not data_files:
        print("ERROR: No data files found.", file=sys.stderr)
        sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"\nLoading data ({args.num_jets}-jet mode) …")
    four_mom_raw, four_mom_norm, labels, ht = load_all_hdf5(data_files, args.num_jets)
    print(f"Total valid events: {len(labels)}\n")

    # ── Run ONNX inference ────────────────────────────────────────────────
    model_input = four_mom_raw if args.no_normalize else four_mom_norm
    print(f"Running ONNX inference on {args.model} …")
    preds, logits = run_onnx(args.model, model_input, args.batch_size)
    print("Inference complete.\n")

    # ── Diagnose ──────────────────────────────────────────────────────────
    diagnose(
        four_mom_raw=four_mom_raw,
        preds=preds,
        labels=labels,
        logits=logits,
        num_jets=args.num_jets,
        n_examples=args.n_examples,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
