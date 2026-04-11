"""
Diagnostic script: inspect an HDF5 file's structure and check compatibility
with the JetAssignmentDataset.

Usage:
    python scripts/inspect_h5.py path/to/file.h5
"""

import argparse
import sys

import h5py
import numpy as np


def inspect_h5(path: str):
    print(f"=== Inspecting: {path} ===\n")

    with h5py.File(path, "r") as f:
        print("Top-level keys:", list(f.keys()))
        print()

        for key in f.keys():
            ds = f[key]
            print(f"  {key}:")
            print(f"    shape: {ds.shape}")
            print(f"    dtype: {ds.dtype}")

            if len(ds.shape) > 0 and ds.shape[0] > 0:
                # Show first event's data
                sample = ds[0]
                if len(ds.shape) == 1:
                    print(f"    first event: {sample}")
                elif len(ds.shape) == 2:
                    print(f"    first event (first 10 cols): {sample[:10]}")
                elif len(ds.shape) == 3:
                    print(f"    first event, first row: {sample[0]}")
                    # Count non-zero rows
                    if ds.dtype in [np.float32, np.float64, np.int32, np.int64]:
                        nonzero = np.any(sample != 0, axis=-1).sum()
                        print(f"    first event, non-zero rows: {nonzero}/{sample.shape[0]}")
            print()

        # Check for expected datasets
        print("=== Compatibility Check ===\n")

        expected = {
            "jet_features": "Expected shape (N, max_jets, num_features)",
            "jet_mask": "Expected shape (N, max_jets)",
            "event_features": "Expected shape (N, num_event_features)",
        }

        for name, desc in expected.items():
            if name in f:
                print(f"  [OK] {name}: {f[name].shape} — {desc}")
            else:
                print(f"  [MISSING] {name} — {desc}")

        print()

        # If jet_features exists, inspect its columns
        if "jet_features" in f:
            jf = f["jet_features"]
            n_events = jf.shape[0]
            max_jets = jf.shape[1] if len(jf.shape) > 1 else 0
            n_feat = jf.shape[2] if len(jf.shape) > 2 else 0

            print(f"  jet_features: {n_events} events, {max_jets} max jets, {n_feat} features/jet")

            if n_events > 0 and n_feat > 0:
                # Show a few events' leading jets
                sample = jf[:min(3, n_events), :min(10, max_jets), :]
                for i in range(sample.shape[0]):
                    print(f"\n  Event {i}, leading jets:")
                    for j in range(sample.shape[1]):
                        row = sample[i, j]
                        if np.any(row != 0):
                            cols = ", ".join(f"{v:.3f}" for v in row)
                            print(f"    jet {j}: [{cols}]")

                # Check column ranges to guess meaning
                if n_feat >= 7:
                    print(f"\n  Column analysis (over first 100 events, leading 7 jets):")
                    chunk = jf[:min(100, n_events), :min(7, max_jets), :]
                    for c in range(n_feat):
                        vals = chunk[:, :, c].flatten()
                        vals = vals[vals != 0]  # ignore padding
                        if len(vals) > 0:
                            print(f"    col {c}: min={vals.min():.3f}, max={vals.max():.3f}, "
                                  f"mean={vals.mean():.3f}, unique_count={len(np.unique(vals))}")
                        else:
                            print(f"    col {c}: all zeros")

                # Check is_signal column (expected index 6)
                if n_feat >= 7:
                    is_signal_col = jf[:min(100, n_events), :min(7, max_jets), 6]
                    unique_vals = np.unique(is_signal_col)
                    print(f"\n  Column 6 (expected: is_signal) unique values: {unique_vals}")

                    # Check how many events have exactly 6 signal jets in leading 7
                    is_sig_7 = jf[:, :min(7, max_jets), 6]
                    n_signal_per_event = (is_sig_7 == 1).sum(axis=1)
                    print(f"\n  Signal jet counts in leading 7 jets:")
                    for count in sorted(np.unique(n_signal_per_event)):
                        n = (n_signal_per_event == count).sum()
                        print(f"    {int(count)} signal jets: {n} events ({100*n/n_events:.1f}%)")

                # Check parent_pdg column (expected index 5)
                if n_feat >= 6:
                    parent_col = jf[:min(100, n_events), :min(7, max_jets), 5]
                    unique_parents = np.unique(parent_col)
                    print(f"\n  Column 5 (expected: parent_pdg) unique values: {unique_parents[:20]}")

        # Check event_features columns
        if "event_features" in f:
            ef = f["event_features"]
            if len(ef.shape) >= 2 and ef.shape[0] > 0:
                print(f"\n  event_features: {ef.shape[0]} events, {ef.shape[1]} features")
                print(f"  First event: {ef[0]}")
                if ef.shape[1] >= 5:
                    ht_col = ef[:, 4]
                    print(f"  Column 4 (expected: HT): min={ht_col.min():.1f}, max={ht_col.max():.1f}, mean={ht_col.mean():.1f}")

        # Check for alternative key names
        print(f"\n  All available keys: {list(f.keys())}")
        for key in f.keys():
            if key not in expected:
                ds = f[key]
                print(f"    Extra key '{key}': shape={ds.shape}, dtype={ds.dtype}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect HDF5 file structure")
    parser.add_argument("path", help="Path to HDF5 file")
    args = parser.parse_args()
    inspect_h5(args.path)
