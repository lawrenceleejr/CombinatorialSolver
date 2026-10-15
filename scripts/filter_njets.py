"""Write a copy of an h5 testbed file keeping only events with >= N stored jets.

The legacy loader has no jet-multiplicity selection; this makes its inputs match
the signal region used everywhere else.
  python scripts/filter_njets.py in.h5 out.h5 --min-jets 6
"""
import argparse

import h5py
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--min-jets", type=int, default=6)
a = ap.parse_args()
with h5py.File(a.src) as f, h5py.File(a.dst, "w") as g:
    keep = f["INPUTS/Source/MASK"][:].sum(1) >= a.min_jets

    def copy(name, obj):
        if isinstance(obj, h5py.Dataset):
            g[name] = obj[:][keep] if obj.shape and obj.shape[0] == len(keep) else obj[()]

    f.visititems(copy)
    for k, v in f.attrs.items():
        g.attrs[k] = v
print(f"{a.dst}: kept {keep.sum()}/{len(keep)}")
