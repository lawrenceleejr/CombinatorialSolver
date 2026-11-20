"""Append gen_f2 per-event variables (pthat, sym_pass) to the unfiltered score_lowmass npz. usage: f2_post.py in.h5 in.npz"""
import sys, json
import h5py, numpy as np
h5, npz = sys.argv[1], sys.argv[2]
z = dict(np.load(npz))
with h5py.File(h5, "r") as h:
    mask = h["INPUTS/Source/MASK"][:].astype(bool)
    keep6 = mask.sum(1) >= 6
    z["pthat"] = h["EventVars/pthat"][:][keep6]
    z["sym_pass"] = h["EventVars/sym_pass"][:][keep6]
    z["f2_args"] = np.array(h.attrs["f2_args"])
assert len(z["pthat"]) == len(z["w"]), (len(z["pthat"]), len(z["w"]))
np.savez_compressed(npz, **z)
print("f2_post ok", len(z["w"]))
