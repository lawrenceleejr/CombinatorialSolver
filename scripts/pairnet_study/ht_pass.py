"""HT (scalar pT sum of all stored jets: pT > 30 GeV, |eta| < 3) per event with >= 6 jets, in big_vars.py file order."""
import glob, sys
from multiprocessing import Pool
import h5py
import numpy as np


def ht(fp):
    with h5py.File(fp, "r") as h:
        pt = h["INPUTS/Source/pt"][:]; m = h["INPUTS/Source/MASK"][:].astype(bool)
    return ((pt * m).sum(1)[m.sum(1) >= 6]).astype(np.float32)


if __name__ == "__main__":
    P = "/scratch/llee/pairnet/data_r3/prod"
    files = sorted(glob.glob(f"{P}/sig_*_c*9.h5")) + sorted(glob.glob(f"{P}/qcd_c*.h5"))
    with Pool(8) as p:
        out = p.map(ht, files, chunksize=4)
    np.save(sys.argv[1], np.concatenate(out))
    print(sum(len(x) for x in out))
