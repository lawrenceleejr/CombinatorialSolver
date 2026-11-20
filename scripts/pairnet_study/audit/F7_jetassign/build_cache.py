import sys, argparse
sys.path.insert(0, "/scratch/llee/pairnet/audit/F7_jetassign/wt")
sys.argv = ["x"] + open("/scratch/llee/pairnet/audit/F7_jetassign/args_base.txt").read().split() + ["--out", "x"]
import scripts.train_jetassign as T
ap = argparse.ArgumentParser()
# reuse the training script's parser by running main up to the cache step
import types
src = open(T.__file__).read()
a_src = src[src.index("    ap = argparse.ArgumentParser()"):src.index("    a = ap.parse_args()")]
ns = {"argparse": argparse}
exec("def _p():\n" + a_src + "    return ap\n", ns)
a = ns["_p"]().parse_args(sys.argv[1:])
for w in ("train", "val", "test"):
    d = T.cached(a, w)
    print(w, len(d["pt"]), int(d["is_signal"].sum()), flush=True)
