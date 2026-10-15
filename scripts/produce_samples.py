"""
Mass-produce signal and QCD samples with scripts/generate_pythia.py.

Every output file is one "chunk" of a fixed size with its own seed, so a
production is reproducible, resumable (finished chunks are skipped) and easy to
split across machines:

  # run locally on all cores
  python scripts/produce_samples.py --outdir data/prod --masses 400:2400:100 \
      --sig-events 50000 --qcd-events 300000 --cores 32

  # or just write one command per line for Condor / Slurm / GNU parallel
  python scripts/produce_samples.py ... --joblist jobs.txt
  parallel -j 64 < jobs.txt

File names encode the split, so training and test never share events:
  sig_m{M}_c{i}.h5, qcd_c{i}.h5 ; chunk i -> split by i % 10:
  0-7 train, 8 val, 9 test  (see split_of() and docs/PRODUCTION.md)

Throughput measured on one core (Pythia + hadronisation + FastJet):
  signal  ~55 events/s                 ->  ~5 core-hours per million events
  QCD     ~9 events/s kept (>= 5 jets) -> ~30 core-hours per million kept events
          (about 40% of kept QCD events have >= 6 jets, i.e. are in the SR)
"""

import argparse
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, "generate_pythia.py")


def parse_masses(spec: str) -> list[int]:
    """'400:2400:100' -> range (inclusive), or '500,1000,1500' -> list."""
    if ":" in spec:
        lo, hi, step = (int(x) for x in spec.split(":"))
        return list(range(lo, hi + 1, step))
    return [int(x) for x in spec.split(",") if x]


def split_of(chunk: int) -> str:
    r = chunk % 10
    return "train" if r < 8 else ("val" if r == 8 else "test")


def seed_for(kind: str, mass: int, chunk: int, offset: int) -> int:
    # unique, deterministic, < 900,000,000 (Pythia's limit)
    base = 1 if kind == "sig" else 2
    return (offset + base * 100_000_000 + mass * 10_000 + chunk) % 900_000_000


def build_jobs(a) -> list[tuple[str, list[str]]]:
    jobs = []
    py = sys.executable
    common = ["--etamax", str(a.etamax), "--ptmin", str(a.ptmin)]
    for m in parse_masses(a.masses):
        n_chunks = -(-a.sig_events // a.chunk)
        for c in range(n_chunks):
            out = os.path.join(a.outdir, f"sig_m{m}_c{c}.h5")
            jobs.append((out, [py, GEN, "--process", "signal", "--mass", str(m),
                               "--width-frac", str(a.width_frac),
                               "--n", str(a.chunk), "--seed", str(seed_for("sig", m, c, a.seed_offset)),
                               "--min-jets", str(a.min_jets), "--out", out] + common))
    n_chunks = -(-a.qcd_events // a.chunk)
    for c in range(n_chunks):
        out = os.path.join(a.outdir, f"qcd_c{c}.h5")
        jobs.append((out, [py, GEN, "--process", "qcd", "--n", str(a.chunk),
                           "--seed", str(seed_for("qcd", 0, c, a.seed_offset)),
                           "--min-jets", str(a.qcd_min_jets), "--pthat-min", str(a.pthat_min),
                           "--bias-pow", str(a.bias_pow), "--out", out] + common))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--masses", default="400:2400:100", help="lo:hi:step or comma list (GeV)")
    ap.add_argument("--sig-events", type=int, default=50000, help="events written per mass point")
    ap.add_argument("--qcd-events", type=int, default=0, help="QCD events written in total")
    ap.add_argument("--chunk", type=int, default=10000, help="events per output file")
    ap.add_argument("--cores", type=int, default=os.cpu_count())
    ap.add_argument("--seed-offset", type=int, default=0, help="change to make an independent production")
    ap.add_argument("--width-frac", type=float, default=0.01)
    ap.add_argument("--etamax", type=float, default=3.0)
    ap.add_argument("--ptmin", type=float, default=30.0)
    ap.add_argument("--min-jets", type=int, default=6, help="signal: minimum jets kept")
    ap.add_argument("--qcd-min-jets", type=int, default=5, help="QCD: keep 5-jet events for the CR")
    ap.add_argument("--pthat-min", type=float, default=200.0)
    ap.add_argument("--bias-pow", type=float, default=2.0)
    ap.add_argument("--joblist", default=None, help="write commands to this file instead of running")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    jobs = build_jobs(a)
    todo = [(o, cmd) for o, cmd in jobs if not os.path.exists(o)]
    print(f"{len(jobs)} chunks, {len(jobs) - len(todo)} already done, {len(todo)} to run")

    if a.joblist:
        with open(a.joblist, "w") as f:
            for o, cmd in todo:
                f.write(" ".join(shlex.quote(x) for x in cmd) + f" > {shlex.quote(o)}.log 2>&1\n")
        print(f"wrote {len(todo)} commands to {a.joblist}")
        return

    t0 = time.time()
    done = [0]

    def run(job):
        out, cmd = job
        with open(out + ".log", "w") as log:
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT,
                                 env={**os.environ, "OMP_NUM_THREADS": "1"})
        done[0] += 1
        el = time.time() - t0
        eta = el / done[0] * (len(todo) - done[0])
        print(f"[{done[0]}/{len(todo)}] {'ok ' if rc == 0 else 'FAIL'} {os.path.basename(out)}  "
              f"elapsed {el / 60:.0f} min, eta {eta / 60:.0f} min", flush=True)
        return rc

    with ThreadPoolExecutor(a.cores) as ex:
        rcs = list(ex.map(run, todo))
    bad = sum(rc != 0 for rc in rcs)
    print(f"finished: {len(todo) - bad} ok, {bad} failed (see *.log next to each file)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
