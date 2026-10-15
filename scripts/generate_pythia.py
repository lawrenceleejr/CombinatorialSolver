"""
Generate a realistic jet-level testbed with Pythia8 + FastJet.

Two processes:
  signal : p p > gluino gluino, gluino -> u d s (RPV UDD), at a chosen mass.
           Full ISR/FSR/MPI/hadronisation, so jets are showered and the
           event usually has extra radiation jets.
  qcd    : HardQCD:all 2->2 with parton shower (multijet background), with a
           pTHat bias so the high-HT tail is populated (weights stored).

Jets: anti-kt R=0.4 on final-state visible hadrons, pT > ptmin, |eta| < etamax,
with a simple calorimeter-like pT resolution applied afterwards.

Output follows the MadGraphMLProducer / SPANet layout used by the rest of
this repo (INPUTS/Source/*, TARGETS/g1|g2/j1..3), so files are drop-in:
  INPUTS/Source/{pt,eta,phi,mass,btag,MASK}   (N, max_jets), pT-sorted
  TARGETS/g{1,2}/j{1,2,3}                     index into the jet list, -1 if
                                              the quark has no matched jet
  EventVars/normweight                        generator weight
  EventVars/parent_mass                       true gluino masses (mean of two)
  truth/quarks_{pt,eta,phi}                   (N, 6) decay quarks (g1 then g2)

Usage:
  python scripts/generate_pythia.py --process signal --mass 1000 --n 20000 \
      --seed 1 --out data/pythia/sig_m1000_s1.h5
  python scripts/generate_pythia.py --process qcd --n 200000 --seed 1 \
      --out data/pythia/qcd_s1.h5
"""

import argparse
import os
import tempfile
import time

import fastjet
import h5py
import numpy as np
import pythia8mc as pythia8

GLUINO = 1000021


def write_slha(mass: float, width: float, path: str) -> None:
    """Minimal SLHA: decouple everything except the gluino, force g~ -> u d s."""
    heavy = 5.0e4
    sparticles = [
        1000001, 1000002, 1000003, 1000004, 1000005, 1000006,
        2000001, 2000002, 2000003, 2000004, 2000005, 2000006,
        1000011, 1000012, 1000013, 1000014, 1000015, 1000016,
        2000011, 2000013, 2000015,
        1000022, 1000023, 1000025, 1000035, 1000024, 1000037,
    ]
    lines = [
        "BLOCK MODSEL",
        "    1   0",
        "    4   1   # RPV on",
        "BLOCK SMINPUTS",
        "    1   1.27934e+02",
        "    2   1.16637e-05",
        "    3   1.18000e-01",
        "    4   9.11876e+01",
        "    5   4.20000e+00",
        "    6   1.73000e+02",
        "    7   1.77700e+00",
        "BLOCK MASS",
        f"  {GLUINO}  {mass:.6e}",
    ]
    lines += [f"  {pid}  {heavy:.6e}" for pid in sparticles]
    lines += [
        f"DECAY {GLUINO} {width:.6e}",
        "   5.000000e-01   3    2    1    3",
        "   5.000000e-01   3   -2   -1   -3",
    ]
    for pid in sparticles:
        lines.append(f"DECAY {pid} 1.0")
        lines.append("   1.000000e+00   2    1000021    1")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def make_pythia(args) -> pythia8.Pythia:
    p = pythia8.Pythia("", False)
    cmds = [
        "Beams:eCM = 13600.",
        f"Random:setSeed = on",
        f"Random:seed = {args.seed}",
        "Next:numberCount = 0",
        "Init:showChangedSettings = off",
        "Init:showChangedParticleData = off",
        "Init:showProcesses = off",
        "Print:quiet = on",
    ]
    if args.process == "signal":
        slha = os.path.join(tempfile.mkdtemp(), "gluino.slha")
        write_slha(args.mass, args.mass * args.width_frac, slha)
        cmds += [
            "SUSY:all = off",
            "SUSY:gg2gluinogluino = on",
            "SUSY:qqbar2gluinogluino = on",
            f"SLHA:file = {slha}",
            "SLHA:useDecayTable = on",
            "SLHA:allowUserOverride = on",
            f"{GLUINO}:mayDecay = on",
        ]
    else:
        cmds += [
            "HardQCD:all = on",
            f"PhaseSpace:pTHatMin = {args.pthat_min}",
            "PhaseSpace:bias2Selection = on",
            f"PhaseSpace:bias2SelectionPow = {args.bias_pow}",
            f"PhaseSpace:bias2SelectionRef = {args.pthat_min}",
        ]
    if args.no_mpi:
        cmds.append("PartonLevel:MPI = off")
    if args.no_had:
        cmds.append("HadronLevel:all = off")
    for c in cmds:
        if not p.readString(c):
            raise RuntimeError(f"Pythia rejected: {c}")
    if not p.init():
        raise RuntimeError("Pythia init failed")
    return p


def decay_quarks(event) -> list[list[tuple[float, float, float]]] | None:
    """Return [[(pt,eta,phi) x3] for each gluino] from the last gluino copies."""
    groups = []
    for i in range(event.size()):
        prt = event[i]
        if abs(prt.id()) != GLUINO:
            continue
        # last copy: the one whose daughters are not gluinos
        d1, d2 = prt.daughter1(), prt.daughter2()
        if d1 <= 0:
            continue
        dl = list(range(d1, max(d1, d2) + 1))
        if any(abs(event[d].id()) == GLUINO for d in dl):
            continue
        qs = [event[d] for d in dl if 1 <= abs(event[d].id()) <= 6]
        if len(qs) != 3:
            return None
        groups.append([(q.pT(), q.eta(), q.phi()) for q in qs])
    if len(groups) != 2:
        return None
    return groups


def smear_pt(pt: np.ndarray, rng) -> np.ndarray:
    """Calorimeter-like resolution: noise (+) stochastic (+) constant."""
    rel = np.sqrt((5.0 / pt) ** 2 + (0.6 / np.sqrt(pt)) ** 2 + 0.03**2)
    return pt * np.clip(1.0 + rel * rng.standard_normal(pt.shape), 0.3, 3.0)


def dphi(a, b):
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--process", choices=["signal", "qcd"], required=True)
    ap.add_argument("--mass", type=float, default=1000.0)
    ap.add_argument("--width-frac", type=float, default=0.01)
    ap.add_argument("--n", type=int, default=10000, help="events to WRITE")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ptmin", type=float, default=30.0)
    ap.add_argument("--etamax", type=float, default=2.5)
    ap.add_argument("--min-jets", type=int, default=6)
    ap.add_argument("--max-jets", type=int, default=20)
    ap.add_argument("--min-ht", type=float, default=0.0)
    ap.add_argument("--pthat-min", type=float, default=200.0)
    ap.add_argument("--bias-pow", type=float, default=3.0)
    ap.add_argument("--no-mpi", action="store_true")
    ap.add_argument("--no-had", action="store_true")
    ap.add_argument("--match-dr", type=float, default=0.4)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed + 7919)
    p = make_pythia(args)
    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)

    MJ = args.max_jets
    out = {k: [] for k in ["pt", "eta", "phi", "mass", "w", "pm", "tg", "qk", "ntot"]}
    n_tried = 0
    t0 = time.time()
    while len(out["pt"]) < args.n:
        n_tried += 1
        if not p.next():
            continue
        ev = p.event
        groups = None
        if args.process == "signal":
            groups = decay_quarks(ev)
            if groups is None:
                continue
        parts = []
        for i in range(ev.size()):
            prt = ev[i]
            if not prt.isFinal() or not prt.isVisible():
                continue
            if abs(prt.eta()) > 4.9:
                continue
            parts.append(fastjet.PseudoJet(prt.px(), prt.py(), prt.pz(), prt.e()))
        cs = fastjet.ClusterSequence(parts, jet_def)
        jets = cs.inclusive_jets(0.8 * args.ptmin)
        if len(jets) < args.min_jets:
            continue
        tpt = np.array([j.pt() for j in jets])
        teta = np.array([j.eta() for j in jets])
        tphi = np.array([j.phi_std() for j in jets])
        tm = np.array([max(j.m(), 0.0) for j in jets])
        spt = smear_pt(tpt, rng)
        seta = teta + 0.01 * rng.standard_normal(teta.shape)
        sphi = dphi(tphi + 0.01 * rng.standard_normal(tphi.shape), 0.0)
        sm = tm * spt / tpt
        sel = (spt > args.ptmin) & (np.abs(seta) < args.etamax)
        if sel.sum() < args.min_jets:
            continue
        order = np.argsort(-spt[sel])
        idx = np.where(sel)[0][order][:MJ]
        if spt[idx].sum() < args.min_ht:
            continue
        nj = len(idx)

        def pad(a):
            z = np.zeros(MJ, dtype=np.float32)
            z[:nj] = a[idx]
            return z

        out["pt"].append(pad(spt))
        out["eta"].append(pad(seta))
        out["phi"].append(pad(sphi))
        out["mass"].append(pad(sm))
        out["ntot"].append(nj)
        out["w"].append(p.info.weight())

        tg = np.full(6, -1, dtype=np.int32)
        qk = np.zeros((6, 3), dtype=np.float32)
        if groups is not None:
            # Unique greedy DeltaR matching of the 6 decay quarks to the
            # selected (true-level direction) jets.
            q = np.array(groups[0] + groups[1])  # (6,3)
            qk[:] = q
            jeta, jphi = teta[idx], tphi[idx]
            dr = np.sqrt((q[:, 1:2] - jeta[None]) ** 2 + dphi(q[:, 2:3], jphi[None]) ** 2)
            used_q, used_j = set(), set()
            for flat in np.argsort(dr, axis=None):
                qi, ji = divmod(int(flat), nj)
                if dr[qi, ji] > args.match_dr:
                    break
                if qi in used_q or ji in used_j:
                    continue
                tg[qi] = ji
                used_q.add(qi)
                used_j.add(ji)
            out["pm"].append(args.mass)
        else:
            out["pm"].append(0.0)
        out["tg"].append(tg)
        out["qk"].append(qk)

        if len(out["pt"]) % 1000 == 0:
            el = time.time() - t0
            print(f"  {len(out['pt'])}/{args.n} written, {n_tried} tried, {el:.0f}s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pt = np.stack(out["pt"])
    with h5py.File(args.out, "w") as f:
        g = f.create_group("INPUTS/Source")
        g["pt"] = pt
        g["eta"] = np.stack(out["eta"])
        g["phi"] = np.stack(out["phi"])
        g["mass"] = np.stack(out["mass"])
        g["btag"] = np.zeros_like(pt)
        g["MASK"] = pt > 0
        tg = np.stack(out["tg"])
        if args.process == "signal":
            for gi in range(2):
                for ji in range(3):
                    f[f"TARGETS/g{gi + 1}/j{ji + 1}"] = tg[:, 3 * gi + ji]
        f["truth/jet_index"] = tg
        f["truth/quarks"] = np.stack(out["qk"])
        f["EventVars/normweight"] = np.array(out["w"], dtype=np.float64)
        f["EventVars/parent_mass"] = np.array(out["pm"], dtype=np.float32)
        f["EventVars/n_jets"] = np.array(out["ntot"], dtype=np.int32)
        f.attrs["process"] = args.process
        f.attrs["mass"] = args.mass
        f.attrs["n_tried"] = n_tried
        f.attrs["sigma_mb"] = p.info.sigmaGen()
        f.attrs["generator"] = "pythia8 + fastjet anti-kt R=0.4, smeared"
    print(f"done: {len(pt)} written / {n_tried} tried in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
