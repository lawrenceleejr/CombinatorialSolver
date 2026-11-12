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
  truth/quarks                                (N, 6, 3) (pt, eta, phi) of the decay
                                              quarks (g1 then g2) used for matching
  truth/quarks_before_fsr, truth/quarks_after_fsr
                                              both definitions (--truth-quarks): the
                                              quark after recoil but before its own
                                              shower (Pythia iBotCopy), or traced
                                              through q -> q g to the end of its
                                              shower (iBotCopyId)

Usage:
  python scripts/generate_pythia.py --process signal --mass 1000 --n 20000 \
      --seed 1 --out data/pythia/sig_m1000_s1.h5
  python scripts/generate_pythia.py --process qcd --n 200000 --seed 1 \
      --out data/pythia/qcd_s1.h5
"""

import argparse
import itertools
import os
import tempfile
import time

import fastjet
import h5py
import numpy as np
try:  # PyPI build
    import pythia8mc as pythia8
except ImportError:  # conda-forge / source build
    import pythia8

GLUINO = 1000021


SQUARK_UR = 2000002
SQUARKS = set(range(1000001, 1000007)) | set(range(2000001, 2000007))


def write_slha(mass: float, width: float, path: str, decay: str = "direct",
               squark_mass: float | None = None) -> None:
    """Minimal SLHA: decouple everything except the gluino, force g~ -> u d s.

    decay="chain": g~ -> ~u_R u~ (+ c.c.) with an on-shell ~u_R of squark_mass,
    ~u_R -> d~ s~ (RPV UDD), so each gluino still gives three quarks but two of
    them form a resonance."""
    heavy = 5.0e4
    sparticles = [
        1000001, 1000002, 1000003, 1000004, 1000005, 1000006,
        2000001, 2000002, 2000003, 2000004, 2000005, 2000006,
        1000011, 1000012, 1000013, 1000014, 1000015, 1000016,
        2000011, 2000013, 2000015,
        1000022, 1000023, 1000025, 1000035, 1000024, 1000037,
    ]
    ident = lambda name, n: [f"BLOCK {name}"] + [
        f"  {i} {j}  {1.0 if i == j else 0.0:.6e}" for i in range(1, n + 1) for j in range(1, n + 1)
    ]
    lines = [
        "BLOCK MODSEL",
        "    1   1",
        "BLOCK SMINPUTS",
        "    1   1.27934e+02",
        "    2   1.16637e-05",
        "    3   1.18000e-01",
        "    4   9.11876e+01",
        "    5   4.20000e+00",
        "    6   1.73000e+02",
        "    7   1.77700e+00",
        "BLOCK MINPAR",
        "    1   1.0e+04",
        "    2   1.0e+04",
        "    3   1.0e+01",
        "    4   1.0e+00",
        "    5   0.0e+00",
        "BLOCK GAUGE Q= 1.0e+03",
        "    1   3.6e-01",
        "    2   6.4e-01",
        "    3   1.1e+00",
        "BLOCK HMIX Q= 1.0e+03",
        "    1   5.0e+04",
        "    2   1.0e+01",
        "    3   2.4e+02",
        "    4   2.5e+09",
        "BLOCK ALPHA",
        "   -1.0e-01",
    ]
    lines += ident("NMIX", 4) + ident("UMIX", 2) + ident("VMIX", 2)
    lines += ident("STOPMIX", 2) + ident("SBOTMIX", 2) + ident("STAUMIX", 2)
    lines += [
        "BLOCK MASS",
        f"  {GLUINO}  {mass:.6e}",
        "  25  1.25e+02",
        "  35  5.0e+04",
        "  36  5.0e+04",
        "  37  5.0e+04",
    ]
    chain = decay == "chain"
    lines += [f"  {pid}  {(squark_mass if chain and pid == SQUARK_UR else heavy):.6e}" for pid in sparticles]
    if chain:
        lines += [
            f"DECAY {GLUINO} {width:.6e}",
            f"   5.000000e-01   2    {SQUARK_UR}   -2",
            f"   5.000000e-01   2   -{SQUARK_UR}    2",
            f"DECAY {SQUARK_UR} {0.01 * squark_mass:.6e}",
            "   1.000000e+00   2   -1   -3",
        ]
    else:
        lines += [
            f"DECAY {GLUINO} {width:.6e}",
            "   5.000000e-01   3    2    1    3",
            "   5.000000e-01   3   -2   -1   -3",
        ]
    for pid in sparticles:
        if chain and pid == SQUARK_UR:
            continue
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
        write_slha(args.mass, args.mass * args.width_frac, slha, args.decay, args.squark_mass)
        cmds += [
            "SUSY:all = off",
            "SUSY:gg2gluinogluino = on",
            "SUSY:qqbar2gluinogluino = on",
            f"SLHA:file = {slha}",
            "SLHA:useDecayTable = on",
            "SLHA:allowUserOverride = on",
            f"{GLUINO}:mayDecay = on",
        ]
        if args.decay == "chain":
            cmds.append(f"{SQUARK_UR}:mayDecay = on")
    else:
        cmds += [
            "HardQCD:all = on",
            f"PhaseSpace:pTHatMin = {args.pthat_min}",
            "PhaseSpace:bias2Selection = on",
            f"PhaseSpace:bias2SelectionPow = {args.bias_pow}",
            f"PhaseSpace:bias2SelectionRef = {args.pthat_min}",
        ]
    if args.alphas_fsr is not None:
        cmds.append(f"TimeShower:alphaSvalue = {args.alphas_fsr}")
    if args.alphas_isr is not None:
        cmds.append(f"SpaceShower:alphaSvalue = {args.alphas_isr}")
    if args.power_shower:
        cmds.append("SpaceShower:pTmaxMatch = 2")  # ISR up to the kinematic limit: hard-ISR proxy
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


_SPLITS = [(t, tuple(sorted(set(six) - set(t))))
           for six in itertools.combinations(range(7), 6)
           for t in itertools.combinations(six, 3) if t[0] == six[0]]  # 7 x 10 = 70 splits


def has_symmetric_split(pt, eta, phi, m, lo, hi, amax=0.1):
    """True if some split of the leading (up to 7) jets into two triplets has A < amax and lo < m_avg < hi."""
    n = min(len(pt), 7)
    px, py, pz = pt[:n] * np.cos(phi[:n]), pt[:n] * np.sin(phi[:n]), pt[:n] * np.sinh(eta[:n])
    e = np.sqrt(px ** 2 + py ** 2 + pz ** 2 + m[:n] ** 2)
    p4 = np.stack([e, px, py, pz], 1)
    for a, b in _SPLITS:
        if max(a + b) >= n:
            continue
        s1, s2 = p4[list(a)].sum(0), p4[list(b)].sum(0)
        m1 = np.sqrt(max(s1[0] ** 2 - s1[1] ** 2 - s1[2] ** 2 - s1[3] ** 2, 0.0))
        m2 = np.sqrt(max(s2[0] ** 2 - s2[1] ** 2 - s2[2] ** 2 - s2[3] ** 2, 0.0))
        if m1 + m2 > 0 and abs(m1 - m2) / (m1 + m2) < amax and lo < 0.5 * (m1 + m2) < hi:
            return True
    return False


def _decay_quarks_from(event, idx, after_fsr) -> list[int]:
    """Decay quarks reached from entry idx, following squarks: the last carbon
    copy (after recoil, before the quark's own FSR) or, with after_fsr, the last
    same-id copy (traced through q -> q g to the end of the quark's shower)."""
    prt = event[idx]
    a = abs(prt.id())
    if 1 <= a <= 6:
        return [prt.iBotCopyId() if after_fsr else prt.iBotCopy()]
    if a in SQUARKS:
        bot = event[prt.iBotCopyId()]  # the squark copy that decays
        d1, d2 = bot.daughter1(), bot.daughter2()
        out = []
        for d in range(d1, max(d1, d2) + 1):
            out += _decay_quarks_from(event, d, after_fsr)
        return out
    return []


def decay_quarks(event, after_fsr=False) -> list[list[tuple[float, float, float]]] | None:
    """Return [[(pt,eta,phi) x3] for each gluino] from the last gluino copies
    (directly, or through an intermediate squark)."""
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
        qi = [q for d in dl for q in _decay_quarks_from(event, d, after_fsr)]
        if len(qi) != 3:
            return None
        groups.append([(event[q].pT(), event[q].eta(), event[q].phi()) for q in qi])
    if len(groups) != 2:
        return None
    return groups


def last_gluinos(event) -> list[int]:
    """Indices of the decaying gluino copies (those whose daughters are not gluinos), in event-record order
    (the same order as decay_quarks)."""
    out = []
    for i in range(event.size()):
        prt = event[i]
        if abs(prt.id()) != GLUINO:
            continue
        d1, d2 = prt.daughter1(), prt.daughter2()
        if d1 <= 0 or any(abs(event[d].id()) == GLUINO for d in range(d1, max(d1, d2) + 1)):
            continue
        out.append(i)
    return out


def ancestry_label(event, i, gl, memo) -> int:
    """0 / 1: entry i descends from the decay of gluino gl[0] / gl[1]; 2: anything else (ISR, MPI, beam remnants,
    radiation off the gluino before it decays). Follows mothers upwards; where an entry has several mothers
    (colour-reconnection / junction bookkeeping) it takes the label of the most energetic one."""
    path = []
    while True:
        if i in memo:
            lab = memo[i]
            break
        path.append(i)
        if i == gl[0] or i == gl[1]:
            lab = 0 if i == gl[0] else 1
            break
        if i <= 2:
            lab = 2
            break
        prt = event[i]
        m1, m2 = prt.mother1(), prt.mother2()
        if m1 <= 0:
            lab = 2
            break
        if m2 == 0 or m2 == m1:
            i = m1
            continue
        ms = [m for m in prt.motherList() if m > 0]
        labs = [ancestry_label(event, m, gl, memo) for m in ms]
        lab = labs[int(np.argmax([event[m].e() for m in ms]))] if len(set(labs)) > 1 else labs[0]
        break
    for j in path:
        memo[j] = lab
    return lab


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
    ap.add_argument("--decay", choices=["direct", "chain"], default="direct",
                    help="direct: g~ -> uds (3-body); chain: g~ -> ~u_R u~, ~u_R -> d~ s~")
    ap.add_argument("--squark-mass", type=float, default=None, help="for --decay chain")
    ap.add_argument("--alphas-fsr", type=float, default=None, help="TimeShower:alphaSvalue (Monash 0.1365)")
    ap.add_argument("--alphas-isr", type=float, default=None, help="SpaceShower:alphaSvalue (Monash 0.1365)")
    ap.add_argument("--n", type=int, default=10000, help="events to WRITE")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ptmin", type=float, default=30.0)
    ap.add_argument("--etamax", type=float, default=3.0)  # CMS Run 3 scouting jets: |eta|<3
    ap.add_argument("--min-jets", type=int, default=6)
    ap.add_argument("--max-jets", type=int, default=20)
    ap.add_argument("--min-ht", type=float, default=0.0)
    ap.add_argument("--sym-filter", default=None,
                    help="LO,HI: keep only events where some split of the leading 7 jets into two triplets has "
                         "|m1-m2|/(m1+m2) < 0.1 and LO < (m1+m2)/2 < HI (a superset of any A < 0.1 window selection)")
    ap.add_argument("--power-shower", action="store_true", help="SpaceShower:pTmaxMatch = 2 (hard-ISR proxy)")
    ap.add_argument("--pthat-min", type=float, default=200.0)
    ap.add_argument("--bias-pow", type=float, default=3.0)
    ap.add_argument("--no-mpi", action="store_true")
    ap.add_argument("--no-had", action="store_true")
    ap.add_argument("--match-dr", type=float, default=0.4)
    ap.add_argument("--truth-quarks", choices=["before_fsr", "after_fsr"], default="after_fsr",
                    help="decay-quark momenta used for jet matching (both are stored)")
    ap.add_argument("--ancestry", action="store_true",
                    help="signal only: store, per jet, the momentum of the final partons (before hadronisation, "
                         "ghost-associated to the jets) descending from gluino 1 / gluino 2 / neither, for all jets "
                         "down to --anc-ptmin and |eta| < 4.9 (ANC/*), the fractions for the stored jets "
                         "(INPUTS/Source/anc), and the gluino and descendant four-momenta (truth/gluino_p4, desc_p4)")
    ap.add_argument("--anc-ptmin", type=float, default=10.0, help="true jet pT threshold of the ANC/* jet list")
    ap.add_argument("--anc-max-jets", type=int, default=40)
    ap.add_argument("--anc-keep-all", action="store_true",
                    help="with --ancestry: also write the events failing the jet selection (EventVars/pass_sel = 0), "
                         "so that alternative jet definitions can be studied relative to all produced events")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed + 7919)
    p = make_pythia(args)
    jet_def = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)

    MJ = args.max_jets
    sym_filter = tuple(float(x) for x in args.sym_filter.split(",")) if args.sym_filter else None
    out = {k: [] for k in ["pt", "eta", "phi", "mass", "w", "pm", "tg", "qk", "qb", "qa", "ntot"]}
    anc = args.ancestry and args.process == "signal"
    AJ = args.anc_max_jets
    aout = {k: [] for k in ["frac", "gp4", "dp4", "tj", "sj", "lp4", "lpt", "std", "nall", "pass"]}
    GHOST = 1e-18
    n_tried = 0
    t0 = time.time()
    while len(out["pt"]) < args.n:
        n_tried += 1
        if not p.next():
            continue
        ev = p.event
        groups = None
        if args.process == "signal":
            alt = {f: decay_quarks(ev, after_fsr=f) for f in (False, True)}
            if alt[False] is None or alt[True] is None:
                continue
            groups = alt[args.truth_quarks == "after_fsr"]
        parts = []
        plab, pp4 = [], []  # final partons before hadronisation: ancestry label, (E, px, py, pz)
        if anc:
            gl = last_gluinos(ev)
            if len(gl) != 2:
                continue
            memo = {}
        for i in range(ev.size()):
            prt = ev[i]
            if anc and prt.isFinalPartonLevel() and (prt.isParton() or prt.isDiquark()):
                plab.append(ancestry_label(ev, i, gl, memo))
                pp4.append((prt.e(), prt.px(), prt.py(), prt.pz()))
                if prt.pT() > 1e-3 and abs(prt.eta()) < 6.0:
                    g = fastjet.PseudoJet(prt.px() * GHOST, prt.py() * GHOST, prt.pz() * GHOST, prt.e() * GHOST)
                    g.set_user_index(len(plab) - 1)
                    parts.append(g)
            if not prt.isFinal() or not prt.isVisible():
                continue
            if abs(prt.eta()) > 4.9:
                continue
            parts.append(fastjet.PseudoJet(prt.px(), prt.py(), prt.pz(), prt.e()))
        cs = fastjet.ClusterSequence(parts, jet_def)
        jets = cs.inclusive_jets(min(0.8 * args.ptmin, args.anc_ptmin) if anc else 0.8 * args.ptmin)
        keep_all = anc and args.anc_keep_all
        if len(jets) < args.min_jets and not keep_all:
            continue
        tpt = np.array([j.pt() for j in jets])
        teta = np.array([j.eta() for j in jets])
        tphi = np.array([j.phi_std() for j in jets])
        tm = np.array([max(j.m(), 0.0) for j in jets])
        spt = smear_pt(tpt, rng) if len(jets) else tpt.copy()
        seta = teta + 0.01 * rng.standard_normal(teta.shape)
        sphi = dphi(tphi + 0.01 * rng.standard_normal(tphi.shape), 0.0)
        sm = tm * spt / np.maximum(tpt, 1e-9)
        # the stored jets keep their definition (true pT > 0.8 ptmin before smearing) when softer jets are clustered
        sel = (spt > args.ptmin) & (np.abs(seta) < args.etamax) & (tpt >= 0.8 * args.ptmin)
        passed = sel.sum() >= args.min_jets
        if not passed and not keep_all:
            continue
        order = np.argsort(-spt[sel])
        idx = np.where(sel)[0][order][:MJ]
        if spt[idx].sum() < args.min_ht:
            passed = False
        elif sym_filter is not None and passed and not has_symmetric_split(
                spt[idx], seta[idx], sphi[idx], sm[idx], *sym_filter):
            passed = False
        if not passed and not keep_all:
            continue
        nj = len(idx)
        if anc:
            plab_a, pp4_a = np.array(plab, dtype=int), np.array(pp4, dtype=np.float64).reshape(-1, 4)
            lp4 = np.zeros((len(jets), 3, 4)); lpt = np.zeros((len(jets), 3))
            for ji, j in enumerate(jets):
                for c in j.constituents():
                    u = c.user_index()
                    if u >= 0:
                        lp4[ji, plab_a[u]] += pp4_a[u]
                        lpt[ji, plab_a[u]] += np.hypot(pp4_a[u, 1], pp4_a[u, 2])
            oall = np.argsort(-tpt)[:AJ]
            na = len(oall)
            std = np.full(len(jets), -1, dtype=np.int32); std[idx] = np.arange(nj)
            z = lambda shape: np.zeros(shape, dtype=np.float32)
            a_tj, a_sj, a_lp4, a_lpt, a_std = z((AJ, 4)), z((AJ, 4)), z((AJ, 3, 4)), z((AJ, 3)), np.full(AJ, -1, np.int32)
            a_tj[:na] = np.stack([tpt, teta, tphi, tm], 1)[oall]; a_sj[:na] = np.stack([spt, seta, sphi, sm], 1)[oall]
            a_lp4[:na], a_lpt[:na], a_std[:na] = lp4[oall], lpt[oall], std[oall]
            fr = z((MJ, 3)); tot = lpt[idx].sum(1, keepdims=True)
            fr[:nj] = np.where(tot > 0, lpt[idx] / np.maximum(tot, 1e-12), np.array([0.0, 0.0, 1.0]))
            aout["pass"].append(passed)
            aout["frac"].append(fr); aout["tj"].append(a_tj); aout["sj"].append(a_sj); aout["lp4"].append(a_lp4)
            aout["lpt"].append(a_lpt); aout["std"].append(a_std); aout["nall"].append(na)
            aout["gp4"].append(np.array([[ev[g].e(), ev[g].px(), ev[g].py(), ev[g].pz()] for g in gl], dtype=np.float32))
            aout["dp4"].append(np.stack([pp4_a[plab_a == k].sum(0) for k in range(3)]).astype(np.float32))

        def pad(a):
            z = np.zeros(MJ, dtype=np.float32)
            z[:nj] = a[idx]
            return z

        out["pt"].append(pad(spt))
        out["eta"].append(pad(seta))
        out["phi"].append(pad(sphi))
        out["mass"].append(pad(sm))
        out["ntot"].append(nj)
        out["w"].append(p.infoPython().weight())

        tg = np.full(6, -1, dtype=np.int32)
        qk = np.zeros((6, 3), dtype=np.float32)
        qb, qa = np.zeros((6, 3), np.float32), np.zeros((6, 3), np.float32)
        if groups is not None:
            qb[:], qa[:] = np.array(alt[False][0] + alt[False][1]), np.array(alt[True][0] + alt[True][1])
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
        out["qb"].append(qb)
        out["qa"].append(qa)

        if len(out["pt"]) % 1000 == 0:
            el = time.time() - t0
            print(f"  {len(out['pt'])}/{args.n} written, {n_tried} tried, {el:.0f}s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    pt = np.stack(out["pt"])
    tmp = args.out + ".tmp"  # written then renamed: a killed job never leaves a "finished" file
    with h5py.File(tmp, "w") as f:
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
        f["truth/quarks_before_fsr"] = np.stack(out["qb"])
        f["truth/quarks_after_fsr"] = np.stack(out["qa"])
        f.attrs["truth_quarks"] = args.truth_quarks
        f["EventVars/normweight"] = np.array(out["w"], dtype=np.float64)
        f["EventVars/parent_mass"] = np.array(out["pm"], dtype=np.float32)
        f["EventVars/n_jets"] = np.array(out["ntot"], dtype=np.int32)
        if anc:
            g["anc"] = np.stack(aout["frac"])  # (N, max_jets, 3): parton pT fraction from gluino 1 / gluino 2 / other
            f["truth/gluino_p4"] = np.stack(aout["gp4"])  # (N, 2, 4) (E, px, py, pz) of the decaying gluinos
            f["truth/desc_p4"] = np.stack(aout["dp4"])  # (N, 3, 4) sum of final partons per label (g1, g2, other)
            a = f.create_group("ANC")  # all jets with true pT > anc_ptmin, |eta_particles| < 4.9, sorted by true pT
            for k, name in (("tj", "true"), ("sj", "smeared")):  # (N, AJ, 4): pt, eta, phi, mass
                a.create_dataset(name, data=np.stack(aout[k]), compression="gzip", compression_opts=4)
            a.create_dataset("label_p4", data=np.stack(aout["lp4"]), compression="gzip", compression_opts=4)  # (N, AJ, 3, 4)
            a.create_dataset("label_pt", data=np.stack(aout["lpt"]), compression="gzip", compression_opts=4)  # (N, AJ, 3) scalar pT
            a.create_dataset("std_index", data=np.stack(aout["std"]), compression="gzip", compression_opts=4)  # slot in INPUTS/Source or -1
            a["n"] = np.array(aout["nall"], dtype=np.int32)
            f["EventVars/pass_sel"] = np.array(aout["pass"], dtype=bool)
            f.attrs["anc_ptmin"] = args.anc_ptmin
        f.attrs["process"] = args.process
        f.attrs["mass"] = args.mass
        f.attrs["n_tried"] = n_tried
        f.attrs["sigma_mb"] = p.infoPython().sigmaGen()
        f.attrs["generator"] = "pythia8 + fastjet anti-kt R=0.4, smeared"
        # sum of weights over ALL generated events: per-event cross-section weight is
        # normweight * sigma_mb / weight_sum, which lets samples with different
        # pTHat settings be combined
        f.attrs["weight_sum"] = float(p.infoPython().weightSum())
    os.replace(tmp, args.out)
    print(f"done: {len(pt)} written / {n_tried} tried in {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
