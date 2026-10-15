"""
Generator-level check of the truth-quark definition (runs Pythia directly).

  raw   : the gluino's daughters as first written
  bot   : last carbon copy, after recoil but before the quark's own FSR
          (Pythia iBotCopy; CMS isLastCopyBeforeFSR)
  botid : last copy with the same id, traced through q -> q g to the end of
          the quark's shower (iBotCopyId; CMS isLastCopy). data/pythia used this.

Jets and matching as in generate_pythia.py (unique greedy dR < 0.4). Prints the
fully-matched fraction, the mass of the matched triplets and, for events
matched under both bot and botid, how often the true pairing differs and which
definition's triplets are closer to the gluino mass there.

  python scripts/truth_quarks_generator.py 1000 1500     # mass, events
"""

import os
import sys
from types import SimpleNamespace

import fastjet
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_pythia as gp  # noqa: E402

MODES = ("raw", "bot", "botid")


def quarks(ev, mode):
    """[(pt, eta, phi) x 3] per gluino for one truth definition, or None."""
    groups = []
    for i in range(ev.size()):
        g = ev[i]
        if abs(g.id()) != gp.GLUINO or g.daughter1() <= 0:
            continue
        dl = list(range(g.daughter1(), max(g.daughter1(), g.daughter2()) + 1))
        if any(abs(ev[d].id()) == gp.GLUINO for d in dl):
            continue
        qs = [d for d in dl if 1 <= abs(ev[d].id()) <= 6]
        qs = [{"raw": q, "bot": ev[q].iBotCopy(), "botid": ev[q].iBotCopyId()}[mode] for q in qs]
        groups.append([(ev[q].pT(), ev[q].eta(), ev[q].phi()) for q in qs])
    return groups if len(groups) == 2 and all(len(x) == 3 for x in groups) else None


def match(q, jeta, jphi, dr_max=0.4):
    q = np.array(q)
    nj = len(jeta)
    dr = np.sqrt((q[:, 1:2] - jeta[None]) ** 2 + gp.dphi(q[:, 2:3], jphi[None]) ** 2)
    tg = np.full(6, -1)
    uq, uj = set(), set()
    for flat in np.argsort(dr, axis=None):
        qi, ji = divmod(int(flat), nj)
        if dr[qi, ji] > dr_max:
            break
        if qi in uq or ji in uj:
            continue
        tg[qi] = ji
        uq.add(qi)
        uj.add(ji)
    return tg


def main():
    mass, n_events = float(sys.argv[1]), int(sys.argv[2])
    p = gp.make_pythia(SimpleNamespace(process="signal", mass=mass, width_frac=0.01, decay="direct",
                                       squark_mass=None, seed=int(mass) + 9, alphas_fsr=None,
                                       alphas_isr=None, no_mpi=False, no_had=False))
    rng = np.random.default_rng(1)
    jd = fastjet.JetDefinition(fastjet.antikt_algorithm, 0.4)
    full = {m: [] for m in MODES}
    m3 = {m: [] for m in MODES}
    same_pairing, m_differ = [], {"bot": [], "botid": []}
    n = 0
    while n < n_events:
        if not p.next():
            continue
        ev = p.event
        qd = {m: quarks(ev, m) for m in MODES}
        if any(v is None for v in qd.values()):
            continue
        parts = [fastjet.PseudoJet(x.px(), x.py(), x.pz(), x.e()) for x in (ev[i] for i in range(ev.size()))
                 if x.isFinal() and x.isVisible() and abs(x.eta()) < 4.9]
        jets = fastjet.ClusterSequence(parts, jd).inclusive_jets(24.0)
        pt = np.array([j.pt() for j in jets])
        eta = np.array([j.eta() for j in jets])
        phi = np.array([j.phi_std() for j in jets])
        p4 = np.array([[j.e(), j.px(), j.py(), j.pz()] for j in jets])
        spt = gp.smear_pt(pt, rng)
        sel = (spt > 30) & (np.abs(eta) < 3.0)
        if sel.sum() < 6:
            continue
        idx = np.where(sel)[0][np.argsort(-spt[sel])][:20]
        n += 1

        def tmass(h):
            s = p4[idx[h]].sum(0)
            return np.sqrt(max(s[0] ** 2 - (s[1:] ** 2).sum(), 0)) / mass

        tgs = {m: match(qd[m][0] + qd[m][1], eta[idx], phi[idx]) for m in MODES}
        for m in MODES:
            ok = (tgs[m] >= 0).all()
            full[m].append(ok)
            if ok:
                m3[m] += [tmass(tgs[m][:3]), tmass(tgs[m][3:])]
        if full["bot"][-1] and full["botid"][-1]:
            pb = {frozenset(tgs["bot"][:3]), frozenset(tgs["bot"][3:])}
            pi = {frozenset(tgs["botid"][:3]), frozenset(tgs["botid"][3:])}
            same_pairing.append(pb == pi)
            if pb != pi:
                for m in ("bot", "botid"):
                    m_differ[m] += [tmass(tgs[m][:3]), tmass(tgs[m][3:])]
    for m in MODES:
        v = np.array(m3[m])
        lo, hi = np.quantile(v, [0.16, 0.84])
        print(f"M={mass:.0f} {m:6s} all six matched {np.mean(full[m]):.3f}   matched-triplet mass/M "
              f"median {np.median(v):.3f}, 16-84%: {lo:.3f}-{hi:.3f}")
    both = np.mean(np.array(full["bot"]) & np.array(full["botid"]))
    print(f"M={mass:.0f} matched under both: {both:.3f}; of those the true pairing differs in "
          f"{1 - np.mean(same_pairing):.3f}")
    for m in ("bot", "botid"):
        v = np.array(m_differ[m])
        if len(v):
            print(f"M={mass:.0f}   where it differs, {m:6s} triplets: median mass/M {np.median(v):.3f}, "
                  f"within 0.95 +- 0.1 for {np.mean(np.abs(v - 0.95) < 0.1):.2f}")


if __name__ == "__main__":
    main()
