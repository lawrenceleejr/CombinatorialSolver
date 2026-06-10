"""
Generate synthetic HDF5 data matching the MadGraphMLProducer format.

Creates fake pair-produced resonance events where each parent decays
to 3 jets. Matches the real data layout including TARGETS and INPUTS groups.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np


def generate_event(parent_mass: float, include_isr: bool = False) -> dict:
    """Generate one event with two parent particles decaying to 3 jets each.

    Jets are stored in truth-group order (g1 first, g2 second, optional ISR last).
    The TARGETS encode this original ordering.

    Args:
        parent_mass: Invariant mass of each parent particle in GeV.
        include_isr: If True, add a 7th ISR jet.
    """
    jets = []

    # Generate two parent particles (roughly back-to-back in transverse plane).
    # Heavy pair production happens near threshold, so the parents are CENTRAL
    # with a small, correlated rapidity separation (shared longitudinal boost
    # plus a small y*), unlike QCD's forward-peaked t-channel topology.
    parent_pt = np.random.exponential(parent_mass * 0.3)
    parent_phi1 = np.random.uniform(-np.pi, np.pi)
    parent_phi2 = parent_phi1 + np.pi
    y_boost = np.random.normal(0, 0.5)
    y_star_pair = np.random.normal(0, 0.35)
    parent_eta1 = y_boost + y_star_pair
    parent_eta2 = y_boost - y_star_pair

    for p_phi, p_eta in [(parent_phi1, parent_eta1), (parent_phi2, parent_eta2)]:
        fracs = np.random.dirichlet([2, 2, 2])
        jet_pt_total = np.sqrt(parent_pt**2 + parent_mass**2) * 0.8

        for frac in fracs:
            pt = max(jet_pt_total * frac + np.random.normal(0, 10), 25.0)
            eta = p_eta + np.random.normal(0, 0.5)
            phi = p_phi + np.random.normal(0, 0.4)
            phi = ((phi + np.pi) % (2 * np.pi)) - np.pi
            mass = np.random.exponential(0.005)  # near-massless partons
            jets.append({"pt": pt, "eta": eta, "phi": phi, "mass": mass})

    if include_isr:
        isr_pt = max(np.random.exponential(40.0), 25.0)
        jets.append({
            "pt": isr_pt,
            "eta": np.random.uniform(-2.5, 2.5),
            "phi": np.random.uniform(-np.pi, np.pi),
            "mass": np.random.exponential(0.003),
        })

    n_jets = len(jets)
    max_jets = 20

    # Build arrays in truth order (g1=[0,1,2], g2=[3,4,5], ISR=6 if present)
    pt = np.zeros(max_jets, dtype=np.float32)
    eta = np.zeros(max_jets, dtype=np.float32)
    phi = np.zeros(max_jets, dtype=np.float32)
    mass = np.zeros(max_jets, dtype=np.float32)
    mask = np.zeros(max_jets, dtype=bool)

    for i, j in enumerate(jets):
        pt[i] = j["pt"]
        eta[i] = j["eta"]
        phi[i] = j["phi"]
        mass[i] = j["mass"]
        mask[i] = True

    # Compute energy
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(px**2 + py**2 + pz**2 + mass**2)

    ht = pt[mask].sum()

    # jet_features: [pt, eta, phi, mass, parent_idx, is_signal]
    jet_features = np.zeros((max_jets, 6), dtype=np.float32)
    for i in range(n_jets):
        jet_features[i, 0] = pt[i]
        jet_features[i, 1] = eta[i]
        jet_features[i, 2] = phi[i]
        jet_features[i, 3] = mass[i]
        if i < 3:
            jet_features[i, 4] = 1.0  # parent_idx for g1
        elif i < 6:
            jet_features[i, 4] = 2.0  # parent_idx for g2
        else:
            jet_features[i, 4] = 0.0  # ISR
        jet_features[i, 5] = 0.0  # is_signal (matches real data convention)

    event_features = np.array(
        [n_jets, 0.0, 0.0, 0.0, ht, 6, 1.0], dtype=np.float32
    )

    return {
        "jet_features": jet_features,
        "jet_mask": mask,
        "event_features": event_features,
        "pt": pt, "eta": eta, "phi": phi, "mass": mass,
        "energy": energy, "mask": mask,
    }


def _lorentz_boost_to_lab(q_rest: np.ndarray, parent_lab: np.ndarray) -> np.ndarray:
    """Boost a rest-frame four-vector ``q_rest`` (E, px, py, pz) into the lab
    frame, where ``parent_lab`` is the parent's lab four-vector (its mass defines
    the rest frame).  Standard active Lorentz boost along the parent velocity."""
    E_p, px, py, pz = parent_lab
    M = np.sqrt(max(E_p**2 - px**2 - py**2 - pz**2, 1e-12))
    beta = np.array([px, py, pz]) / E_p
    b2 = float(beta @ beta)
    gamma = E_p / M
    E_q = q_rest[0]
    p_q = q_rest[1:]
    bp = float(beta @ p_q)
    if b2 < 1e-12:
        return q_rest.copy()
    E_lab = gamma * (E_q + bp)
    p_lab = p_q + ((gamma - 1.0) * bp / b2 + gamma * E_q) * beta
    return np.array([E_lab, p_lab[0], p_lab[1], p_lab[2]])


def _two_body_decay(parent_lab: np.ndarray, m1: float, m2: float) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic relativistic 2-body decay of ``parent_lab`` → (m1, m2).

    Returns the two daughter four-vectors in the lab frame.
    """
    E_p, px, py, pz = parent_lab
    M = np.sqrt(max(E_p**2 - px**2 - py**2 - pz**2, 1e-12))
    # Daughter momentum magnitude in the parent rest frame (Källén function).
    p_star = np.sqrt(max((M**2 - (m1 + m2) ** 2) * (M**2 - (m1 - m2) ** 2), 0.0)) / (2 * M)
    # Isotropic direction in the rest frame.
    cos_t = np.random.uniform(-1, 1)
    sin_t = np.sqrt(max(1 - cos_t**2, 0.0))
    phi = np.random.uniform(0, 2 * np.pi)
    n = np.array([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t])
    d1_rest = np.array([np.sqrt(p_star**2 + m1**2), *(p_star * n)])
    d2_rest = np.array([np.sqrt(p_star**2 + m2**2), *(-p_star * n)])
    return _lorentz_boost_to_lab(d1_rest, parent_lab), _lorentz_boost_to_lab(d2_rest, parent_lab)


def _fourvec_to_ptetaphim(p: np.ndarray) -> dict:
    """Convert (E, px, py, pz) → {pt, eta, phi, mass} jet dict."""
    E, px, py, pz = p
    pt = np.sqrt(px**2 + py**2)
    eta = np.arcsinh(pz / pt) if pt > 1e-9 else 0.0
    phi = np.arctan2(py, px)
    mass = np.sqrt(max(E**2 - px**2 - py**2 - pz**2, 0.0))
    return {"pt": float(pt), "eta": float(eta), "phi": float(phi), "mass": float(mass)}


def generate_cascade_event(
    gluino_mass: float, squark_mass: float, include_isr: bool = False
) -> dict:
    """Generate a resonant-triplet signal event: g~ → q + sq~(→ q q).

    Each of the two (equal-mass) gluinos decays to a quark plus an on-shell
    squark, and the squark decays to two quarks via an RPV vertex.  Every gluino
    therefore yields a 3-jet triplet whose total invariant mass is the gluino
    mass, but with a *resonant* pairwise mass at the squark mass — a populated
    band/line inside the Dalitz triangle.  This is the topology the network must
    still recognise as signal (the two triplets are equal-mass, so the
    argmin|m1-m2| truth labelling groups them correctly), even though its
    internal substructure differs from a flat 3-body RPV decay.

    Jets are stored in truth-group order (g1=[0,1,2], g2=[3,4,5], optional ISR
    last); the resonant quark pair from each squark is placed at indices (1,2)
    and (4,5).
    """
    if squark_mass >= gluino_mass:
        raise ValueError("squark_mass must be < gluino_mass for an on-shell cascade")

    jets = []
    # Two gluinos, roughly back-to-back in the transverse plane and CENTRAL in
    # rapidity (near-threshold pair production: shared longitudinal boost plus
    # a small correlated y*), unlike QCD's forward-peaked t-channel topology.
    parent_pt = np.random.exponential(gluino_mass * 0.3)
    phi1 = np.random.uniform(-np.pi, np.pi)
    y_boost = np.random.normal(0, 0.5)
    y_star_pair = np.random.normal(0, 0.35)
    for sign, p_eta in [(0.0, y_boost + y_star_pair), (np.pi, y_boost - y_star_pair)]:
        p_phi = phi1 + sign
        # Gluino lab four-vector.
        glu_pt = parent_pt
        glu_px = glu_pt * np.cos(p_phi)
        glu_py = glu_pt * np.sin(p_phi)
        glu_pz = np.sqrt(glu_pt**2 + gluino_mass**2) * np.sinh(p_eta)
        glu_E = np.sqrt(glu_px**2 + glu_py**2 + glu_pz**2 + gluino_mass**2)
        glu = np.array([glu_E, glu_px, glu_py, glu_pz])

        # g~ → q (massless) + sq~ (on-shell).
        q1, squark = _two_body_decay(glu, 0.0, squark_mass)
        # sq~ → q + q (massless RPV decay).
        q2, q3 = _two_body_decay(squark, 0.0, 0.0)
        for p in (q1, q2, q3):
            jets.append(_fourvec_to_ptetaphim(p))

    if include_isr:
        isr_pt = max(np.random.exponential(40.0), 25.0)
        jets.append({
            "pt": isr_pt,
            "eta": float(np.random.uniform(-2.5, 2.5)),
            "phi": float(np.random.uniform(-np.pi, np.pi)),
            "mass": float(np.random.exponential(0.003)),
        })

    return _assemble_event(jets, include_isr)


def _assemble_event(jets: list[dict], include_isr: bool) -> dict:
    """Pack a list of jet dicts (truth-group order) into the HDF5 event layout."""
    n_jets = len(jets)
    max_jets = 20

    pt = np.zeros(max_jets, dtype=np.float32)
    eta = np.zeros(max_jets, dtype=np.float32)
    phi = np.zeros(max_jets, dtype=np.float32)
    mass = np.zeros(max_jets, dtype=np.float32)
    mask = np.zeros(max_jets, dtype=bool)
    for i, j in enumerate(jets):
        pt[i], eta[i], phi[i], mass[i], mask[i] = j["pt"], j["eta"], j["phi"], j["mass"], True

    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    ht = pt[mask].sum()

    jet_features = np.zeros((max_jets, 6), dtype=np.float32)
    for i in range(n_jets):
        jet_features[i, 0:4] = pt[i], eta[i], phi[i], mass[i]
        jet_features[i, 4] = 1.0 if i < 3 else (2.0 if i < 6 else 0.0)
        jet_features[i, 5] = 0.0  # is_signal (matches real data convention)

    event_features = np.array([n_jets, 0.0, 0.0, 0.0, ht, 6, 1.0], dtype=np.float32)
    return {
        "jet_features": jet_features, "jet_mask": mask,
        "event_features": event_features,
        "pt": pt, "eta": eta, "phi": phi, "mass": mass,
        "energy": energy, "mask": mask,
    }


def generate_background_event(include_isr: bool = False) -> dict:
    """Generate one QCD-like multijet background event (no resonance).

    Modeled as a t-channel-like 2->2 dijet skeleton with collinear splitting,
    so the toy reproduces the QCD correlations the analysis exploits:

      - the two leading partons are back-to-back in phi with a falling pT
        spectrum (so the assembled triplets are roughly back-to-back, like
        real QCD recoil — Delta-phi is NOT a free discriminator);
      - the rapidity separation is drawn forward-peaked (flat in chi = e^{2y*},
        the Rutherford t-channel limit), so high-y* is QCD-enriched — the
        chi-sideband background transfer has a populated control region;
      - each parton splits collinearly into 3 jets with soft/collinear-enhanced
        energy sharing, populating the Dalitz edges/corners.

    Jets respect the analysis acceptance (pT > 30 GeV, |eta| < 2.4).
    ``is_signal`` is 0 for every jet and the event carries no real parent.
    Used to exercise the background-rejection / chi-transfer paths before real
    MadGraph QCD is available.
    """
    # --- Hard 2->2 skeleton -------------------------------------------------
    # Falling parton pT spectrum; back-to-back in phi with a small acoplanarity
    # kick (toy ISR recoil).  The scale is chosen so the reconstructed m_avg
    # spectrum falls steeply but keeps a populated tail through the TeV search
    # range (a zero-background search region would make the toy bump hunt
    # degenerate).
    parton_pt = 30.0 + np.random.exponential(200.0)
    phi1 = np.random.uniform(-np.pi, np.pi)
    phi2 = phi1 + np.pi + np.random.normal(0.0, 0.25)

    # Forward-peaked rapidity separation: flat in chi (t-channel Rutherford
    # limit), i.e. y* = ln(chi)/2 with chi uniform in [1, chi_max].  The boost
    # of the dijet system spreads events in y_boost (PDF proxy).
    chi_max = 12.0
    y_star = 0.5 * np.log(np.random.uniform(1.0, chi_max))
    y_boost = np.random.normal(0.0, 0.6)
    eta1 = y_boost + y_star
    eta2 = y_boost - y_star

    jets: list[dict] = []
    for p_phi, p_eta in ((phi1, eta1), (phi2, eta2)):
        # Soft/collinear-enhanced energy sharing: Dirichlet with small alpha
        # gives one hard + two soft fragments (DGLAP-like), populating the
        # Dalitz edges — unlike a resonance decay's flat interior.
        fracs = np.random.dirichlet([1.0, 1.0, 1.0])
        for frac in np.sort(fracs)[::-1]:
            pt_j = max(parton_pt * frac + np.random.normal(0.0, 8.0), 30.0)
            # Collinear spread shrinks for harder fragments.
            spread = 0.25 + 0.25 * (1.0 - frac)
            eta_j = np.clip(p_eta + np.random.normal(0.0, spread), -2.4, 2.4)
            phi_j = p_phi + np.random.normal(0.0, spread)
            phi_j = ((phi_j + np.pi) % (2 * np.pi)) - np.pi
            jets.append({
                "pt": float(pt_j),
                "eta": float(eta_j),
                "phi": float(phi_j),
                "mass": float(np.random.exponential(0.005)),
            })

    if include_isr:
        jets.append({
            "pt": float(max(np.random.exponential(40.0), 30.0)),
            "eta": float(np.random.uniform(-2.4, 2.4)),
            "phi": float(np.random.uniform(-np.pi, np.pi)),
            "mass": float(np.random.exponential(0.003)),
        })

    event = _assemble_event(jets, include_isr)
    # Background: no parent, n_signal = 0.
    event["jet_features"][:, 4] = 0.0
    event["event_features"][5] = 0
    return event


def generate_dataset(
    output_path: str,
    n_events: int = 10000,
    parent_masses: list[float] | None = None,
    include_isr: bool = False,
    background: bool = False,
    cascade: bool = False,
    squark_mass: float | None = None,
):
    """Generate a full HDF5 dataset matching the real data layout.

    When ``background`` is True, QCD-like multijet events (no resonance,
    ``is_signal=0``) are produced instead of signal events — a synthetic stand-in
    for a MadGraph QCD sample to test the background-rejection training path.

    When ``cascade`` is True, resonant-triplet signal events are produced
    (g~ → q + sq~(→ q q)): each gluino's triplet carries the gluino mass with an
    on-shell squark resonance inside it.  ``squark_mass`` sets the resonance
    (default: 0.4 × gluino mass per event).
    """
    if parent_masses is None:
        parent_masses = [300.0, 500.0, 700.0, 1000.0, 1500.0]

    max_jets = 20
    n_jets_per_event = 7 if include_isr else 6

    # Preallocate
    jet_features_all = np.zeros((n_events, max_jets, 6), dtype=np.float32)
    jet_mask_all = np.zeros((n_events, max_jets), dtype=bool)
    event_features_all = np.zeros((n_events, 7), dtype=np.float32)
    particle_features_all = np.zeros((n_events, max_jets, 100, 5), dtype=np.float32)
    particle_mask_all = np.zeros((n_events, max_jets, 100), dtype=bool)

    # INPUTS/Source arrays
    src_pt = np.zeros((n_events, max_jets), dtype=np.float32)
    src_eta = np.zeros((n_events, max_jets), dtype=np.float32)
    src_phi = np.zeros((n_events, max_jets), dtype=np.float32)
    src_mass = np.zeros((n_events, max_jets), dtype=np.float32)
    src_btag = np.zeros((n_events, max_jets), dtype=np.float32)
    src_mask = np.zeros((n_events, max_jets), dtype=bool)

    # source group arrays (E, eta, phi, pt)
    source_e = np.zeros((n_events, max_jets), dtype=np.float32)
    source_eta = np.zeros((n_events, max_jets), dtype=np.float32)
    source_phi = np.zeros((n_events, max_jets), dtype=np.float32)
    source_pt = np.zeros((n_events, max_jets), dtype=np.float32)
    source_mask = np.zeros((n_events, max_jets), dtype=bool)

    # TARGETS
    g1_j1 = np.zeros(n_events, dtype=np.int32)
    g1_j2 = np.ones(n_events, dtype=np.int32)
    g1_j3 = np.full(n_events, 2, dtype=np.int32)
    g2_j1 = np.full(n_events, 3, dtype=np.int32)
    g2_j2 = np.full(n_events, 4, dtype=np.int32)
    g2_j3 = np.full(n_events, 5, dtype=np.int32)

    for i in range(n_events):
        if background:
            event = generate_background_event(include_isr=include_isr)
        elif cascade:
            mass = float(np.random.choice(parent_masses))
            sq = squark_mass if squark_mass is not None else 0.4 * mass
            event = generate_cascade_event(
                gluino_mass=mass, squark_mass=min(sq, 0.9 * mass), include_isr=include_isr
            )
        else:
            mass = np.random.choice(parent_masses)
            event = generate_event(parent_mass=mass, include_isr=include_isr)

        jet_features_all[i] = event["jet_features"]
        jet_mask_all[i] = event["jet_mask"]
        event_features_all[i] = event["event_features"]

        src_pt[i] = event["pt"]
        src_eta[i] = event["eta"]
        src_phi[i] = event["phi"]
        src_mass[i] = event["mass"]
        src_mask[i] = event["mask"]

        source_e[i] = event["energy"]
        source_eta[i] = event["eta"]
        source_phi[i] = event["phi"]
        source_pt[i] = event["pt"]
        source_mask[i] = event["mask"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # Standard datasets
        f.create_dataset("jet_features", data=jet_features_all)
        f.create_dataset("jet_mask", data=jet_mask_all)
        f.create_dataset("event_features", data=event_features_all)
        f.create_dataset("particle_features", data=particle_features_all)
        f.create_dataset("particle_mask", data=particle_mask_all)

        # INPUTS/Source group
        inputs_src = f.create_group("INPUTS/Source")
        inputs_src.create_dataset("pt", data=src_pt)
        inputs_src.create_dataset("eta", data=src_eta)
        inputs_src.create_dataset("phi", data=src_phi)
        inputs_src.create_dataset("mass", data=src_mass)
        inputs_src.create_dataset("btag", data=src_btag)
        inputs_src.create_dataset("MASK", data=src_mask)

        # source group
        source_grp = f.create_group("source")
        source_grp.create_dataset("e", data=source_e)
        source_grp.create_dataset("eta", data=source_eta)
        source_grp.create_dataset("phi", data=source_phi)
        source_grp.create_dataset("pt", data=source_pt)
        source_grp.create_dataset("mask", data=source_mask)

        # TARGETS group
        g1 = f.create_group("TARGETS/g1")
        g1.create_dataset("j1", data=g1_j1)
        g1.create_dataset("j2", data=g1_j2)
        g1.create_dataset("j3", data=g1_j3)
        g2 = f.create_group("TARGETS/g2")
        g2.create_dataset("j1", data=g2_j1)
        g2.create_dataset("j2", data=g2_j2)
        g2.create_dataset("j3", data=g2_j3)

        # EventVars
        ev = f.create_group("EventVars")
        ev.create_dataset("normweight", data=np.ones(n_events, dtype=np.float32))

    print(f"Generated {n_events} events -> {output_path}")
    if background:
        print("  Sample type: QCD-like background (is_signal=0, no resonance)")
    elif cascade:
        sq_desc = f"{squark_mass:.0f} GeV" if squark_mass is not None else "0.4 x gluino mass"
        print(f"  Sample type: resonant-triplet signal g~->q+sq~(->qq), squark mass = {sq_desc}")
        print(f"  Gluino masses sampled from: {parent_masses} GeV")
    else:
        print(f"  Parent masses sampled from: {parent_masses} GeV")
    print(f"  Jets per event: {n_jets_per_event} ({'with' if include_isr else 'without'} ISR)")
    print(f"  File size: {Path(output_path).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mock HDF5 data")
    parser.add_argument("--output", type=str, default="data/mock_data.h5")
    parser.add_argument("--n-events", type=int, default=10000)
    parser.add_argument("--masses", type=float, nargs="+", default=None)
    parser.add_argument("--include-isr", action="store_true", help="Add ISR jet (7 jets)")
    parser.add_argument(
        "--background",
        action="store_true",
        help="Generate QCD-like multijet background events (is_signal=0, no resonance)",
    )
    parser.add_argument(
        "--cascade",
        action="store_true",
        help="Generate resonant-triplet signal: g~ -> q + sq~(-> q q) (on-shell squark inside each triplet)",
    )
    parser.add_argument(
        "--squark-mass",
        type=float,
        default=None,
        help="Squark resonance mass in GeV for --cascade (default: 0.4 x gluino mass per event)",
    )
    args = parser.parse_args()
    generate_dataset(
        args.output, args.n_events, args.masses, args.include_isr,
        background=args.background, cascade=args.cascade, squark_mass=args.squark_mass,
    )
