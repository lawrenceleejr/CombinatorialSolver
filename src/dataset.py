"""
HDF5 dataset for jet assignment training.

Reads MadGraphMLProducer-format HDF5 files. Supports two data layouts:
  1. TARGETS-based: truth groups in TARGETS/g1/{j1,j2,j3} and TARGETS/g2/{j1,j2,j3}
  2. Column-based: truth from jet_features columns (parent_pdg, is_signal)

Jets are pT-sorted before being fed to the model. Truth indices are remapped
accordingly. Four-vectors are converted from (pt, eta, phi, mass) to (E, px, py, pz).

By default (``use_mass_asymmetry_labels=True``) truth labels are computed by
finding the assignment that minimises |m1 - m2| over all 70 interpretations
(for 7-jet events) or 10 interpretations (for 6-jet events).  This is the
physically correct approach for pair-produced, equal-mass resonances: in a
sample with no detector smearing and no jet fragmentation the truth grouping
gives m1 ≈ m2 ≈ M_parent while any wrong-ISR interpretation yields a much
larger |m1 - m2|.  It also removes any dependence on potentially incorrect
TARGETS labels in the HDF5 file (e.g. samples where the generator stores jets
in an ordering that does not reflect parent-decay membership).
"""

import glob as globmod
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .combinatorics import build_assignment_tensors, enumerate_assignments, match_truth_groups


def pt_eta_phi_mass_to_epxpypz(pt, eta, phi, mass):
    """Convert (pt, eta, phi, mass) to (E, px, py, pz)."""
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    energy = np.sqrt(px**2 + py**2 + pz**2 + mass**2)
    return np.stack([energy, px, py, pz], axis=-1)


class JetAssignmentDataset(Dataset):
    """Dataset that loads jet four-momenta and truth assignment labels from HDF5.

    Args:
        data_paths: Path(s) to HDF5 files — a single path, list, or glob pattern.
        num_jets: Number of leading jets to use (6 or 7).
        normalize_by_ht: If True, divide 4-vectors by event HT for scale invariance.
        pt_smear_frac: If > 0, apply Gaussian pT smearing with this fractional
            resolution (e.g. 0.05 for 5%). Smearing is applied per-jet before
            the four-vector conversion, simulating detector resolution effects.
            The smearing is applied once at load time (fixed per event).
        use_mass_asymmetry_labels: If True (default), compute truth labels as the
            assignment index that minimises |m1 - m2| over all interpretations,
            using the raw (E, px, py, pz) four-vectors.  This is the physically
            correct ground truth for equal-mass pair-produced resonances and
            removes any dependence on TARGETS entries in the HDF5 file, which
            may be incorrect for some topologies (e.g. when the generator stores
            jets in an ordering that does not reflect parent-decay membership).
            Set to False to fall back to the TARGETS-based labelling.
    """

    def __init__(
        self,
        data_paths: str | list[str],
        num_jets: int = 7,
        normalize_by_ht: bool = True,
        pt_smear_frac: float = 0.0,
        use_mass_asymmetry_labels: bool = True,
    ):
        self.num_jets = num_jets
        self.normalize_by_ht = normalize_by_ht
        self.pt_smear_frac = pt_smear_frac
        self.use_mass_asymmetry_labels = use_mass_asymmetry_labels

        # Resolve file paths
        if isinstance(data_paths, str):
            files = sorted(globmod.glob(data_paths))
            if not files:
                files = [data_paths]
        else:
            files = list(data_paths)

        # Load and concatenate all files
        all_four_momenta = []
        all_labels = []
        all_masses = []
        all_ht = []

        for fpath in files:
            four_mom, labels, parent_mass, ht = self._load_file(fpath)
            all_four_momenta.append(four_mom)
            all_labels.append(labels)
            all_masses.append(parent_mass)
            all_ht.append(ht)

        self.four_momenta = torch.cat(all_four_momenta, dim=0)
        self.labels = torch.cat(all_labels, dim=0)
        self.parent_masses = torch.cat(all_masses, dim=0)
        self.ht = torch.cat(all_ht, dim=0)

        # Normalize parent masses to TeV scale for stable adversarial training
        self.parent_masses = self.parent_masses / 1000.0

        # Filter out events with invalid labels
        valid = self.labels >= 0
        n_before = len(self.labels)
        self.four_momenta = self.four_momenta[valid]
        self.labels = self.labels[valid]
        self.parent_masses = self.parent_masses[valid]
        self.ht = self.ht[valid]
        n_after = len(self.labels)

        if n_after < n_before:
            print(f"  Filtered {n_before - n_after}/{n_before} events with invalid labels")

        if n_after == 0:
            raise ValueError(
                f"No valid events found for num_jets={num_jets}. "
                f"The data may not contain enough jets per event "
                f"(need >={num_jets}). Check that your data has ISR jets "
                f"if using num_jets=7."
            )

        if self.normalize_by_ht:
            ht_expanded = self.ht.unsqueeze(-1).unsqueeze(-1)
            # Normalize all four-vector dims (E, px, py, pz) by event HT
            self.four_momenta = self.four_momenta / ht_expanded.clamp(min=1e-6)

    def _load_file(self, fpath: str):
        """Load a single HDF5 file and return processed tensors."""
        with h5py.File(fpath, "r") as f:
            # Detect data layout
            has_targets = "TARGETS" in f and "g1" in f["TARGETS"]
            has_source_input = "INPUTS" in f and "Source" in f["INPUTS"]

            # Read jet kinematics
            if has_source_input:
                pt, eta, phi, mass, mask = self._read_inputs_source(f)
            else:
                pt, eta, phi, mass, mask = self._read_jet_features(f)

            n_events = pt.shape[0]
            max_jets_in_file = pt.shape[1]

            # Apply pT smearing if requested (simulates detector resolution)
            if self.pt_smear_frac > 0:
                rng = np.random.RandomState(seed=12345)
                smear = 1.0 + rng.normal(0, self.pt_smear_frac, size=pt.shape).astype(np.float32)
                smear = np.clip(smear, 0.5, 1.5)  # Prevent extreme outliers
                pt = pt * smear * mask  # Only smear valid jets

            # Count valid jets per event
            n_valid = mask.sum(axis=1)

            # Determine actual num_jets to use per event
            effective_num_jets = min(self.num_jets, max_jets_in_file)

            # pT-sort jets (descending) and build reindex mapping
            sorted_four_mom, sorted_mask, sort_indices = self._pt_sort_and_select(
                pt, eta, phi, mass, mask, effective_num_jets
            )

            # Read truth labels
            if self.use_mass_asymmetry_labels:
                # Primary path: compute labels as argmin |m1-m2| over all
                # interpretations from the raw four-momenta.  This is the
                # physically correct ground truth and is immune to incorrect
                # TARGETS entries in the HDF5 file.
                labels, parent_mass_arr = self._compute_mass_asymmetry_labels(
                    sorted_four_mom, effective_num_jets
                )
            elif has_targets:
                labels, parent_mass_arr = self._read_targets(
                    f, sort_indices, n_valid, effective_num_jets
                )
            else:
                labels, parent_mass_arr = self._read_truth_from_columns(
                    f, sort_indices, n_valid, effective_num_jets
                )

            # Compute HT (sum of valid jet pTs)
            ht = np.zeros(n_events, dtype=np.float32)
            for i in range(n_events):
                valid_pts = pt[i, mask[i]]
                ht[i] = valid_pts.sum() if len(valid_pts) > 0 else 1.0

            # Compute parent mass from truth grouping if not already set
            if parent_mass_arr is None:
                parent_mass_arr = np.zeros(n_events, dtype=np.float32)

            four_mom_t = torch.tensor(sorted_four_mom, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.long)
            parent_mass_t = torch.tensor(parent_mass_arr, dtype=torch.float32)
            ht_t = torch.tensor(ht, dtype=torch.float32)

            return four_mom_t, labels_t, parent_mass_t, ht_t

    def _read_inputs_source(self, f):
        """Read kinematics from INPUTS/Source group."""
        pt = f["INPUTS/Source/pt"][:]
        eta = f["INPUTS/Source/eta"][:]
        phi = f["INPUTS/Source/phi"][:]
        mass = f["INPUTS/Source/mass"][:]
        mask = f["INPUTS/Source/MASK"][:].astype(bool)
        return pt, eta, phi, mass, mask

    def _read_jet_features(self, f):
        """Read kinematics from jet_features dataset."""
        jf = f["jet_features"][:]
        pt = jf[:, :, 0]
        eta = jf[:, :, 1]
        phi = jf[:, :, 2]
        mass = jf[:, :, 3]
        if "jet_mask" in f:
            mask = f["jet_mask"][:].astype(bool)
        else:
            mask = pt > 0
        return pt, eta, phi, mass, mask

    def _pt_sort_and_select(self, pt, eta, phi, mass, mask, num_jets):
        """pT-sort jets descending and select top num_jets.

        Returns:
            sorted_four_mom: (N, num_jets, 4) in (E, px, py, pz)
            sorted_mask: (N, num_jets) bool
            sort_indices: (N, num_jets) original indices of selected jets
        """
        n_events = pt.shape[0]
        max_jets = pt.shape[1]

        sorted_four_mom = np.zeros((n_events, num_jets, 4), dtype=np.float32)
        sorted_mask = np.zeros((n_events, num_jets), dtype=bool)
        sort_indices = np.full((n_events, num_jets), -1, dtype=np.int64)

        for i in range(n_events):
            # Get valid jet indices sorted by pT descending
            valid_idx = np.where(mask[i])[0]
            if len(valid_idx) == 0:
                continue

            pts_valid = pt[i, valid_idx]
            order = np.argsort(-pts_valid)  # descending
            selected = valid_idx[order[:num_jets]]

            n_selected = len(selected)
            sort_indices[i, :n_selected] = selected
            sorted_mask[i, :n_selected] = True

            # Convert to (E, px, py, pz)
            four_mom = pt_eta_phi_mass_to_epxpypz(
                pt[i, selected], eta[i, selected],
                phi[i, selected], mass[i, selected],
            )
            sorted_four_mom[i, :n_selected] = four_mom

        return sorted_four_mom, sorted_mask, sort_indices

    @staticmethod
    def _compute_mass_asymmetry_labels(
        sorted_four_mom: np.ndarray, num_jets: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute truth labels as argmin of |m1 - m2| over all assignments.

        For each event, enumerates every valid split of the ``num_jets`` pT-sorted
        jets into (ISR, triplet1, triplet2) and selects the one that minimises the
        absolute mass difference |m1 - m2|.

        With no detector smearing and no jet fragmentation the physically correct
        interpretation gives m1 ≈ m2 ≈ M_parent while any wrong-ISR interpretation
        yields a much larger |m1 - m2| (the triplet that absorbs the ISR jet has a
        very different invariant mass from the signal triplet).

        Args:
            sorted_four_mom: ``(N, num_jets, 4)`` array of ``(E, px, py, pz)`` in
                physical units (GeV), *before* any HT normalisation.
            num_jets: Number of jets per event (6 or 7).

        Returns:
            labels: ``(N,)`` int64 array of assignment indices.
            parent_masses: ``(N,)`` float32 array of estimated parent masses in GeV,
                computed as ``(m1 + m2) / 2`` for the best-scoring assignment.
        """
        at = build_assignment_tensors(num_jets)
        g1_idx = at["group1_indices"].detach().cpu().numpy()  # (N_assign, 3)
        g2_idx = at["group2_indices"].detach().cpu().numpy()  # (N_assign, 3)

        # Gather group four-momenta for every assignment simultaneously.
        # sorted_four_mom[:, g1_idx, :] → (N, N_assign, 3, 4)
        g1_sum = sorted_four_mom[:, g1_idx, :].sum(axis=2)  # (N, N_assign, 4)
        g2_sum = sorted_four_mom[:, g2_idx, :].sum(axis=2)  # (N, N_assign, 4)

        def inv_mass(v: np.ndarray) -> np.ndarray:
            m2 = v[..., 0] ** 2 - v[..., 1] ** 2 - v[..., 2] ** 2 - v[..., 3] ** 2
            return np.sqrt(np.maximum(m2, 0.0))

        m1 = inv_mass(g1_sum)  # (N, N_assign)
        m2 = inv_mass(g2_sum)  # (N, N_assign)

        mass_diff = np.abs(m1 - m2)  # (N, N_assign)
        labels = mass_diff.argmin(axis=1).astype(np.int64)  # (N,)

        # Estimate parent mass from winning assignment: average of the two triplet masses
        n = len(labels)
        best_m1 = m1[np.arange(n), labels]
        best_m2 = m2[np.arange(n), labels]
        parent_masses = ((best_m1 + best_m2) / 2.0).astype(np.float32)

        return labels, parent_masses

    def _read_targets(self, f, sort_indices, n_valid, num_jets):
        """Read truth from TARGETS/g1 and TARGETS/g2 groups.

        Maps truth jet indices through the pT-sort permutation.
        """
        n_events = sort_indices.shape[0]

        g1_orig = np.stack([
            f["TARGETS/g1/j1"][:],
            f["TARGETS/g1/j2"][:],
            f["TARGETS/g1/j3"][:],
        ], axis=1)  # (N, 3)

        g2_orig = np.stack([
            f["TARGETS/g2/j1"][:],
            f["TARGETS/g2/j2"][:],
            f["TARGETS/g2/j3"][:],
        ], axis=1)  # (N, 3)

        labels = np.full(n_events, -1, dtype=np.int64)
        parent_mass = np.zeros(n_events, dtype=np.float32)

        # Read source 4-vectors for mass computation
        has_source_e = "source" in f and "e" in f["source"]
        if has_source_e:
            src_e = f["source/e"][:]
            src_pt = f["source/pt"][:]
            src_eta = f["source/eta"][:]
            src_phi = f["source/phi"][:]
        elif "INPUTS" in f and "Source" in f["INPUTS"]:
            src_pt = f["INPUTS/Source/pt"][:]
            src_eta = f["INPUTS/Source/eta"][:]
            src_phi = f["INPUTS/Source/phi"][:]
            src_mass = f["INPUTS/Source/mass"][:]
            # Compute E from pt, eta, mass
            src_px = src_pt * np.cos(src_phi)
            src_py = src_pt * np.sin(src_phi)
            src_pz = src_pt * np.sinh(src_eta)
            src_e = np.sqrt(src_px**2 + src_py**2 + src_pz**2 + src_mass**2)

        n_signal_outside = 0
        n_not_enough_jets = 0
        n_wrong_isr_count = 0

        for i in range(n_events):
            # Map original truth indices to sorted positions
            idx_map = {}
            for new_pos in range(num_jets):
                orig_idx = sort_indices[i, new_pos]
                if orig_idx >= 0:
                    idx_map[int(orig_idx)] = new_pos

            # Remap group indices
            g1_sorted = []
            g2_sorted = []
            valid = True

            for j in g1_orig[i]:
                if int(j) in idx_map:
                    g1_sorted.append(idx_map[int(j)])
                else:
                    valid = False
                    break

            if valid:
                for j in g2_orig[i]:
                    if int(j) in idx_map:
                        g2_sorted.append(idx_map[int(j)])
                    else:
                        valid = False
                        break

            if not valid or len(g1_sorted) != 3 or len(g2_sorted) != 3:
                n_signal_outside += 1
                continue

            # Determine ISR: any jet in the sorted list not in g1 or g2
            all_assigned = set(g1_sorted + g2_sorted)
            isr_jets = [j for j in range(num_jets) if j not in all_assigned and sort_indices[i, j] >= 0]
            # Also consider zero-padded slots as ISR candidates (events with fewer jets)
            padded_slots = [j for j in range(num_jets) if j not in all_assigned and sort_indices[i, j] < 0]

            if num_jets == 6:
                truth_isr = None
            elif len(isr_jets) == 1:
                truth_isr = isr_jets[0]
            elif len(isr_jets) == 0 and len(padded_slots) == 1:
                truth_isr = padded_slots[0]
            elif len(isr_jets) == 0 and num_jets > 6:
                n_not_enough_jets += 1
                continue
            else:
                n_wrong_isr_count += 1
                continue

            labels[i] = match_truth_groups(g1_sorted, g2_sorted, num_jets, truth_isr)

            # Compute parent mass from original 4-vectors
            if has_source_e or "INPUTS" in f:
                g1_4vec = np.zeros(4)
                for j in g1_orig[i]:
                    g1_4vec[0] += src_e[i, j]
                    px = src_pt[i, j] * np.cos(src_phi[i, j])
                    py = src_pt[i, j] * np.sin(src_phi[i, j])
                    pz = src_pt[i, j] * np.sinh(src_eta[i, j])
                    g1_4vec[1] += px
                    g1_4vec[2] += py
                    g1_4vec[3] += pz
                m2 = g1_4vec[0]**2 - g1_4vec[1]**2 - g1_4vec[2]**2 - g1_4vec[3]**2
                parent_mass[i] = np.sqrt(max(m2, 0))

        n_valid = (labels >= 0).sum()
        n_invalid = n_events - n_valid
        if n_invalid > 0:
            print(f"  Label failures ({n_invalid}/{n_events}):")
            if n_signal_outside > 0:
                print(f"    Signal jet outside top {num_jets} by pT: {n_signal_outside}")
            if n_not_enough_jets > 0:
                print(f"    Fewer than {num_jets} valid jets: {n_not_enough_jets}")
            if n_wrong_isr_count > 0:
                print(f"    Multiple ISR candidates: {n_wrong_isr_count}")

        return labels, parent_mass

    def _read_truth_from_columns(self, f, sort_indices, n_valid, num_jets):
        """Fallback: read truth from jet_features columns (is_signal, parent_pdg).

        Used when TARGETS group is not present.
        """
        jf = f["jet_features"][:]
        n_events = jf.shape[0]

        labels = np.full(n_events, -1, dtype=np.int64)
        parent_mass = np.zeros(n_events, dtype=np.float32)

        for i in range(n_events):
            # Get the original indices of jets in our sorted selection
            orig_indices = sort_indices[i]
            valid_count = (orig_indices >= 0).sum()

            if valid_count < 6:
                continue

            # Extract is_signal (col 5) and parent_pdg (col 4) for selected jets
            is_sig = np.array([jf[i, orig_indices[j], 5] if orig_indices[j] >= 0 else 0
                              for j in range(num_jets)])
            p_ids = np.array([jf[i, orig_indices[j], 4] if orig_indices[j] >= 0 else 0
                             for j in range(num_jets)])

            # Find signal jets and group by parent
            signal_jets = np.where(is_sig == 1)[0]
            if len(signal_jets) < 6:
                continue

            parent_to_jets = {}
            for j in signal_jets:
                pid = int(p_ids[j])
                if pid not in parent_to_jets:
                    parent_to_jets[pid] = []
                parent_to_jets[pid].append(j)

            groups = list(parent_to_jets.values())
            if len(groups) != 2 or any(len(g) != 3 for g in groups):
                continue

            isr_jets = [j for j in range(num_jets) if is_sig[j] == 0 and orig_indices[j] >= 0]

            if num_jets == 6:
                truth_isr = None
            elif len(isr_jets) == 1:
                truth_isr = isr_jets[0]
            else:
                continue

            labels[i] = match_truth_groups(groups[0], groups[1], num_jets, truth_isr)

        return labels, parent_mass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "four_momenta": self.four_momenta[idx],
            "label": self.labels[idx],
            "parent_mass": self.parent_masses[idx],
        }
