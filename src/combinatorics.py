"""
Enumerate all possible jet-to-parent assignments and provide truth matching.

Supports two modes:
  - 6 jets (no ISR): C(6,3)/2 = 10 assignments — partition into two groups of 3
  - 7 jets (with ISR): C(7,1) * C(6,3)/2 = 70 assignments — pick ISR + partition

Also supports a factored decomposition for 7-jet mode:
  - ISR identification: 7-way classification
  - Grouping (given ISR): 10-way classification
  - Combined: assignment_idx = isr_idx * 10 + grouping_idx
"""

from itertools import combinations

import torch


def enumerate_assignments(num_jets: int = 7) -> list[tuple[int | None, tuple[int, ...], tuple[int, ...]]]:
    """Enumerate all (isr_idx, group1, group2) assignments.

    The two groups are unordered (symmetric parents), so we canonicalize
    by placing the group with the smaller first index first.

    For 6 jets: isr_idx is None, returns 10 assignments.
    For 7 jets: isr_idx is 0-6, returns 70 assignments.

    Returns:
        List of tuples: (isr_index_or_None, group1_indices, group2_indices)
    """
    assignments = []

    if num_jets == 6:
        # No ISR — just partition 6 jets into two groups of 3
        all_jets = list(range(6))
        seen = set()
        for group1 in combinations(all_jets, 3):
            group2 = tuple(j for j in all_jets if j not in group1)
            canon = (min(group1, group2), max(group1, group2))
            if canon not in seen:
                seen.add(canon)
                assignments.append((None, canon[0], canon[1]))
        return assignments

    # 7+ jets: choose ISR then partition remaining 6
    for isr in range(num_jets):
        remaining = [j for j in range(num_jets) if j != isr]
        seen = set()
        for group1 in combinations(remaining, 3):
            group2 = tuple(j for j in remaining if j not in group1)
            canon = (min(group1, group2), max(group1, group2))
            if canon not in seen:
                seen.add(canon)
                assignments.append((isr, canon[0], canon[1]))
    return assignments


def build_assignment_tensors(num_jets: int = 7) -> dict[str, torch.Tensor]:
    """Build index tensors for efficient batched gathering.

    Returns dict with:
        - isr_indices: (N,) int tensor of ISR jet index (-1 if no ISR)
        - group1_indices: (N, 3) int tensor of group1 jet indices
        - group2_indices: (N, 3) int tensor of group2 jet indices
        - num_assignments: int
    """
    assignments = enumerate_assignments(num_jets)
    isr = torch.tensor(
        [a[0] if a[0] is not None else -1 for a in assignments],
        dtype=torch.long,
    )
    g1 = torch.tensor([list(a[1]) for a in assignments], dtype=torch.long)
    g2 = torch.tensor([list(a[2]) for a in assignments], dtype=torch.long)
    return {
        "isr_indices": isr,
        "group1_indices": g1,
        "group2_indices": g2,
        "num_assignments": len(assignments),
    }


def build_factored_tensors(num_jets: int = 7) -> dict[str, torch.Tensor]:
    """Build index tensors for the factored (ISR + grouping) decomposition.

    For each ISR candidate j (0..6), enumerate the 10 groupings of the
    remaining 6 jets. Returns tensors indexed as [isr_idx, grouping_idx].

    Returns dict with:
        - group1_indices: (num_jets, 10, 3) — for each ISR choice, 10 groupings
        - group2_indices: (num_jets, 10, 3)
        - num_groupings: 10
        - flat_to_factored: (70, 2) — maps flat assignment idx to (isr_idx, grouping_idx)
        - factored_to_flat: (num_jets, 10) — maps (isr_idx, grouping_idx) to flat idx
    """
    assignments = enumerate_assignments(num_jets)

    g1_all = []  # (num_jets, 10, 3)
    g2_all = []

    for isr in range(num_jets):
        remaining = [j for j in range(num_jets) if j != isr]
        seen = set()
        g1_for_isr = []
        g2_for_isr = []
        for group1 in combinations(remaining, 3):
            group2 = tuple(j for j in remaining if j not in group1)
            canon = (min(group1, group2), max(group1, group2))
            if canon not in seen:
                seen.add(canon)
                g1_for_isr.append(list(canon[0]))
                g2_for_isr.append(list(canon[1]))
        g1_all.append(g1_for_isr)
        g2_all.append(g2_for_isr)

    n_groupings = len(g1_all[0])  # 10

    # Build flat <-> factored mappings
    flat_to_factored = torch.zeros(len(assignments), 2, dtype=torch.long)
    factored_to_flat = torch.full((num_jets, n_groupings), -1, dtype=torch.long)

    for flat_idx, (a_isr, a_g1, a_g2) in enumerate(assignments):
        isr_idx = a_isr
        # Find which grouping index this corresponds to
        for gi in range(n_groupings):
            if tuple(g1_all[isr_idx][gi]) == tuple(a_g1) and tuple(g2_all[isr_idx][gi]) == tuple(a_g2):
                flat_to_factored[flat_idx] = torch.tensor([isr_idx, gi])
                factored_to_flat[isr_idx, gi] = flat_idx
                break

    return {
        "group1_indices": torch.tensor(g1_all, dtype=torch.long),  # (7, 10, 3)
        "group2_indices": torch.tensor(g2_all, dtype=torch.long),  # (7, 10, 3)
        "num_groupings": n_groupings,
        "flat_to_factored": flat_to_factored,  # (70, 2)
        "factored_to_flat": factored_to_flat,  # (7, 10)
    }


def match_truth_groups(
    truth_g1: list[int],
    truth_g2: list[int],
    num_jets: int,
    truth_isr: int | None = None,
) -> int:
    """Find the assignment index matching truth group indices.

    Args:
        truth_g1: List of 3 jet indices for group 1.
        truth_g2: List of 3 jet indices for group 2.
        num_jets: Total number of jets (6 or 7).
        truth_isr: ISR jet index (None for 6-jet events).

    Returns:
        Assignment index, or -1 if no match.
    """
    assignments = enumerate_assignments(num_jets)
    g1 = tuple(sorted(truth_g1))
    g2 = tuple(sorted(truth_g2))
    # Canonical: smaller first element first
    canon = (min(g1, g2), max(g1, g2))

    for idx, (a_isr, a_g1, a_g2) in enumerate(assignments):
        isr_match = (truth_isr is None and a_isr is None) or (a_isr == truth_isr)
        if isr_match and (a_g1, a_g2) == canon:
            return idx

    return -1
