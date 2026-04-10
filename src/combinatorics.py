"""
Enumerate all possible jet-to-parent assignments and provide truth matching.

With 7 jets, we choose 1 as ISR and partition the remaining 6 into two
symmetric groups of 3. Total: C(7,1) * C(6,3)/2 = 70 assignments.
"""

from itertools import combinations

import torch


def enumerate_assignments(num_jets: int = 7) -> list[tuple[int, tuple[int, ...], tuple[int, ...]]]:
    """Enumerate all (isr_idx, group1, group2) assignments.

    The two groups are unordered (symmetric parents), so we canonicalize
    by placing the group with the smaller first index first.

    Returns:
        List of 70 tuples: (isr_index, group1_indices, group2_indices)
    """
    assignments = []
    for isr in range(num_jets):
        remaining = [j for j in range(num_jets) if j != isr]
        seen = set()
        for group1 in combinations(remaining, 3):
            group2 = tuple(j for j in remaining if j not in group1)
            # Canonical form: group with smaller first element comes first
            canon = (min(group1, group2), max(group1, group2))
            if canon not in seen:
                seen.add(canon)
                assignments.append((isr, canon[0], canon[1]))
    return assignments


def build_assignment_tensors(num_jets: int = 7) -> dict[str, torch.Tensor]:
    """Build index tensors for efficient batched gathering.

    Returns dict with:
        - isr_indices: (70,) int tensor of ISR jet index per assignment
        - group1_indices: (70, 3) int tensor of group1 jet indices
        - group2_indices: (70, 3) int tensor of group2 jet indices
    """
    assignments = enumerate_assignments(num_jets)
    isr = torch.tensor([a[0] for a in assignments], dtype=torch.long)
    g1 = torch.tensor([list(a[1]) for a in assignments], dtype=torch.long)
    g2 = torch.tensor([list(a[2]) for a in assignments], dtype=torch.long)
    return {"isr_indices": isr, "group1_indices": g1, "group2_indices": g2}


def match_truth_to_assignment(
    is_signal: torch.Tensor,
    parent_ids: torch.Tensor,
    num_jets: int = 7,
) -> torch.Tensor:
    """Find the assignment index matching the truth labels for a batch.

    Args:
        is_signal: (batch, num_jets) bool/float — 1 if jet is from signal
        parent_ids: (batch, num_jets) int/float — parent PDG ID per jet

    Returns:
        (batch,) int tensor — index into the 70 assignments, or -1 if no match
    """
    assignments = enumerate_assignments(num_jets)
    batch_size = is_signal.shape[0]
    result = torch.full((batch_size,), -1, dtype=torch.long)

    for b in range(batch_size):
        sig = is_signal[b]
        pids = parent_ids[b]

        # Identify ISR jets (is_signal == 0)
        isr_mask = sig == 0
        signal_mask = sig == 1

        # Find unique parent IDs among signal jets
        signal_pids = pids[signal_mask.bool() if not signal_mask.dtype == torch.bool else signal_mask]
        unique_parents = signal_pids.unique()
        unique_parents = unique_parents[unique_parents != 0]  # filter padding

        if signal_mask.sum() < 6 or len(unique_parents) < 2:
            # Best-effort: try to find the best matching assignment
            result[b] = _best_effort_match(sig, pids, assignments, num_jets)
            continue

        # Build truth groups: jets belonging to each parent
        parent_to_jets = {}
        for j in range(num_jets):
            if sig[j] == 1:
                pid = int(pids[j].item())
                if pid not in parent_to_jets:
                    parent_to_jets[pid] = []
                parent_to_jets[pid].append(j)

        groups = list(parent_to_jets.values())
        if len(groups) != 2 or any(len(g) != 3 for g in groups):
            result[b] = _best_effort_match(sig, pids, assignments, num_jets)
            continue

        truth_g1 = tuple(sorted(groups[0]))
        truth_g2 = tuple(sorted(groups[1]))
        # Canonical: smaller first element first
        truth_canon = (min(truth_g1, truth_g2), max(truth_g1, truth_g2))

        # Find ISR jet
        isr_jets = [j for j in range(num_jets) if sig[j] == 0]
        if len(isr_jets) != 1:
            result[b] = _best_effort_match(sig, pids, assignments, num_jets)
            continue

        truth_isr = isr_jets[0]

        # Match against enumerated assignments
        for idx, (a_isr, a_g1, a_g2) in enumerate(assignments):
            if a_isr == truth_isr and (a_g1, a_g2) == truth_canon:
                result[b] = idx
                break

    return result


def _best_effort_match(
    sig: torch.Tensor,
    pids: torch.Tensor,
    assignments: list,
    num_jets: int,
) -> int:
    """Best-effort matching when truth is imperfect.

    Scores each assignment by how well it agrees with available truth info:
    - ISR jet should have is_signal=0 (or lowest signal confidence)
    - Jets in the same group should share parent_id where known
    """
    best_score = -1
    best_idx = 0

    for idx, (a_isr, a_g1, a_g2) in enumerate(assignments):
        score = 0

        # Reward: ISR jet is not signal
        if sig[a_isr] == 0:
            score += 10

        # Reward: jets within each group share the same parent_id
        g1_pids = [int(pids[j].item()) for j in a_g1 if sig[j] == 1]
        g2_pids = [int(pids[j].item()) for j in a_g2 if sig[j] == 1]

        if len(g1_pids) > 0 and len(set(g1_pids)) == 1:
            score += len(g1_pids)
        if len(g2_pids) > 0 and len(set(g2_pids)) == 1:
            score += len(g2_pids)

        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx
