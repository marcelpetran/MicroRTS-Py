"""Shared hindsight (claim-count) labeling for team-based exploration.

Used by BOTH the RL agent (QLearningAgent._apply_hindsight_relabeling) and
offline data collection (omexplore.collect_data.collect_team_offline_data),
so the pretraining labels are guaranteed consistent with the labels the OMs
see at RL fine-tuning time.

Semantics: for every agent, the intended subgoal at step t is the next goal
it collects after t (hindsight); agents that never collect are labeled with
their final position (truncated-episode heuristic). Aggregated per team this
yields claim-COUNT maps (a cell can hold > 1 when several agents aim at the
same goal).
"""

from typing import Dict, List

import numpy as np


def compute_agent_goals(step_records: List[Dict], final_positions: Dict) -> List[Dict]:
    """Backward pass over an episode's collection records.

    Args:
        step_records: per-step dicts {"collectors": {goal: [agent_ids]}}.
        final_positions: agent -> (r, c) at episode end (fallback goal).

    Returns:
        agent_goals[t] = {agent: goal} — the next goal each agent collects
        at step t or later (final position if it never collects).
    """
    next_goal = dict(final_positions)
    agent_goals: List[Dict] = [None] * len(step_records)  # type: ignore[list-item]
    for t in range(len(step_records) - 1, -1, -1):
        for goal, collectors in step_records[t]["collectors"].items():
            for a in collectors:
                next_goal[a] = goal
        agent_goals[t] = dict(next_goal)
    return agent_goals


def claim_count_map(agent_ids: List, goals: Dict, H: int, W: int):
    """Dense (H, W) float32 claim-count map for the given agents."""
    m = np.zeros((H, W), dtype=np.float32)
    for a in agent_ids:
        p = goals[a]
        m[p[0], p[1]] += 1.0
    return m


def sparse_claims(agent_ids: List, goals: Dict) -> List[tuple]:
    """Claim counts as a sparse list [(r, c, weight), ...] (offline storage)."""
    counts: Dict = {}
    for a in agent_ids:
        p = goals[a]
        counts[p] = counts.get(p, 0.0) + 1.0
    return [(p[0], p[1], w) for p, w in counts.items()]


def dense_from_sparse(cells: List[List[tuple]], H: int, W: int):
    """Batch of sparse cell lists -> dense (B, H, W) float32 array."""
    m = np.zeros((len(cells), H, W), dtype=np.float32)
    for i, cell_list in enumerate(cells):
        for r, c, w in cell_list:
            m[i, r, c] += w
    return m
