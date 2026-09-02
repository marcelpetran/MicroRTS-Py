"""Unit test: team claim-count hindsight labeling (synthetic scenario).

Scenario (team_sizes=(2,2), learn_ids=[0,1], hostile_ids=[2,3]):
  - hostile agent 2 collects g1 at step 1 and g2 at step 3, then stops there
    (final position p2 = g2's location)
  - hostile agent 3 never collects (final position p3)
  - teammate agent 1 never collects (final position p1)

Expected hostile claim map per step (acting agent = 0):
  steps 0-1: {g1: 1, p3: 1}   (agent 2 is heading to g1, agent 3 to its final pos)
  steps 2-3: {g2: 1, p3: 1}
  step 4   : {p2: 1, p3: 1}   (post-last-collection: final-position fallback)

Expected friendly claim map at every step: {p1: 1} (agent 1 never collects).

Run: /opt/homebrew/anaconda3/envs/om/bin/python tests/test_hindsight_labeling.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import wandb
from omexplore.agents.q_agent import QLearningAgent
from omexplore.envs.roadmap_foraging_env import TeamRoadmapEnv
from omexplore.models.opponent_model import OpponentModel
from omexplore.models.transformers import SpatialOpponentModel
from omexplore.utils.omg_args import OMGArgs


def cell_counts(m):
    """Map -> dict {(r, c): value} for nonzero cells."""
    return {
        (int(r), int(c)): float(m[r, c]) for r, c in zip(*np.nonzero(m)) if m[r, c] != 0
    }


def main():
    wandb.init(mode="disabled")

    env = TeamRoadmapEnv(max_steps=5, num_goals=4, team_sizes=(2, 2))
    args = OMGArgs(
        device="cpu",
        state_shape=(env.height, env.width, env.features),
        H=env.height,
        W=env.width,
        action_dim=8,
        max_steps=5,
        max_history_length=4,
        capacity=100,
        min_replay=100,
        batch_size=4,
        train_every=1000,
    )
    hostile_om = OpponentModel(SpatialOpponentModel(args), args)
    friendly_om = OpponentModel(SpatialOpponentModel(args), args)
    agent = QLearningAgent(env, hostile_om, friendly_om, args=args)
    assert agent.learn_ids == [0, 1] and agent.hostile_ids == [2, 3]

    H, W = env.height, env.width
    g1 = (10, 11)
    g2 = (20, 22)
    p2 = g2  # agent 2 ends where it made its last collection
    p3 = (30, 33)
    p1 = (5, 6)

    # 5 transitions for acting agent 0 (one per step)
    episode_transitions = [{"agent_id": 0, "step_idx": t} for t in range(5)]
    # collectors recorded by the env at each step: goal -> agent ids
    step_records = [
        {"collectors": {}},  # step 0
        {"collectors": {g1: [2]}},  # step 1: hostile 2 takes g1
        {"collectors": {}},  # step 2
        {"collectors": {g2: [2]}},  # step 3: hostile 2 takes g2
        {"collectors": {}},  # step 4
    ]
    final_positions = {0: (1, 2), 1: p1, 2: p2, 3: p3}

    agent._apply_hindsight_relabeling(
        episode_transitions, step_records, final_positions, H, W
    )

    expected_hostile = [
        {g1: 1.0, p3: 1.0},  # step 0
        {g1: 1.0, p3: 1.0},  # step 1 (label applies from this step onward)
        {g2: 1.0, p3: 1.0},  # step 2
        {g2: 1.0, p3: 1.0},  # step 3
        {p2: 1.0, p3: 1.0},  # step 4: final-position fallback for agent 2
    ]
    for t, tr in enumerate(episode_transitions):
        hm = cell_counts(tr["true_goal_map"])
        tm = cell_counts(tr["true_team_goal_map"])
        assert hm == expected_hostile[t], (
            f"step {t}: hostile map {hm} != {expected_hostile[t]}"
        )
        assert tm == {p1: 1.0}, f"step {t}: friendly map {tm} != {{p1: 1.0}}"
        # invariants
        assert tr["true_goal_map"].sum() == 2.0
        assert tr["true_team_goal_map"].sum() == 1.0
        assert tr["true_goal_map"].dtype == np.float32
        assert tr["true_team_goal_map"].shape == (H, W)

    # --- Extra: claim-count collision (both hostiles aim at the same goal).
    tr2 = [{"agent_id": 0, "step_idx": 0}]
    step_records2 = [{"collectors": {g1: [2, 3]}}]
    final_positions2 = {0: (1, 2), 1: p1, 2: p2, 3: p3}
    agent._apply_hindsight_relabeling(tr2, step_records2, final_positions2, H, W)
    hm2 = cell_counts(tr2[0]["true_goal_map"])
    assert hm2 == {g1: 2.0}, f"claim-count collision: {hm2} != {{g1: 2.0}}"

    # --- Empty records: labels should be all zeros.
    tr3 = [{"agent_id": 0, "step_idx": 0}]
    agent._apply_hindsight_relabeling(tr3, [], final_positions, H, W)
    assert tr3[0]["true_goal_map"].sum() == 0.0
    assert tr3[0]["true_team_goal_map"].sum() == 0.0

    print("ALL HINDSIGHT LABELING UNIT TESTS PASSED")


if __name__ == "__main__":
    main()
