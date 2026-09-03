"""Quick check: friendly_om=False ablation path (Q-net hostile-map only)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import wandb
from omexplore.agents.q_agent import QLearningAgent
from omexplore.agents.team_agents import TeamAgent
from omexplore.envs.roadmap_foraging_env import TeamRoadmapEnv
from omexplore.models.opponent_model import OpponentModel
from omexplore.models.transformers import SpatialOpponentModel
from omexplore.utils.omg_args import OMGArgs


def main():
    wandb.init(mode="disabled")
    torch.manual_seed(0)

    env = TeamRoadmapEnv(max_steps=40, num_goals=6, team_sizes=(2, 2))
    args = OMGArgs(
        device="cpu",
        state_shape=(env.height, env.width, env.features),
        H=env.height,
        W=env.width,
        action_dim=8,
        max_steps=40,
        max_history_length=4,
        capacity=500,
        min_replay=30,
        batch_size=8,
        train_every=2,
        friendly_om=False,
    )
    hostile_om = OpponentModel(SpatialOpponentModel(args), args)
    friendly_om = OpponentModel(SpatialOpponentModel(args), args)
    agent = QLearningAgent(env, hostile_om, friendly_om, args=args)
    opp = TeamAgent(env, team_id=1)

    for _ in range(3):
        s = agent.run_episode(opp, max_steps=40)
    print("ablation ep:", s)
    assert s["avg_q_loss"] > 0
    assert s["avg_model_loss"] > 0
    assert s["avg_team_model_loss"] > 0

    n_in = agent.q.cnn[0].in_channels
    exp = env.features + args.belief_channels + 1
    assert n_in == exp, (n_in, exp)
    print(f"friendly_om=False: QNet input channels = {n_in} (expected {exp}) OK")

    # And with friendly_om=True -> +2 channels
    args2 = OMGArgs(
        device="cpu",
        state_shape=(env.height, env.width, env.features),
        H=env.height,
        W=env.width,
        action_dim=8,
        max_steps=40,
        max_history_length=4,
        capacity=500,
        min_replay=30,
        batch_size=8,
        train_every=2,
        friendly_om=True,
    )
    hostile2 = OpponentModel(SpatialOpponentModel(args2), args2)
    friendly2 = OpponentModel(SpatialOpponentModel(args2), args2)
    agent2 = QLearningAgent(env, hostile2, friendly2, args=args2)
    n_in2 = agent2.q.cnn[0].in_channels
    exp2 = env.features + args2.belief_channels + 2
    assert n_in2 == exp2, (n_in2, exp2)
    print(f"friendly_om=True:  QNet input channels = {n_in2} (expected {exp2}) OK")

    print("ABLATION WIRING OK")


if __name__ == "__main__":
    main()
