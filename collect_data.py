import random
from typing import Dict, List

import numpy as np
import torch

from maps import *
from omg_args import OMGArgs
from renderer import RealtimeRenderer
from simple_foraging_env import (
    ChameleonAgent,
    GreedySwitchAgent,
    SimpleAgent,
    SimpleForagingEnv,
    StalkerAgent,
)


def _apply_hindsight_relabeling(episode_transitions: List, H: int, W: int):
    """
    Applies Hindsight Experience Replay (HER) labeling to a trajectory.
    Modifies the transitions in-place to include 'true_goal_map'.
    """
    current_true_goal_pos = None

    if len(episode_transitions) > 0:
        final_t = episode_transitions[-1]

        if final_t["opp_reward"] == 0:
            opp_pos_arr = np.argwhere(final_t["global_state"][:, :, 3] == 1)
            if len(opp_pos_arr) > 0:
                current_true_goal_pos = tuple(opp_pos_arr[0])

    for t in reversed(episode_transitions):
        if t["opp_reward"] > 0:
            opp_pos_indices = np.argwhere(t["next_global_state"][:, :, 3] == 1)
            if len(opp_pos_indices) > 0:
                current_true_goal_pos = tuple(opp_pos_indices[0])

        true_map = np.zeros((H, W), dtype=np.float32)
        if current_true_goal_pos is not None:
            true_map[current_true_goal_pos[0], current_true_goal_pos[1]] = 1.0

        t["true_goal_map"] = true_map

        del t["opp_reward"]


def collect_offline_data(
    num_episodes: int = 1000,
    save_path: str = "./dataset/dataset.pt",
    map_layout: list[str] = MAP_1,
    om_args: OMGArgs = OMGArgs(),
):
    args = om_args
    env = SimpleForagingEnv(max_steps=args.max_steps, map_layout=map_layout)
    obs = env.reset()
    agent_0 = SimpleAgent(0, map_layout=map_layout)
    # precompute paths for other agents to use during data collection
    # Dummy action to trigger path precomputation in the environment
    _ = agent_0.select_action(obs[0])
    precomputed_paths = agent_0.precomputed_paths
    agent_1 = SimpleAgent(1, precomputed_paths=precomputed_paths, map_layout=map_layout)
    agent_2 = GreedySwitchAgent(
        0, precomputed_paths=precomputed_paths, map_layout=map_layout
    )
    agent_3 = GreedySwitchAgent(
        1, precomputed_paths=precomputed_paths, map_layout=map_layout
    )
    agent_4 = StalkerAgent(
        0, precomputed_paths=precomputed_paths, map_layout=map_layout
    )
    agent_5 = StalkerAgent(
        1, precomputed_paths=precomputed_paths, map_layout=map_layout
    )
    agent_combinations = [
        (agent_0, agent_1),
        (agent_0, agent_3),
        (agent_2, agent_1),
        (agent_2, agent_3),
    ]

    master_dataset = []

    print(f"Starting offline data collection for {num_episodes} episodes...")
    for agent0, agent1 in agent_combinations:
        print(
            f"Collecting data for agent combination: {agent0.__class__.__name__} vs {agent1.__class__.__name__}"
        )
        for ep in range(num_episodes):
            obs = env.reset()
            if random.random() < 0.3:
                obs = env.reset_random_spawn()
            elif random.random() < 0.5:
                # 50% of the time swap spawns to add more diversity
                obs = env.swap_agents()
            agent0.reset()
            agent1.reset()

            episode_transitions = []
            ep_states = []

            H, W, _ = obs[0].shape

            for step in range(args.max_steps):
                # Both agents act using pure heuristics
                a_0, _, _ = agent0.select_action(obs[0])
                a_1, _, true_opp_heatmap = agent1.select_action(obs[1])
                actions = {0: a_0, 1: a_1}

                global_state = env.get_global_state()
                next_obs, reward, done, info = env.step(actions)
                next_global_state = env.get_global_state()

                transition = {
                    "state": obs[0].copy(),
                    "global_state": global_state.copy(),
                    "reward": float(reward[0]),
                    "opp_reward": float(reward[1]),
                    "next_state": next_obs[0].copy(),
                    "next_global_state": next_global_state.copy(),
                    "done": bool(done),
                    "true_opp_heatmap": true_opp_heatmap.copy(),
                    "hist_len": len(ep_states) - 1,
                }
                ep_states.append(obs[0].copy())
                episode_transitions.append(transition)

                obs = next_obs
                if done:
                    break

            _apply_hindsight_relabeling(episode_transitions, H, W)

            states_arr = np.stack(ep_states)
            for t in episode_transitions:
                t["history"] = {"states": states_arr}
            master_dataset.extend(episode_transitions)

            if (ep + 1) % 100 == 0:
                print(
                    f"Collected {ep + 1} episodes... Total valid transitions: {len(master_dataset)}"
                )

    print(f"Saving dataset to {save_path}...")
    torch.save(master_dataset, save_path)
    print("Done!")


# To test heurisctic agents, no collection, only run episodes and render gameplay


def run_episode(agent0, agent1, env, args, render=False):
    obs = env.reset()
    agent0.reset()
    agent1.reset()
    total_reward_0 = 0
    total_reward_1 = 0
    renderer = RealtimeRenderer() if render else None

    for step in range(100):
        if render and renderer is not None:
            global_state = env.get_global_state()
            H, W, _ = global_state.shape
            dummy_om = np.zeros((H, W), dtype=np.float32)
            renderer.render(global_state, obs[0], obs[1], dummy_om)

        a_0, _, _ = agent0.select_action(obs[0])
        a_1, _, _ = agent1.select_action(obs[1])
        actions = {0: a_0, 1: a_1}

        next_obs, reward, done, info = env.step(actions)

        total_reward_0 += reward[0]
        total_reward_1 += reward[1]

        obs = next_obs
        if done:
            if render and renderer is not None:
                global_state = env.get_global_state()
                H, W, _ = global_state.shape
                dummy_om = np.zeros((H, W), dtype=np.float32)
                renderer.render(global_state, obs[0], obs[1], dummy_om)
            break


if __name__ == "__main__":
    # collect_offline_data(num_episodes=10)
    map_layout = MAP_4
    args = OMGArgs()
    env = SimpleForagingEnv(max_steps=args.max_steps, map_layout=map_layout)
    obs = env.reset()
    agent_0 = SimpleAgent(0, map_layout=map_layout)
    _ = agent_0.select_action(obs[0])
    precomputed_paths = agent_0.precomputed_paths
    agent_1 = SimpleAgent(1, precomputed_paths=precomputed_paths, map_layout=map_layout)
    agent_2 = GreedySwitchAgent(
        0, precomputed_paths=precomputed_paths, map_layout=map_layout
    )
    agent_3 = GreedySwitchAgent(
        1, precomputed_paths=precomputed_paths, map_layout=map_layout
    )
    agent_4 = StalkerAgent(
        0, precomputed_paths=precomputed_paths, map_layout=map_layout
    )
    agent_5 = StalkerAgent(
        1, precomputed_paths=precomputed_paths, map_layout=map_layout
    )

    for ep in range(1):
        print(f"Episode {ep + 1}")
        run_episode(agent_2, agent_3, env, args, render=True)
