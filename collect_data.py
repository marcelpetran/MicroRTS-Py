import torch
import numpy as np
import random
from collections import deque

from simple_foraging_env import SimpleForagingEnv, SimpleAgent, GreedySwitchAgent, StalkerAgent, ChameleonAgent
from maps import *
from omg_args import OMGArgs


def collect_offline_data(num_episodes=1000, save_path="./dataset/dataset.pt", map_layout=MAP_1):
  args = OMGArgs()
  env = SimpleForagingEnv(max_steps=args.max_steps, map_layout=map_layout)
  obs = env.reset()
  agent_0 = SimpleAgent(0)
  # precompute paths for other agents to use during data collection
  # Dummy action to trigger path precomputation in the environment
  _ = agent_0.select_action(obs[0])
  precomputed_paths = agent_0.precomputed_paths
  agent_1 = SimpleAgent(1, precomputed_paths=precomputed_paths)
  agent_2 = GreedySwitchAgent(0, precomputed_paths=precomputed_paths)
  agent_3 = GreedySwitchAgent(1, precomputed_paths=precomputed_paths)
  agent_4 = StalkerAgent(0, precomputed_paths=precomputed_paths)
  agent_5 = StalkerAgent(1, precomputed_paths=precomputed_paths)
  agent_combinations = [
    (agent_0, agent_1),
    (agent_0, agent_3),
    (agent_2, agent_1),
    (agent_2, agent_3)
  ]

  master_dataset = []

  print(f"Starting offline data collection for {num_episodes} episodes...")
  for (agent0, agent1) in agent_combinations:
    print(
      f"Collecting data for agent combination: {agent0.__class__.__name__} vs {agent1.__class__.__name__}")
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
      history_len = args.max_history_length
      history = {
          "states": deque(maxlen=history_len),
          "actions": deque(maxlen=history_len)
      }

      H, W, _ = obs[0].shape

      for step in range(args.max_steps or 500):
        current_history = {k: list(v) for k, v in history.items()}

        # Both agents act using pure heuristics
        a_0, _ = agent0.select_action(obs[0])
        a_1, _ = agent1.select_action(obs[1])
        actions = {0: a_0, 1: a_1}

        next_obs, reward, done, info = env.step(actions)

        transition = {
            "state": obs[0].copy(),
            "action": a_0,
            "opp_action": a_1,
            "reward": float(reward[0]),
            "opp_reward": float(reward[1]),
            "next_state": next_obs[0].copy(),
            "done": bool(done),
            "history": {k: [np.copy(item) if isinstance(item, np.ndarray) else item for item in v] for k, v in current_history.items()},
        }
        episode_transitions.append(transition)

        history["states"].append(obs[0].copy())
        history["actions"].append(a_1)

        obs = next_obs
        if done:
          break

      current_true_goal_pos = None

      if len(episode_transitions) > 0:
        final_t = episode_transitions[-1]

        if final_t["opp_reward"] == 0:
          opp_pos_arr = np.argwhere(final_t["state"][:, :, 3] == 1)

          if len(opp_pos_arr) > 0:
            current_true_goal_pos = tuple(opp_pos_arr[0])

      for t in reversed(episode_transitions):
        if t["opp_reward"] > 0:
          opp_pos_indices = np.argwhere(t["next_state"][:, :, 3] == 1)
          if len(opp_pos_indices) > 0:
            current_true_goal_pos = tuple(opp_pos_indices[0])

        if current_true_goal_pos is not None:
          true_map = np.zeros((H, W), dtype=np.float32)
          true_map[current_true_goal_pos[0], current_true_goal_pos[1]] = 1.0
          t["true_goal_map"] = true_map
          t["valid_for_transformer"] = True
        else:
          true_map = np.zeros((H, W), dtype=np.float32)
          t["true_goal_map"] = true_map
          t["valid_for_transformer"] = False

        del t["opp_reward"]
        del t["reward"]
        del t["next_state"]

        if t["valid_for_transformer"]:
          master_dataset.append(t)

      if (ep + 1) % 100 == 0:
        print(
          f"Collected {ep + 1} episodes... Total valid transitions: {len(master_dataset)}")

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

  for step in range(100):
    if render:
      env.render()

    a_0, _ = agent0.select_action(obs[0])
    a_1, _ = agent1.select_action(obs[1])
    actions = {0: a_0, 1: a_1}

    next_obs, reward, done, info = env.step(actions)

    total_reward_0 += reward[0]
    total_reward_1 += reward[1]

    obs = next_obs
    if done:
      if render:
        env.render()
      break


if __name__ == "__main__":
  # collect_offline_data(num_episodes=10)
  args = OMGArgs()
  env = SimpleForagingEnv(max_steps=args.max_steps, map_layout=MAP_5)
  obs = env.reset()
  agent_0 = SimpleAgent(0)
  _ = agent_0.select_action(obs[0])
  precomputed_paths = agent_0.precomputed_paths
  agent_1 = SimpleAgent(1, precomputed_paths=precomputed_paths)
  agent_2 = GreedySwitchAgent(0, precomputed_paths=precomputed_paths)
  agent_3 = GreedySwitchAgent(1, precomputed_paths=precomputed_paths)
  agent_4 = StalkerAgent(0, precomputed_paths=precomputed_paths)
  agent_5 = StalkerAgent(1, precomputed_paths=precomputed_paths)

  for ep in range(2):
    print(f"Episode {ep + 1}")
    run_episode(agent_0, agent_5, env, args, render=True)
