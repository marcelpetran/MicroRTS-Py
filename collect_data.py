import torch
import numpy as np
from collections import deque
from simple_foraging_env import SimpleForagingEnv, SimpleAgent, GreedySwitchAgent
from maps import *
from omg_args import OMGArgs


def collect_offline_data(num_episodes=1000, save_path="./dataset/dataset.pt", map_layout=MAP_1):
  args = OMGArgs()
  env = SimpleForagingEnv(max_steps=args.max_steps, map_layout=map_layout)
  agent_0 = SimpleAgent(0)
  agent_1 = SimpleAgent(1)

  master_dataset = []

  print(f"Starting offline data collection for {num_episodes} episodes...")

  for ep in range(num_episodes):
    obs = env.reset()
    agent_0.reset()
    agent_1.reset()

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
      a_0 = agent_0.select_action(obs[0])
      a_1 = agent_1.select_action(obs[1])
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
        food_pos_arr = np.argwhere(final_t["state"][:, :, 1] == 1)
        if len(opp_pos_arr) > 0 and len(food_pos_arr) > 0:
          opp_pos = tuple(opp_pos_arr[0])
          closest_food = None
          min_dist = float('inf')
          for f_pos in food_pos_arr:
            dist = abs(opp_pos[0] - f_pos[0]) + abs(opp_pos[1] - f_pos[1])
            if dist < min_dist:
              min_dist = dist
              closest_food = tuple(f_pos)
          if closest_food is not None:
            current_true_goal_pos = closest_food

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


if __name__ == "__main__":
  collect_offline_data(num_episodes=2000)
