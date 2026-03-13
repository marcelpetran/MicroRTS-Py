from typing import Deque, Dict, List, Tuple, Optional
import random
from collections import deque

from wandb import agent

from omg_args import OMGArgs

from simple_foraging_env import SimpleAgent, RandomAgent, SimpleForagingEnv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class QNet(nn.Module):
  """
  Simple MLP Q-network for Q(s, g, a)
  state_shape: (H, W, F)
  latent_dim: dimension of latent opponent representation
  action_dim: number of discrete actions
  Returns Q-values for all actions.

  """

  def __init__(self, args: OMGArgs):
    super().__init__()
    H, W, F_dim = args.state_shape
    self.state_dim = H * W * F_dim
    self.action_dim = args.action_dim
    cnn_hidden = args.cnn_hidden

    self.flat_dim = cnn_hidden * H * W
    input_channels = F_dim + 1

    self.cnn = nn.Sequential(
        nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, cnn_hidden, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(cnn_hidden, cnn_hidden, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten()
    )

    # Heads (Dueling)
    self.advantage_head = nn.Sequential(
        nn.Linear(self.flat_dim, args.qnet_hidden),
        nn.ReLU(),
        nn.Linear(args.qnet_hidden, self.action_dim)
    )

    self.value_head = nn.Sequential(
        nn.Linear(self.flat_dim, args.qnet_hidden),
        nn.ReLU(),
        nn.Linear(args.qnet_hidden, 1)
    )

    # Initialize weights to small values to prevent initial explosion
    self.apply(self._init_weights)

  def _init_weights(self, m):
    if isinstance(m, nn.Linear):
      nn.init.xavier_uniform_(m.weight)
      if m.bias is not None:
        nn.init.constant_(m.bias, 0.01)

  def forward(self, batch: torch.Tensor, g_map: torch.Tensor) -> torch.Tensor:
    """
    Args:
        batch: Game state (B, H, W, F)
        g_map: Heatmap of inferred opponent subgoal (B, H, W)
        return_aux: Whether to return auxiliary prediction (used during training)
    """
    # Batch shape: (B, H, W, F)
    # Permute to (B, F, H, W) for PyTorch Conv2d
    s = batch.permute(0, 3, 1, 2)
    # (B, 1, H, W) - broadcast latent g across spatial dimensions
    g = g_map.unsqueeze(1)

    x = torch.cat([s, g], dim=1)
    features = self.cnn(x)

    # Dueling Heads
    adv = self.advantage_head(features)
    val = self.value_head(features)
    q_vals = val + adv - adv.mean(dim=1, keepdim=True)

    return q_vals


class ReplayBuffer:
  """
  Simple FIFO experience replay buffer for Q-learning.
  """

  def __init__(self, capacity: int):
    self.capacity = capacity
    self.buf: Deque[Dict] = deque(maxlen=capacity)

  def push(self, item: Dict):
    self.buf.append(item)

  def sample(self, batch_size: int) -> List[Dict]:
    return random.sample(self.buf, batch_size)

  def __len__(self):
    return len(self.buf)


class QLearningAgent:
  """
  Q(s, g, a) OMG agent that:
    • 

  Expected OpponentModel API:
    - 
  """

  def __init__(self, env, opponent_model, args: OMGArgs = OMGArgs()):
    self.env = env
    self.model = opponent_model
    self.args = args
    self.device = torch.device(args.device)
    self.opponent_agent = RandomAgent(1)

    # Try to infer dims from env
    if args.state_shape is None:
      # env observation: (H, W, F)
      obs = self.env.reset()
      H, W, F_dim = obs.shape
      self.args.state_shape = (H, W, F_dim)
    if not hasattr(self.env, "action_space") or self.env.action_space is None:
      raise ValueError("Env must have action_space (list or int).")
    self.args.action_dim = len(self.env.action_space) if hasattr(
      self.env.action_space, "__len__") else int(self.env.action_space)

    # Networks
    self.q = QNet(args).to(self.device)
    self.q_tgt = QNet(args).to(self.device)
    self.q_tgt.load_state_dict(self.q.state_dict())
    self.opt = torch.optim.Adam(self.q.parameters(), lr=self.args.lr)

    # Replay
    self.replay = ReplayBuffer(self.args.capacity)

    # Schedules
    self.global_step = 0

  def reset(self):
    pass

  # ------------- epsilon schedules --------------

  def _eps(self) -> float:
    t = min(self.global_step, self.args.eps_decay_steps)
    return self.args.eps_end + (self.args.eps_start - self.args.eps_end) * (1 - t / self.args.eps_decay_steps)

  def _tau(self) -> float:
    t = min(self.global_step, self.args.tau_decay_steps)
    return self.args.tau_end + (self.args.tau_start - self.args.tau_end) * (1 - t / self.args.tau_decay_steps)

  def _beta(self) -> float:
    t = min(self.global_step, self.args.beta_decay_steps)
    return self.args.beta_end + (self.args.beta_start - self.args.beta_end) * (1 - t / self.args.beta_decay_steps)

  @torch.no_grad()
  def value(self, s_t: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """
    s_t: (1, H, W, F), g: (1, latent_dim) -> Q(1, A)
    API to compute V(s,g) = mean_a Q(s,g,a)
    """
    self.q.eval()
    return self.q(s_t, g)  # (1, A)

  # ------------- visualization utility -------------
  @torch.no_grad()
  def heatmap_q_values(self, g: torch.Tensor, filename: str = "q_heatmap.png"):
    """
    Utility to visualize Q-values as a heatmap over the grid for a given state and subgoal.

    Args:
        state_hwf (np.ndarray): The current state grid, shape (H, W, F).
        g (torch.Tensor): The inferred subgoal, shape (1, latent_dim).
        filename (str): Path to save the heatmap image.
    """
    self.q.eval()
    H, W, _ = self.args.state_shape
    g = g.unsqueeze(0)  # (1, latent_dim)

    # This will store the max Q-value for each grid cell
    q_value_map = np.zeros((H, W))
    # This will store the best action (0:Up, 1:Down, 2:Left, 3:Right) for each cell
    policy_map = np.zeros((H, W))

    # Find the original position of our agent (agent 1, feature index 2)
    original_pos = self.env._get_agent_positions()[0]
    # Iterate over every possible cell in the grid
    for pos in self.env._get_freed_positions() + [original_pos]:
      r, c = pos

      self.env._place_agent(0, pos)
      temp_state = self.env._get_observations()[0]  # Get the modified state

      s_tensor = torch.from_numpy(
        temp_state).float().unsqueeze(0).to(self.device)

      # subgoal is valid only for the current agent position
      # but true q-values with correct subgoals are expensive to compute
      # so this is an approximation
      q_values = self.q(s_tensor, g)  # (1, num_actions)

      max_q_val, best_action = torch.max(q_values, dim=1)
      q_value_map[r, c] = max_q_val.item()
      policy_map[r, c] = best_action.item()

    # Restore the agent's original position
    self.env._place_agent(0, original_pos)
    agent_pos = self.env._get_agent_positions()[0]
    opp_pos = self.env._get_agent_positions()[1]
    food_pos = self.env._get_food_positions()
    wall_pos = self.env._get_wall_positions()

    # --- Plotting the Heatmap ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # Mark agent, opponent, and food positions on the heatmap
    ax1.scatter(agent_pos[1], agent_pos[0], color='blue',
                marker='X', s=100, label='Agent')
    ax1.scatter(opp_pos[1], opp_pos[0], color='red',
                marker='X', s=100, label='Opponent')
    if food_pos:
      food_x = [pos[1] for pos in food_pos]
      food_y = [pos[0] for pos in food_pos]
      ax1.scatter(food_x, food_y, color='green',
                  marker='o', s=50, label='Food')
    if wall_pos:
      wall_x = [pos[1] for pos in wall_pos]
      wall_y = [pos[0] for pos in wall_pos]
      ax1.scatter(wall_x, wall_y, color='black',
                  marker='s', s=50, label='Wall')
    # Plot Q-value heatmap
    im1 = ax1.imshow(q_value_map, cmap='viridis')
    ax1.set_title("Max Q(s, g, a) Heatmap")
    fig.colorbar(im1, ax=ax1)
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)

    # Plot Policy map with arrows
    ax2.imshow(q_value_map, cmap='gray')  # Show background values
    ax2.set_title("Learned Policy (Arrows)")
    action_arrows = ['^', 'v', '<', '>']
    for r in range(H):
      for c in range(W):
        action = int(policy_map[r, c])
        ax2.text(c, r, action_arrows[action], ha='center',
                 va='center', color='red', fontsize=12)

    plt.suptitle("Policy and Q-value Heatmap")
    plt.savefig(filename)
    plt.close('all')

  def heatmap_subgoal(self, g_map: torch.Tensor, filename: str = "subgoal_heatmap.png"):
    """
    Utility to visualize the inferred subgoal heatmap, with marked agent positions and food locations.

    Args:
        s_t (torch.Tensor): Current state, shape (1, H, W, F).
        g_map (torch.Tensor): Inferred subgoal heatmap, shape (1, H, W).
        filename (str): Path to save the heatmap image.
    """
    self.q.eval()
    g_map_np = g_map.squeeze(0).cpu().numpy()  # (H, W)
    agent_pos = self.env._get_agent_positions()[0]
    opponent_pos = self.env._get_agent_positions()[1]
    food_pos = self.env._get_food_positions()
    wall_pos = self.env._get_wall_positions()

    plt.figure(figsize=(6, 6))
    plt.imshow(g_map_np, cmap='viridis')
    plt.colorbar(label='Inferred Subgoal Probability')
    plt.scatter(agent_pos[1], agent_pos[0], color='blue',
                marker='X', s=100, label='Agent')
    plt.scatter(opponent_pos[1], opponent_pos[0],
                color='red', marker='X', s=100, label='Opponent')
    if food_pos:
      food_x = [pos[1] for pos in food_pos]
      food_y = [pos[0] for pos in food_pos]
      plt.scatter(food_x, food_y, color='green',
                  marker='o', s=50, label='Food')
    if wall_pos:
      wall_x = [pos[1] for pos in wall_pos]
      wall_y = [pos[0] for pos in wall_pos]
      plt.scatter(wall_x, wall_y, color='black',
                  marker='s', s=50, label='Wall')
    plt.title("Inferred Subgoal Heatmap")
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)
    plt.savefig(filename)
    plt.close('all')

  # ------------- acting -------------

  def choose_action(self, qvals: torch.Tensor, beta: float, eval=False) -> int:
    gumbel_noise = -beta * torch.empty_like(qvals).exponential_().log()
    if eval == True:
      noise = torch.rand_like(qvals) * 1e-6
      return int(torch.argmax(qvals + noise))
    return int(torch.argmax(qvals + gumbel_noise))

  def select_action(self, s_t: np.ndarray, history: Dict[str, List[torch.Tensor]], eval=False) -> Tuple[int, torch.Tensor]:
    """
    (interaction phase) Infer g_hat and act eps-greedily on Q(s,g_hat,*)
    """
    x = torch.from_numpy(s_t).float().unsqueeze(0).to(self.device)
    collated_history = self.collate_history([history])
    with torch.no_grad():
      g_logits = self.model(x, collated_history)
      g_map = F.softmax(g_logits.view(
        g_logits.shape[0], -1), dim=-1).view_as(g_logits)  # (B, H, W)
    qvals = self.q(x, g_map)
    a = self.choose_action(qvals, self._tau(), eval)
    return a, g_map.squeeze(0)

  # ------------- training -------------

  def compute_targets(self, batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Standard DDQN target computation using Hindsight Experience Replay Goal Maps.
    """
    s = torch.from_numpy(
      np.array([b["state"] for b in batch], dtype=np.float32)).to(self.device)
    sp = torch.from_numpy(
      np.array([b["next_state"] for b in batch], dtype=np.float32)).to(self.device)
    a = torch.from_numpy(
      np.array([b["action"] for b in batch], dtype=np.int64)).to(self.device)
    r = torch.from_numpy(
      np.array([b["reward"] for b in batch], dtype=np.float32)).to(self.device)
    done = torch.from_numpy(
      np.array([b["done"] for b in batch], dtype=np.float32)).to(self.device)

    # The Goal for this trajectory (Hindsight Label)
    # We unsqueeze(1) because the CNN expects (B, 1, H, W) to concat with states
    g_map = torch.from_numpy(
      np.array([b["rollout_goal_map"] for b in batch], dtype=np.float32)).to(self.device)
    g_map_next = torch.from_numpy(
      np.array([b["rollout_goal_map_next"] for b in batch], dtype=np.float32)).to(self.device)

    # 1. Q(s, g, a)
    q_sa = self.q(s, g_map).gather(1, a.unsqueeze(1)).squeeze(1)

    # 2. Target = r + gamma * max_a' Q_tgt(s', g, a')
    with torch.no_grad():
      q_val = self.q(sp, g_map_next)
      noise = torch.rand_like(q_val) * 1e-6
      best_actions = (q_val + noise).argmax(dim=1, keepdim=True)

      q_next = self.q_tgt(sp, g_map_next).gather(1, best_actions).squeeze(1)

      target = r + (1.0 - done) * self.args.gamma * q_next
      target = torch.clamp(target, min=-5.0, max=5.0)

    return q_sa, target

  def update(self):
    if len(self.replay) < self.args.min_replay:
      return (None, None)

    if self.global_step % self.args.train_every != 0:
      return (None, None)

    batch_list = self.replay.sample(self.args.batch_size)

    # --- Update the Opponent Model Transformer ---
    # We ONLY train the transformer on steps where the opponent succeeded!
    valid_indices = [i for i, b in enumerate(
      batch_list) if b.get("valid_for_transformer", False)]

    if len(valid_indices) > 0:
      valid_batch = [batch_list[i] for i in valid_indices]

      om_batch = {
          "states": torch.from_numpy(np.array([b["state"] for b in valid_batch], dtype=np.float32)).to(self.device),
          "history": self.collate_history([b["history"] for b in valid_batch]),
          "true_goal_map": torch.from_numpy(np.array([b["true_goal_map"] for b in valid_batch], dtype=np.float32)).to(self.device)
      }
      model_loss = self.model.train_step(om_batch)
    else:
      model_loss = 0.0

    # Skip Q-learning updates for the first few steps to let the transformer learn something reasonable
    if self.global_step < self.args.update_after and not self.args.oracle:
      return 0.0, model_loss

    # --- Update the Q-Network ---
    q_sa, target = self.compute_targets(batch_list)
    loss = F.smooth_l1_loss(q_sa, target, reduction='mean')
    loss_val = loss.item()
    if loss_val < self.args.aux_loss_threshold:
      return 0.0, model_loss

    self.opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
    self.opt.step()

    # --- Target Update ---
    with torch.no_grad():
      for param, target_param in zip(self.q.parameters(), self.q_tgt.parameters()):
        target_param.data.mul_(1 - self.args.tau_soft)
        target_param.data.add_(self.args.tau_soft * param.data)

    return loss_val, model_loss

  def collate_history(self, histories: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Pads history sequences to max_len within the batch.
    """
    if not histories:
      return {}

    true_lengths = [len(h.get("states", [])) for h in histories]
    max_len = self.args.max_history_length

    if max_len == 0:
      return {"states": torch.empty(0).to(self.device), "actions": torch.empty(0).to(self.device), "mask": torch.empty(0).to(self.device)}

    B = len(histories)
    H, W, F_dim = self.args.state_shape

    padded_states_np = np.zeros((B, max_len, H, W, F_dim), dtype=np.float32)
    padded_actions_np = np.zeros((B, max_len), dtype=np.int64)

    for i, h in enumerate(histories):
      seq_len = true_lengths[i]
      if seq_len > 0:
        padded_states_np[i, :seq_len] = h["states"]
        padded_actions_np[i, :seq_len] = h["actions"]

    final_padded_states = torch.from_numpy(padded_states_np).to(self.device)
    final_padded_actions = torch.from_numpy(padded_actions_np).to(self.device)

    # Fast mask generation
    true_lengths_np = np.array(true_lengths, dtype=np.int64)
    mask = torch.arange(max_len, device=self.device).expand(
      B, max_len) < torch.from_numpy(true_lengths_np).to(self.device).unsqueeze(1)
    return {
        "states": final_padded_states,
        "actions": final_padded_actions,
        "mask": mask
    }

  # ------------- rollout -------------

  def run_episode(self, opponent_agent, max_steps: Optional[int] = None) -> Dict[str, float]:
    """
    Gathers a trajectory, predicts subgoals, and uses Hindsight 
    to label the true subgoals at the end of the episode.
    """
    opp_loss = 0.0
    obs = self.env.reset()
    opponent_agent.reset()

    done = False
    ep_ret = 0.0

    # History container for the Transformer
    history_len = self.args.max_history_length
    history = {
        "states": deque(maxlen=history_len),
        "actions": deque(maxlen=history_len)
    }

    # Temporary list to hold the episode before hindsight labeling
    episode_transitions = []

    # Get grid dimensions for the map
    H, W, _ = obs[0].shape

    for step in range(max_steps or 500):
      current_history = {k: list(v) for k, v in history.items()}

      a, g_map = self.select_action(obs[0], current_history)
      a_opponent = opponent_agent.select_action(obs[1])
      actions = {0: a, 1: a_opponent}

      next_obs, reward, done, info = self.env.step(actions)

      if hasattr(opponent_agent, 'replay'):
        opp_step_info = {
            "state": obs[1].copy(),
            "action": a_opponent,
            "reward": float(reward[1]),
            "next_state": next_obs[1].copy(),
            "done": bool(done),
        }
        opponent_agent.replay.push(opp_step_info)
        opponent_agent.global_step += 1
        opp_loss = opponent_agent.update()

      # 2. Store the step without the true label (we don't know it yet)
      transition = {
          "state": obs[0].copy(),
          "action": a,
          "opp_action": a_opponent,
          "reward": float(reward[0]),
          "opp_reward": float(reward[1]),
          "next_state": next_obs[0].copy(),
          "done": bool(done),
          "rollout_goal_map": g_map.cpu().numpy(),
          "history": {k: [np.copy(item) if isinstance(item, np.ndarray) else item for item in v] for k, v in current_history.items()},
      }
      episode_transitions.append(transition)

      # 3. Update history
      history["states"].append(obs[0].copy())
      history["actions"].append(a_opponent)

      ep_ret += reward[0]
      obs = next_obs

      # 4. Train Step (Optional: can also be moved outside the loop)
      self.global_step += 1
      Q_loss, model_loss = self.update()

      if Q_loss is not None and self.global_step % 100 == 0:
        print(f"Step {self.global_step}: Q_loss={Q_loss:.5f}, Model_loss={model_loss:.5f} "
              f"Opp_Q_loss={opp_loss:.5f} "
              f"Tau={self._tau():.2f}")

      if done:
        break

    current_true_goal_pos = None
    next_map = np.zeros((H, W), dtype=np.float32)
    next_rollout_map = np.zeros((H, W), dtype=np.float32)

    # Hindsight labeling, when opponent did not succeed to get last goal,
    # we label naively with the closest food to the opponent's final position.
    # This is a heuristic but it gives the transformer a better signal to learn from than just labeling discarding the trajectory.
    if len(episode_transitions) > 0:
      final_t = episode_transitions[-1]

      if final_t["opp_reward"] == 0:
        opp_pos_arr = np.argwhere(final_t["state"][:, :, 3] == 1)
        food_pos_arr = np.argwhere(final_t["state"][:, :, 1] == 1)

        if len(opp_pos_arr) > 0 and len(food_pos_arr) > 0:
          opp_pos = tuple(opp_pos_arr[0])

          # Find the closest food using Manhattan distance
          closest_food = None
          min_dist = float('inf')
          for f_pos in food_pos_arr:
            dist = abs(opp_pos[0] - f_pos[0]) + abs(opp_pos[1] - f_pos[1])
            if dist < min_dist:
              min_dist = dist
              closest_food = tuple(f_pos)

          if closest_food is not None:
            current_true_goal_pos = closest_food

    # Walk backward through the episode to label goals
    for t in reversed(episode_transitions):

      # Did the opponent get a reward this step?
      if t["opp_reward"] > 0:
        opp_pos_indices = np.argwhere(t["next_state"][:, :, 3] == 1)
        if len(opp_pos_indices) > 0:
          current_true_goal_pos = tuple(opp_pos_indices[0])

      # Assign the goal to this step
      if current_true_goal_pos is not None:
        true_map = np.zeros((H, W), dtype=np.float32)
        true_map[current_true_goal_pos[0], current_true_goal_pos[1]] = 1.0

        t["true_goal_map"] = true_map
        t["valid_for_transformer"] = True
      else:
        true_map = np.zeros((H, W), dtype=np.float32)
        t["true_goal_map"] = true_map
        t["valid_for_transformer"] = False

      t["true_goal_map_next"] = next_map
      next_map = true_map.copy()
      del t["opp_reward"]

      t["rollout_goal_map_next"] = next_rollout_map
      next_rollout_map = t["rollout_goal_map"].copy()

    for t in episode_transitions:
      self.replay.push(t)

    return {"return": ep_ret, "steps": step + 1}

  def run_test_episode(self, opponent_agent, max_steps: Optional[int] = None, render: bool = False, zigzag: bool = False) -> Dict[str, float]:
    obs = self.env.reset()
    opponent_agent.reset()
    done = False
    ep_ret = 0.0

    history_len = self.args.max_history_length
    history = {
        "states": deque(maxlen=history_len),
        "actions": deque(maxlen=history_len)
    }

    for step in range(max_steps or 500):
      current_history = {k: list(v) for k, v in history.items()}

      a, g_map = self.select_action(
        obs[0], current_history, eval=True)
      a_opponent = opponent_agent.select_action(obs[1], eval=True)
      actions = {0: a, 1: a_opponent}

      if render:
        self.heatmap_q_values(
          g_map, f"./diagrams_{self.args.folder_id}/q_heatmap_step{self.global_step + step}.png")
        if not self.args.oracle:
          self.heatmap_subgoal(
            g_map, f"./diagrams_{self.args.folder_id}/gmap_step{self.global_step + step}.png")
        SimpleForagingEnv.render_from_obs(obs[0])

      next_obs, reward, done, info = self.env.step(actions)

      history["states"].append(obs[0].copy())
      history["actions"].append(a_opponent)

      ep_ret += reward[0]
      obs = next_obs

      if done:
        break

    return {"return": ep_ret, "steps": step + 1}
