from typing import Deque, Dict, List, Tuple, Optional
import random
from collections import deque

from omg_args import OMGArgs

from simple_foraging_env import SimpleAgent, RandomAgent, SimpleForagingEnv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import wandb


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

  # ------------- Tau schedules --------------

  def _tau(self) -> float:
    t = min(self.global_step, self.args.tau_decay_steps)
    return self.args.tau_end + (self.args.tau_start - self.args.tau_end) * (1 - t / self.args.tau_decay_steps)

  # ------------- evaluation --------------

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
  def heatmap_q_values(self, g: torch.Tensor, filename: str = "q_heatmap.png", save: bool = True):
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
    if save:
      plt.savefig(filename)
    else:
      plt.show()
    plt.close('all')

  @torch.no_grad()
  def heatmap_subgoal(self, g_map: torch.Tensor, filename: str = "subgoal_heatmap.png", save: bool = True):
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
    if save:
      plt.savefig(filename)
    else:
      plt.show()
    plt.close('all')

  # ------------- acting -------------

  def choose_action(self, qvals: torch.Tensor, beta: float, eval=False) -> int:
    gumbel_noise = -beta * torch.empty_like(qvals).exponential_().log()

    if eval == True:
      dist = F.softmax(qvals / beta, dim=-1)
      return int(torch.multinomial(dist, num_samples=1).item())

    return int(torch.argmax(qvals + gumbel_noise))

  @torch.no_grad()
  def select_action(self, s_t: np.ndarray, history: Dict[str, torch.Tensor], eval=False) -> Tuple[int, torch.Tensor]:
    """
    (interaction phase) Infer g_hat and act eps-greedily on Q(s,g_hat,*)
    """
    x = torch.from_numpy(s_t).float().unsqueeze(0).to(self.device)
    with torch.no_grad():
      g_logits = self.model(x, history)  # (1, H, W)
      g_map = F.softmax(g_logits.view(
        g_logits.shape[0], -1), dim=-1).view_as(g_logits)  # (B, H, W)

    qvals = self.q(x, g_map)

    tau = 0.05 if eval else self._tau()
    entropy = Categorical(logits=qvals / tau).entropy().item()

    a = self.choose_action(qvals, tau, eval)

    return a, g_map.squeeze(0), entropy

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

    with torch.no_grad():
      self.model.inference_model.eval()

      hist_feats = torch.from_numpy(
          np.array([b["history"]["state_features"][0]
                   for b in batch], dtype=np.float32)
      ).to(self.device)
      hist_acts = torch.from_numpy(
          np.array([b["history"]["actions"][0] for b in batch], dtype=np.int64)
      ).to(self.device)
      hist_mask = torch.from_numpy(
          np.array([b["history"]["mask"][0] for b in batch])
      ).to(self.device)
      hist = {"state_features": hist_feats,
              "actions": hist_acts, "mask": hist_mask}

      g_logits = self.model.inference_model(s, hist, cached_features=True)
      g_map = F.softmax(g_logits.view(len(batch), -1),
                        dim=-1).view_as(g_logits)

      # For next state
      next_feats = torch.from_numpy(
          np.array([b["next_state_feature"] for b in batch], dtype=np.float32)
      ).to(self.device)
      hist_feats_next = torch.roll(hist_feats, shifts=-1, dims=1)
      hist_feats_next[:, -1, :] = next_feats
      hist_acts_next = torch.roll(hist_acts, shifts=-1, dims=1)
      hist_acts_next[:, -1] = torch.from_numpy(
          np.array([b["opp_action"] for b in batch], dtype=np.int64)
      ).to(self.device)
      hist_mask_next = torch.roll(hist_mask, shifts=-1, dims=1)
      hist_mask_next[:, -1] = True
      hist_next = {"state_features": hist_feats_next,
                   "actions": hist_acts_next, "mask": hist_mask_next}

      g_logits_next = self.model.inference_model(
        sp, hist_next, cached_features=True)
      g_map_next = F.softmax(g_logits_next.view(
        len(batch), -1), dim=-1).view_as(g_logits_next)

      self.model.inference_model.train()

    # 1. Q(s, g, a)
    q_sa = self.q(s, g_map).gather(1, a.unsqueeze(1)).squeeze(1)

    # 2. Target = r + gamma * max_a' Q_tgt(s', g, a')
    with torch.no_grad():
      q_val = self.q(sp, g_map_next)
      noise = torch.rand_like(q_val) * 1e-6
      best_actions = (q_val + noise).argmax(dim=1, keepdim=True)

      q_next = self.q_tgt(sp, g_map_next).gather(1, best_actions).squeeze(1)

      target = r + (1.0 - done) * self.args.gamma * q_next
      target = torch.clamp(target, min=-15.0, max=15.0)

    return q_sa, target

  def update(self):
    if len(self.replay) < self.args.min_replay:
      return (None, None)

    if self.global_step % self.args.train_every != 0:
      return (None, None)

    batch_list = self.replay.sample(self.args.batch_size)

    # --- Update the Opponent Model Transformer ---
    om_batch = {
        "states": torch.from_numpy(np.array([b["state"] for b in batch_list], dtype=np.float32)).to(self.device),
        "history": {
            "state_features": torch.from_numpy(np.array([b["history"]["state_features"][0] for b in batch_list], dtype=np.float32)).to(self.device),
            "actions": torch.from_numpy(np.array([b["history"]["actions"][0] for b in batch_list], dtype=np.int64)).to(self.device),
            "mask": torch.from_numpy(np.array([b["history"]["mask"][0] for b in batch_list], dtype=np.bool_)).to(self.device)
        },
        "true_goal_map": torch.from_numpy(np.array([b["true_goal_map"] for b in batch_list], dtype=np.float32)).to(self.device)
    }

    # --- Update the Q-Network ---
    q_sa, target = self.compute_targets(batch_list)
    loss = F.smooth_l1_loss(q_sa, target, reduction='mean')
    loss_val = loss.item()

    self.opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
    self.opt.step()

    # --- Target Update ---
    with torch.no_grad():
      for param, target_param in zip(self.q.parameters(), self.q_tgt.parameters()):
        target_param.lerp_(param, self.args.tau_soft)

    model_loss = self.model.train_step(om_batch)

    return loss_val, model_loss

  def load_historical_policy(self, state_dict: dict, om_state_dict: dict = None):
    """Loads frozen historical weights for Fictitious Play."""
    self.q.load_state_dict(state_dict)
    self.q.eval()  # Freeze layers like BatchNorm/Dropout if you add them later

    if om_state_dict is not None and hasattr(self, 'model'):
      self.model.inference_model.load_state_dict(om_state_dict)
      self.model.inference_model.eval()

  # ------------- rollout -------------

  def run_episode(self, opponent_agent, max_steps: int = 500) -> Dict[str, float]:
    """
    Gathers a trajectory, predicts subgoals, and uses Hindsight 
    to label the true subgoals at the end of the episode.
    """
    opp_loss_val = 0.0
    obs = self.env.reset()
    if random.random() < 0.3:
      obs = self.env.reset_random_spawn()
    elif random.random() < 0.5:
      # 50% of the time swap spawns to add more diversity
      obs = self.env.swap_agents()
    opponent_agent.reset()

    done = False
    ep_ret = 0.0
    opp_ret = 0.0
    ep_entropy = 0.0
    q_losses = []
    model_losses = []
    opp_losses = []

    # History buffer for the transformer
    history_len = self.args.max_history_length
    rolling_feats = torch.zeros(
      (1, history_len, self.args.d_model), device=self.device)
    rolling_actions = torch.zeros(
      (1, history_len), dtype=torch.long, device=self.device)
    rolling_mask = torch.zeros(
      (1, history_len), dtype=torch.bool, device=self.device)
    current_seq_len = 0

    # Temporary list to hold the episode before hindsight labeling
    episode_transitions = []

    # Get grid dimensions for the map
    H, W, _ = obs[0].shape

    for step in range(max_steps):
      history_gpu = {
        "state_features": rolling_feats,
        "actions": rolling_actions,
        "mask": rolling_mask
      }
      a, g_map, step_entropy = self.select_action(obs[0], history_gpu)
      a_opponent, _, _ = opponent_agent.select_action(obs[1])
      actions = {0: a, 1: a_opponent}

      ep_entropy += step_entropy

      next_obs, reward, done, info = self.env.step(actions)

      if hasattr(opponent_agent, 'replay'):
        opp_step_info = {
            "state": obs[0].copy(),
            "action": a,
        }
        opponent_agent.replay.push(opp_step_info)
        opponent_agent.global_step += 1

        opp_loss = opponent_agent.update()

        if opp_loss is not None:
          opp_loss_val = opp_loss

      history_cpu = {
          "state_features": rolling_feats.cpu().numpy(),
          "actions": rolling_actions.cpu().numpy(),
          "mask": rolling_mask.cpu().numpy()
      }

      # Store the step without the true label
      transition = {
          "state": obs[0].copy(),
          "action": a,
          "opp_action": a_opponent,
          "reward": float(reward[0]),
          "opp_reward": float(reward[1]),
          "next_state": next_obs[0].copy(),
          "done": bool(done),
          "history": history_cpu  # Store the raw history for hindsight labeling later
      }
      episode_transitions.append(transition)

      # Update history
      state_tensor = torch.from_numpy(
        obs[0]).float().unsqueeze(0).to(self.device)
      with torch.no_grad():
        new_feat = self.model.inference_model.get_features(
          state_tensor)  # (1, d_model)
      transition["next_state_feature"] = new_feat.squeeze(0).cpu().numpy()

      rolling_feats = torch.roll(rolling_feats, shifts=-1, dims=1)
      rolling_actions = torch.roll(rolling_actions, shifts=-1, dims=1)
      rolling_mask = torch.roll(rolling_mask, shifts=-1, dims=1)

      rolling_feats[:, -1, :] = new_feat
      rolling_actions[:, -1] = a_opponent

      if current_seq_len < history_len:
        current_seq_len += 1
        rolling_mask[:, -current_seq_len:] = True

      ep_ret += reward[0]
      opp_ret += reward[1]
      obs = next_obs

      # Train Step
      self.global_step += 1
      Q_loss, model_loss = self.update()

      q_losses.append(Q_loss)
      model_losses.append(model_loss)
      opp_losses.append(opp_loss_val)

      if done:
        break

    current_true_goal_pos = None
    next_map = np.zeros((H, W), dtype=np.float32)

    # Hindsight labeling, when opponent did not succeed to get last subgoal.
    if len(episode_transitions) > 0:
      final_t = episode_transitions[-1]

      if final_t["opp_reward"] == 0:
        opp_pos_arr = np.argwhere(final_t["state"][:, :, 3] == 1)

        if len(opp_pos_arr) > 0:
          # Label the final achieved state as the intended goal
          current_true_goal_pos = tuple(opp_pos_arr[0])

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
      else:
        true_map = np.zeros((H, W), dtype=np.float32)
        t["true_goal_map"] = true_map

      t["true_goal_map_next"] = next_map
      next_map = true_map.copy()
      del t["opp_reward"]

    for t in episode_transitions:
      self.replay.push(t)

    valid_q_losses = [l for l in q_losses if l is not None]
    valid_model_losses = [l for l in model_losses if l is not None]
    valid_opp_losses = [l for l in opp_losses if l is not None]

    return {
      F"return": ep_ret,
      "steps": step + 1,
      "opp_return": opp_ret,
      "avg_entropy": ep_entropy / (step + 1),
      "avg_q_loss": np.mean(valid_q_losses) if valid_q_losses else 0.0,
      "avg_model_loss": np.mean(valid_model_losses) if valid_model_losses else 0.0,
      "avg_opp_loss": np.mean(valid_opp_losses) if valid_opp_losses else 0.0
    }

  def run_test_episode(self, opponent_agent, max_steps: int = 500, render: bool = False) -> Dict[str, float]:
    obs = self.env.reset()
    opponent_agent.reset()
    done = False
    ep_ret = 0.0
    opp_ret = 0.0
    ep_entropy = 0.0
    kd_errors = []
    spatial_errors = []

    # History container for the Transformer
    history_len = self.args.max_history_length
    rolling_feats = torch.zeros(
      (1, history_len, self.args.d_model), device=self.device)
    rolling_actions = torch.zeros(
      (1, history_len), dtype=torch.long, device=self.device)
    rolling_mask = torch.zeros(
      (1, history_len), dtype=torch.bool, device=self.device)
    current_seq_len = 0

    for step in range(max_steps):
      history = {
        "state_features": rolling_feats,
        "actions": rolling_actions,
        "mask": rolling_mask
      }

      a, g_map, step_entropy = self.select_action(
        obs[0], history, eval=True)
      a_opponent, _, opp_heatmap = opponent_agent.select_action(
        obs[1], eval=True)
      actions = {0: a, 1: a_opponent}

      if render:
        self.heatmap_q_values(
          g_map, f"./diagrams/{self.args.folder_id}/q_heatmap_step{self.global_step + step}.png")
        if not self.args.oracle:
          self.heatmap_subgoal(
            g_map, f"./diagrams/{self.args.folder_id}/gmap_step{self.global_step + step}.png")
        self.env.render()

      if g_map.dim() == 2:
        g_map = g_map.unsqueeze(0)  # (1, H, W)

      # Convert to tensor (1, H, W) and move to device
      opp_heatmap = torch.from_numpy(opp_heatmap).unsqueeze(0).to(self.device)

      kl_error = self.model.heatmap_kl_divergence(g_map, opp_heatmap)
      spatial_error = self.model.top1_spatial_error(g_map, opp_heatmap)
      kd_errors.append(kl_error)
      spatial_errors.append(spatial_error)

      next_obs, reward, done, info = self.env.step(actions)

      state_tensor = torch.from_numpy(
        obs[0]).float().unsqueeze(0).to(self.device)
      with torch.no_grad():
        new_feat = self.model.inference_model.get_features(
          state_tensor)  # (1, d_model)

      rolling_feats = torch.roll(rolling_feats, shifts=-1, dims=1)
      rolling_actions = torch.roll(rolling_actions, shifts=-1, dims=1)
      rolling_mask = torch.roll(rolling_mask, shifts=-1, dims=1)

      rolling_feats[:, -1, :] = new_feat
      rolling_actions[:, -1] = a_opponent

      if current_seq_len < history_len:
        current_seq_len += 1
        rolling_mask[:, -current_seq_len:] = True

      ep_ret += reward[0]
      opp_ret += reward[1]
      obs = next_obs
      ep_entropy += step_entropy

      if done:
        break

    return {
      "return": ep_ret,
      "steps": step + 1,
      "opp_return": opp_ret,
      "avg_entropy": ep_entropy / (step + 1),
      "avg_kl_error": np.mean(kd_errors) if kd_errors else None,
      "avg_spatial_error": np.mean(spatial_errors) if spatial_errors else None
    }
