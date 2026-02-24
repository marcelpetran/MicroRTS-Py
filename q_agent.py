from typing import Deque, Dict, List, Tuple, Optional
import random
from collections import deque

from wandb import agent

from omg_args import OMGArgs

from simple_foraging_env import SimpleAgent, RandomAgent, SimpleForagingEnv, ZigZagAgent

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

    self.flat_dim = 64 * H * W
    input_channels = F_dim + 1

    self.cnn = nn.Sequential(
        nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
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
    # --- Plotting the Heatmap ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Q-value heatmap
    im1 = ax1.imshow(q_value_map, cmap='viridis')
    ax1.set_title("Max Q(s, g, a) Heatmap")
    fig.colorbar(im1, ax=ax1)

    # Plot Policy map with arrows
    ax2.imshow(q_value_map, cmap='gray')  # Show background values
    ax2.set_title("Learned Policy (Arrows)")
    action_arrows = ['^', 'v', '<', '>']
    for r in range(H):
      for c in range(W):
        action = int(policy_map[r, c])
        ax2.text(c, r, action_arrows[action], ha='center',
                 va='center', color='red', fontsize=12)

    plt.suptitle("Agent's Learned Policy for a Given Subgoal")
    plt.savefig(filename)
    plt.close()

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

    plt.figure(figsize=(6, 6))
    plt.imshow(g_map_np, cmap='viridis')
    plt.colorbar(label='Inferred Subgoal Probability')
    plt.scatter(agent_pos[1], agent_pos[0], color='blue',
                marker='X', s=100, label='Agent')
    plt.scatter(opponent_pos[1], opponent_pos[0],
                color='red', marker='X', s=100, label='Opponent')
    for pos in food_pos:
      plt.scatter(pos[1], pos[0], color='green',
                  marker='o', s=50, label='Food')
    plt.title("Inferred Subgoal Heatmap with Agent and Food Positions")
    plt.legend()
    plt.savefig(filename)
    plt.close()

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
    s = torch.stack([torch.from_numpy(b["state"]).float()
                    for b in batch], dim=0).to(self.device)
    sp = torch.stack([torch.from_numpy(b["next_state"]).float()
                     for b in batch], dim=0).to(self.device)
    a = torch.tensor([b["action"] for b in batch],
                     dtype=torch.long, device=self.device)
    r = torch.tensor([b["reward"] for b in batch],
                     dtype=torch.float32, device=self.device)
    done = torch.tensor([b["done"] for b in batch],
                        dtype=torch.float32, device=self.device)

    # The Goal for this trajectory (Hindsight Label)
    # We unsqueeze(1) because the CNN likely expects (B, 1, H, W) to concat with states
    g_map = torch.stack([torch.from_numpy(b["true_goal_map"]).float()
                        for b in batch], dim=0).to(self.device)
    g_map_next = torch.stack([torch.from_numpy(b["true_goal_map_next"]).float()
                             for b in batch], dim=0).to(self.device)


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

    # --- 1. Update the Q-Network ---
    q_sa, target = self.compute_targets(batch_list)
    loss = F.smooth_l1_loss(q_sa, target, reduction='mean')

    self.opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
    self.opt.step()

    # --- 2. Update the Opponent Model (Transformer) ---
    # We ONLY train the transformer on steps where the opponent succeeded!
    valid_indices = [i for i, b in enumerate(
      batch_list) if b.get("valid_for_transformer", False)]

    if len(valid_indices) > 0:
      valid_batch = [batch_list[i] for i in valid_indices]

      om_batch = {
          "states": torch.stack([torch.from_numpy(b["state"]).float() for b in valid_batch], dim=0).to(self.device),
          "history": self.collate_history([b["history"] for b in valid_batch]),
          "true_goal_map": torch.stack([torch.from_numpy(b["true_goal_map"]).float() for b in valid_batch], dim=0).to(self.device)
      }
      model_loss = self.model.train_step(om_batch)
    else:
      model_loss = 0.0

    # --- 3. Target Update ---
    with torch.no_grad():
      for param, target_param in zip(self.q.parameters(), self.q_tgt.parameters()):
        target_param.data.mul_(1 - self.args.tau_soft)
        target_param.data.add_(self.args.tau_soft * param.data)

    return loss.item(), model_loss

  def collate_history(self, histories: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Pads history sequences to max_len within the batch.
    """
    if not histories:
      return {}

    true_lengths = [len(h.get("states", [])) for h in histories]
    max_len = max(true_lengths) if true_lengths else 0

    if max_len == 0:
      return {"states": torch.empty(0).to(self.device), "actions": torch.empty(0).to(self.device), "mask": torch.empty(0).to(self.device)}

    mask = torch.arange(max_len, device=self.device)[None, :] < torch.tensor(
      true_lengths, device=self.device)[:, None]

    padded_states_list = []
    padded_actions_list = []
    # Create null tensors on CPU first
    null_state_cpu = torch.zeros(*self.args.state_shape)
    null_action_cpu = torch.tensor(0)

    for h in histories:
      num_to_pad = max_len - len(h.get("states", []))

      states = list(h.get("states", []))
      actions = list(h.get("actions", []))

      if num_to_pad > 0:
        states.extend([null_state_cpu] * num_to_pad)
        actions.extend([null_action_cpu] * num_to_pad)

      padded_states_list.append(torch.stack(states, dim=0))
      padded_actions_list.append(torch.stack(actions, dim=0))

    # Stack batches and move to device all at once
    final_padded_states = torch.stack(
      padded_states_list, dim=0).to(self.device)
    final_padded_actions = torch.stack(
      padded_actions_list, dim=0).to(self.device)

    return {
        "states": final_padded_states,
        "actions": final_padded_actions,
        "mask": mask
    }

  # ------------- rollout -------------

  def run_episode(self, max_steps: Optional[int] = None) -> Dict[str, float]:
    """
    Gathers a trajectory, predicts subgoals, and uses Hindsight 
    to label the true subgoals at the end of the episode.
    """
    self.opponent_agent = SimpleAgent(1)
    if np.random.rand() < 0.3:
      obs = self.env.reset_random_spawn(0)
    else:
      obs = self.env.reset()

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
      a_opponent = self.opponent_agent.select_action(obs[1])
      actions = {0: a, 1: a_opponent}

      next_obs, reward, done, info = self.env.step(actions)

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
          "history": {k: [t.clone().cpu() for t in v] for k, v in current_history.items()},
      }
      episode_transitions.append(transition)

      # 3. Update history
      history["states"].append(
        torch.from_numpy(obs[0]).float().to(self.device))
      history["actions"].append(torch.tensor(
        a_opponent, dtype=torch.long).to(self.device))

      ep_ret += reward[0]
      obs = next_obs

      # 4. Train Step (Optional: can also be moved outside the loop)
      self.global_step += 1
      Q_loss, model_loss = self.update()

      if Q_loss is not None and self.global_step % 100 == 0:
        print(f"Step {self.global_step}: Q_loss={Q_loss:.5f}, Model_loss={model_loss:.5f} "
              f"Tau={self._tau():.2f}")

      if done:
        break

    current_true_goal_pos = None
    next_map = np.zeros((H, W), dtype=np.float32)

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

    for t in episode_transitions:
      self.replay.push(t)

    return {"return": ep_ret, "steps": step + 1}

  def run_test_episode(self, max_steps: Optional[int] = None, render: bool = False, zigzag: bool = False) -> Dict[str, float]:
    self.opponent_agent = SimpleAgent(1)
    if zigzag:
      self.opponent_agent = ZigZagAgent(1)
    obs = self.env.reset()
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
      a_opponent = self.opponent_agent.select_action(obs[1])
      actions = {0: a, 1: a_opponent}

      if render:
        self.heatmap_q_values(
          g_map, f"./diagrams_{self.args.folder_id}/q_heatmap_step{self.global_step + step}.png")
        self.heatmap_subgoal(
          g_map, f"./diagrams_{self.args.folder_id}/gmap_step{self.global_step + step}.png")
        SimpleForagingEnv.render_from_obs(obs[0])

      next_obs, reward, done, info = self.env.step(actions)

      history["states"].append(
        torch.from_numpy(obs[0]).float().to(self.device))
      history["actions"].append(torch.tensor(
        a_opponent, dtype=torch.long).to(self.device))

      ep_ret += reward[0]
      obs = next_obs

      if done:
        break

    return {"return": ep_ret, "steps": step + 1}
