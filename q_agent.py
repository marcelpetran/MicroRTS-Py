from typing import Deque, Dict, List, Tuple, Optional
import random
from collections import deque

from omg_args import OMGArgs

from simple_foraging_env import SimpleAgent, RandomAgent, SimpleForagingEnv, ZigZagAgent

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ------ helpers ------
def to_tensor(x, device):
  if isinstance(x, torch.Tensor):
    return x.to(device)
  return torch.tensor(x, dtype=torch.float32, device=device)


def flatten_state(obs: np.ndarray) -> torch.Tensor:
  # obs: (H, W, F) -> (F, H, W) for Conv, or keep (H*W*F) for MLP.
  # return torch.from_numpy(obs).float().permute(2, 0, 1)  # (F, H, W) for Conv
  return torch.from_numpy(obs).float()


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
    self.latent_dim = args.latent_dim
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

  # ------------- acting -------------

  def _choose_action(self, qvals: torch.Tensor, beta: float, eval=False) -> int:
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
    with torch.no_grad():
      g_map = self.model(x, history)
    qvals = self.q(x, g_map)
    a = self._choose_action(qvals, self._tau(), eval)
    return a, g_map.squeeze(0)

  # ------------- training -------------

  def _compute_targets(self, batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
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
                        for b in batch], dim=0).to(self.device).unsqueeze(1)

    # 1. Q(s, g, a)
    q_sa = self.q(s, g_map).gather(1, a.unsqueeze(1)).squeeze(1)

    # 2. Target = r + gamma * max_a' Q_tgt(s', g, a')
    with torch.no_grad():
      q_val = self.q(sp, g_map)
      noise = torch.rand_like(q_val) * 1e-6
      best_actions = (q_val + noise).argmax(dim=1, keepdim=True)

      q_next = self.q_tgt(sp, g_map).gather(1, best_actions).squeeze(1)

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
    q_sa, target = self._compute_targets(batch_list)
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
          "history": self._collate_history([b["history"] for b in valid_batch]),
          "true_goal_map": torch.stack([torch.from_numpy(b["true_goal_map"]).float() for b in valid_batch], dim=0).to(self.device)
      }
      model_loss = self.model.train_step(om_batch, self)
    else:
      model_loss = 0.0

    # --- 3. Target Update ---
    with torch.no_grad():
      for param, target_param in zip(self.q.parameters(), self.q_tgt.parameters()):
        target_param.data.mul_(1 - self.args.tau_soft)
        target_param.data.add_(self.args.tau_soft * param.data)

    return loss.item(), model_loss

  def _collate_history(self, histories: List[Dict]) -> Dict[str, torch.Tensor]:
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
    null_state = torch.zeros(*self.args.state_shape, device=self.device)
    null_action = torch.tensor(0, device=self.device)

    for h in histories:
      num_to_pad = max_len - len(h.get("states", []))
      # Handle conversion safely whether they are Tensors or Numpy arrays
      states = [to_tensor(s, self.device) for s in h.get("states", [])]
      actions = [to_tensor(a, self.device) for a in h.get("actions", [])]
      if num_to_pad > 0:
        states.extend([null_state] * num_to_pad)
        actions.extend([null_action] * num_to_pad)

      padded_states_list.append(torch.stack(states, dim=0))
      padded_actions_list.append(torch.stack(actions, dim=0))

    final_padded_states = torch.stack(padded_states_list, dim=0)
    final_padded_actions = torch.stack(padded_actions_list, dim=0)

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

      # 1. Predict the Goal Map (Using Transformer or Oracle)
      # Note: You should pass the predicted map to select_action now!
      # For now, let's assume get_goal_map handles the Teacher Forcing mix
      g_map = self.model(obs[0], current_history)

      a = self.select_action(obs[0], g_map)
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
          # Keep the map used during the episode for Q-learning
          "rollout_goal_map": g_map.cpu().numpy(),
          "history": {k: [t.clone() for t in v] for k, v in current_history.items()},
      }
      episode_transitions.append(transition)

      # 3. Update history
      history["states"].append(torch.from_numpy(obs[0]).float())
      history["actions"].append(torch.tensor(a_opponent, dtype=torch.long))

      ep_ret += reward[0]
      obs = next_obs

      # 4. Train Step (Optional: can also be moved outside the loop)
      self.global_step += 1
      Q_loss, model_loss = self.update()

      if Q_loss is not None and self.global_step % 100 == 0:
        print(f"Step {self.global_step}: Q_loss={Q_loss:.5f}, "
              f"Tau={self._tau():.2f}, Gmix_eps={self._gmix_eps():.2f}")

      if done:
        break


    current_true_goal_pos = None

    # Walk backward through the episode
    for t in reversed(episode_transitions):

      # Did the opponent get a reward this step?
      if t["opp_reward"] > 0:
        # The opponent just ate a food!
        # In the 'next_state', the opponent is standing on the food location.
        # Channel 3 is Agent 2 (Opponent)
        opp_pos_indices = np.argwhere(t["next_state"][:, :, 3] == 1)
        if len(opp_pos_indices) > 0:
          current_true_goal_pos = tuple(opp_pos_indices[0])

      # Assign the goal to this step
      if current_true_goal_pos is not None:
        # Create a 1-hot spatial mask
        true_map = np.zeros((H, W), dtype=np.float32)
        true_map[current_true_goal_pos[0], current_true_goal_pos[1]] = 1.0

        t["true_goal_map"] = true_map
        t["valid_for_transformer"] = True  # We have proof of intent!
      else:
        # Opponent hasn't gotten a reward yet in this backward chain
        # (e.g. they failed, or this is the end of the episode and they missed)
        t["true_goal_map"] = np.zeros((H, W), dtype=np.float32)
        t["valid_for_transformer"] = False  # Do NOT train Transformer on this

      # Remove the temp tracking variables to save memory
      del t["opp_reward"]

      # Push to replay buffer
      # Note: Your ReplayBuffer.push() needs to accept the new keys:
      # rollout_goal_map, true_goal_map, valid_for_transformer
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

    self.model.eval()
    for step in range(max_steps or 500):
      current_history = {k: list(v) for k, v in history.items()}

      a, g_map = self.select_action(
        obs[0], current_history, eval=True)
      a_opponent = self.opponent_agent.select_action(obs[1])
      actions = {0: a, 1: a_opponent}

      if render:
        self.heatmap_q_values(
          g_map, f"./diagrams_{self.args.folder_id}/q_heatmap_step{self.global_step + step}.png")
        SimpleForagingEnv.render_from_obs(obs[0])

      next_obs, reward, done, info = self.env.step(actions)

      history["states"].append(torch.from_numpy(obs[0]).float())
      history["actions"].append(torch.tensor(a_opponent, dtype=torch.long))

      ep_ret += reward[0]
      obs = next_obs

      if done:
        break

    return {"return": ep_ret, "steps": step + 1}

  def visualize_prior(self, reset_global_counter: bool = True):
    """
    Run 1 episode and visualize subgoals sampled from the prior model.
    """
    self.agent1 = SimpleAgent(0)
    self.agent2 = SimpleAgent(1)
    obs = self.env.reset()
    done = False
    while not done:
      a1 = self.agent1.select_action(obs[0])
      a2 = self.agent2.select_action(obs[1])
      actions = {0: a1, 1: a2}
      next_obs, reward, done, info = self.env.step(actions)

      with torch.no_grad():
        self.model.prior_model.eval()
        recon_logits, _, _ = self.model.prior_model(
            torch.from_numpy(obs[0]).float().unsqueeze(0).to(self.device)
        )
        self.model.visualize_subgoal_logits(
          obs[0], recon_logits, f"./diagrams_{self.args.folder_id}/subgoal_logits_prior_step{self.global_step}.png")

      obs = next_obs
      self.global_step += 1

    if reset_global_counter:
      self.global_step = 0
