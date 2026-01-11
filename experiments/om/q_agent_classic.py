from typing import Deque, Dict, List, Tuple, Optional
import random
from collections import deque
from omg_args import OMGArgs

from simple_foraging_env import SimpleAgent, RandomAgent, SimpleForagingEnv, ZigZagAgent
from priority_replay_buffer import PrioritizedReplayBuffer

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


class QNetClassic(nn.Module):
  """
  Simple MLP Q-network for Q(s, a)
  state_shape: (H, W, F)
  latent_dim: dimension of latent opponent representation
  action_dim: number of discrete actions
  Returns Q-values for all actions.
  """

  def __init__(self, args: OMGArgs):
    super().__init__()
    H, W, F_dim = args.state_shape
    self.state_dim = H * W * F_dim
    # WARNING: might not work with complex action spaces
    self.action_dim = args.action_dim
    hidden = args.qnet_hidden
    cnn_2d_out = 32
    cnn_hidden = 16

    self.cnn = nn.Sequential(
        nn.Conv2d(F_dim, cnn_hidden, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(cnn_hidden, cnn_2d_out, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Flatten()
    )
    cnn_out_dim = cnn_2d_out * H * W

    self.head = nn.Sequential(
        nn.Linear(cnn_out_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, self.action_dim)
    )
    self.apply(self._init_weights)

  def _init_weights(self, m):
    if isinstance(m, nn.Linear):
      nn.init.xavier_uniform_(m.weight)
      if m.bias is not None:
        nn.init.constant_(m.bias, 0.01)

  def forward(self, batch: torch.Tensor) -> torch.Tensor:
    # Batch shape: (B, H, W, F) -> Permute to (B, F, H, W) for Conv2d
    s = batch.permute(0, 3, 1, 2)
    x = self.cnn(s)
    return self.head(x)


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


class QLearningAgentClassic:
  """
  Q(s, a) Classic agent without opponent modeling.
  """

  def __init__(self, env, args: OMGArgs = OMGArgs()):
    self.env = env
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
    self.q = QNetClassic(args).to(self.device)
    self.q_tgt = QNetClassic(args).to(self.device)
    self.q_tgt.load_state_dict(self.q.state_dict())
    self.opt = torch.optim.Adam(self.q.parameters(), lr=self.args.lr)

    # Replay
    self.replay = ReplayBuffer(self.args.capacity)
    # self.replay = PrioritizedReplayBuffer(self.args.capacity)

    # Schedules
    self.global_step = 0

  @torch.no_grad()
  def heatmap_q_values(self, filename: str = "q_heatmap.png"):
    """
    Utility to visualize Q-values as a heatmap over the grid for a given state and subgoal.

    Args:
        state_hwf (np.ndarray): The current state grid, shape (H, W, F).
        g (torch.Tensor): The inferred subgoal, shape (1, latent_dim).
        filename (str): Path to save the heatmap image.
    """
    self.q.eval()
    H, W, _ = self.args.state_shape

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

      q_values = self.q(s_tensor)  # (1, num_actions)

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

  # ------------- epsilon schedules --------------

  def _tau(self) -> float:
    t = min(self.global_step, self.args.selector_tau_decay_steps)
    return self.args.selector_tau_end + (self.args.selector_tau_start - self.args.selector_tau_end) * (1 - t / self.args.selector_tau_decay_steps)

  def _eps(self) -> float:
    t = min(self.global_step, self.args.eps_decay_steps)
    return self.args.eps_end + (self.args.eps_start - self.args.eps_end) * (1 - t / self.args.eps_decay_steps)

  # ------------- acting -------------

  def _choose_action(self, qvals: torch.Tensor, beta: float, eval) -> int:
    gumbel_noise = -beta * torch.empty_like(qvals).exponential_().log()
    if eval == True:
      noise = torch.rand_like(qvals) * 1e-6
      return int(torch.argmax(qvals + noise))
    return int(torch.argmax(qvals + gumbel_noise))

  def select_action(self, s_t: np.ndarray, eval=False) -> Tuple[int, torch.Tensor]:
    """
    (interaction phase) Infer g_hat and act eps-greedily on Q(s,g_hat,*)
    """
    s = torch.from_numpy(s_t).float().unsqueeze(0).to(self.device)
    qvals = self.q(s)
    return self._choose_action(qvals, self._tau(), eval)

  # ------------- training -------------

  def _compute_targets(self, batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implements Eq. (4) and (8) mixing between g_hat and g_bar with a decaying switch.
    """
    B = len(batch)
    H, W, F_dim = self.args.state_shape

    # Stack current/next states
    s = torch.stack([torch.from_numpy(b["state"]).float()
                    for b in batch], dim=0).to(self.device)         # (B,H,W,F)
    sp = torch.stack([torch.from_numpy(b["next_state"]).float()
                     for b in batch], dim=0).to(self.device)   # (B,H,W,F)
    a = torch.tensor([b["action"] for b in batch], dtype=torch.long,
                     device=self.device)                  # (B,)
    r = torch.tensor([b["reward"] for b in batch],
                     dtype=torch.float32, device=self.device)  # (B,)
    done = torch.tensor([b["done"] for b in batch],
                        dtype=torch.float32, device=self.device)  # (B,)

    # Q(s,a) and target r + gamma * max_{a'} Q(s',a')
    q_sa = self.q(s).gather(1, a.view(-1, 1)).squeeze(1)
    with torch.no_grad():
      q_val = self.q(sp)
      noise = torch.rand_like(q_val) * 1e-6
      best_actions = (q_val + noise).argmax(dim=1, keepdim=True)
      q_next = self.q_tgt(sp).gather(1, best_actions).squeeze(1)
      target = r + (1.0 - done) * self.args.gamma * q_next
      target = torch.clamp(target, min=-5.0, max=5.0)
    return q_sa, target

  def update(self):
    if len(self.replay) < self.args.min_replay:
      return None  # not enough data yet

    if self.global_step % self.args.train_every != 0:
      return None  # only train every few steps

    # batch_list, is_weights, tree_indices = self.replay.sample(self.args.batch_size)
    batch_list = self.replay.sample(self.args.batch_size)

    # is_weights = torch.tensor(is_weights, dtype=torch.float32, device=self.args.device)

    # --- 1. Update the Q-Network ---
    q_sa, target = self._compute_targets(batch_list)
    with torch.no_grad():
      td_errors = (q_sa - target).cpu().numpy()
    # MSE loss
    # loss_per_element = (q_sa - target) ** 2
    # loss = (loss_per_element * is_weights).mean()

    # Huber loss
    loss = F.smooth_l1_loss(q_sa, target, reduction='mean')
    # loss_per_element = F.smooth_l1_loss(q_sa, target, reduction='none')
    # loss = (loss_per_element * is_weights).mean()

    self.opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
    self.opt.step()

    # self.replay.update_priorities(tree_indices, td_errors)

    if self.global_step % self.args.target_update_every == 0:
      self.q_tgt.load_state_dict(self.q.state_dict())

    return loss.item()

  # ------------- rollout -------------

  def run_episode(self, max_steps: Optional[int] = None) -> Dict[str, float]:
    """
    Gathers a trajectory, stores future slices for subgoal selection,
    and trains the Q-network and OpponentModel.
    """
    self.opponent_agent = SimpleAgent(1)

    obs = self.env.reset_random_spawn(0)
    done = False
    ep_ret = 0.0

    for step in range(max_steps or 500):

      a = self.select_action(obs[0])
      a_opponent = self.opponent_agent.select_action(obs[1])
      actions = {0: a, 1: a_opponent}
      next_obs, reward, done, info = self.env.step(actions)

      # Store the current step's info
      step_info = {
          "state": obs[0].copy(),
          "action": a,
          "reward": float(reward[0]),
          "next_state": next_obs[0].copy(),
          "done": bool(done),
      }

      self.replay.push(step_info)

      ep_ret += reward[0]
      obs = next_obs

      self.global_step += 1
      Q_loss = self.update()

      if Q_loss is not None and self.global_step % 100 == 0:
        print(
          f"Step {self.global_step}: Q_loss={Q_loss:.4f}, Tau={self._tau():.3f}")

      if done:
        break

    return {"return": ep_ret, "steps": step + 1}

  def run_test_episode(self, max_steps: Optional[int] = None, render: bool = False, zigzag: bool = False) -> Dict[str, float]:
    self.opponent_agent = SimpleAgent(1)
    if zigzag:
      self.opponent_agent = ZigZagAgent(1)
    obs = self.env.reset()
    done = False
    ep_ret = 0.0

    for step in range(max_steps or 500):

      a = self.select_action(obs[0], True)
      a_opponent = self.opponent_agent.select_action(obs[1])
      actions = {0: a, 1: a_opponent}

      if render:
        SimpleForagingEnv.render_from_obs(obs[0])
        self.heatmap_q_values(
          f"./diagrams_{self.args.folder_id}/q_heatmap_step{self.global_step + step}.png")

      next_obs, reward, done, info = self.env.step(actions)
      ep_ret += reward[0]
      obs = next_obs

      if done:
        break

    return {"return": ep_ret, "steps": step + 1}
