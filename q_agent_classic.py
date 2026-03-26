from math import e
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
    self.action_dim = args.action_dim
    cnn_hidden = args.cnn_hidden
    self.flat_dim = cnn_hidden * H * W
    self.cnn = nn.Sequential(
        nn.Conv2d(F_dim, 32, kernel_size=3, padding=1),
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
    self.apply(self._init_weights)

  def _init_weights(self, m):
    if isinstance(m, nn.Linear):
      nn.init.xavier_uniform_(m.weight)
      if m.bias is not None:
        nn.init.constant_(m.bias, 0.01)

  def forward(self, batch: torch.Tensor) -> torch.Tensor:
    # Batch shape: (B, H, W, F) -> Permute to (B, F, H, W) for Conv2d
    s = batch.permute(0, 3, 1, 2)
    features = self.cnn(s)

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


class QLearningAgentClassic:
  """
  Q(s, a) Classic agent without opponent modeling.
  """

  def __init__(self, env, args: OMGArgs = OMGArgs()):
    self.env = env
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
    self.q = QNetClassic(args).to(self.device)
    self.q_tgt = QNetClassic(args).to(self.device)
    self.q_tgt.load_state_dict(self.q.state_dict())
    self.opt = torch.optim.Adam(self.q.parameters(), lr=self.args.lr)

    # Replay
    self.replay = ReplayBuffer(self.args.capacity)

    # Schedules
    self.global_step = 0

  def reset(self):
    pass

   # ------------- env interaction helpers -------------

  @torch.no_grad()
  def heatmap_q_values(self, filename: str = "q_heatmap.png", save: bool = True):
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
    for pos in food_pos:
      ax1.scatter(pos[1], pos[0], color='green',
                  marker='o', s=50, label='Food')
    for pos in wall_pos:
      ax1.scatter(pos[1], pos[0], color='black',
                  marker='s', s=50, label='Wall')
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
    if save:
      plt.savefig(filename)
    else:
      plt.show()
    plt.close()

  # ------------- epsilon schedules --------------

  def _tau(self) -> float:
    t = min(self.global_step, self.args.tau_decay_steps)
    return self.args.tau_end + (self.args.tau_start - self.args.tau_end) * (1 - t / self.args.tau_decay_steps)

  def _eps(self) -> float:
    t = min(self.global_step, self.args.eps_decay_steps)
    return self.args.eps_end + (self.args.eps_start - self.args.eps_end) * (1 - t / self.args.eps_decay_steps)

  # ------------- acting -------------

  def choose_action(self, qvals: torch.Tensor, beta: float, eval) -> int:
    gumbel_noise = -beta * torch.empty_like(qvals).exponential_().log()

    if eval == True:
      dist = F.softmax(qvals / beta, dim=-1)
      return int(torch.multinomial(dist, num_samples=1).item())

    return int(torch.argmax(qvals + gumbel_noise))

  @torch.no_grad()
  def select_action(self, s_t: np.ndarray, eval=False) -> int:
    """
    (interaction phase) act eps-greedily on Q(s, *)
    """
    s = torch.from_numpy(s_t).float().unsqueeze(0).to(self.device)
    qvals = self.q(s)

    tau = 0.05 if eval else self._tau()
    entropy = Categorical(logits=qvals / tau).entropy().item()

    return self.choose_action(qvals, tau, eval), entropy

  # ------------- training -------------

  def compute_targets(self, batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Implements Eq. (4) and (8) mixing between g_hat and g_bar with a decaying switch.
    """
    B = len(batch)
    H, W, F_dim = self.args.state_shape

    s = torch.from_numpy(np.stack([b["state"]
                         for b in batch])).float().to(self.device)
    sp = torch.from_numpy(np.stack([b["next_state"]
                          for b in batch])).float().to(self.device)

    a = torch.from_numpy(
      np.array([b["action"] for b in batch], dtype=np.int64)).to(self.device)
    r = torch.from_numpy(
      np.array([b["reward"] for b in batch], dtype=np.float32)).to(self.device)
    done = torch.from_numpy(
      np.array([b["done"] for b in batch], dtype=np.float32)).to(self.device)

    # Q(s,a) and target r + gamma * max_{a'} Q(s',a')
    q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)

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

    batch_list = self.replay.sample(self.args.batch_size)

    q_sa, target = self.compute_targets(batch_list)
    loss = F.smooth_l1_loss(q_sa, target, reduction='mean')

    if loss.item() < self.args.aux_loss_threshold:
      return 0.0

    self.opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
    self.opt.step()

    with torch.no_grad():
      for param, target_param in zip(self.q.parameters(), self.q_tgt.parameters()):
        target_param.lerp_(param, self.args.tau_soft)

    return loss.item()

  def load_historical_policy(self, state_dict: dict, om_state_dict: dict = None):
    """Loads frozen historical weights for Fictitious Play."""
    self.q.load_state_dict(state_dict)
    self.q.eval()  # Freeze layers like BatchNorm/Dropout if you add them later

    if om_state_dict is not None and hasattr(self, 'model'):
      self.model.inference_model.load_state_dict(om_state_dict)
      self.model.inference_model.eval()

  # ------------- rollout -------------

  def run_episode(self, opponent_agent, max_steps: Optional[int] = None) -> Dict[str, float]:
    """
    Gathers a trajectory and trains the Q-network.
    """
    opp_loss_val = 0.0
    obs = self.env.reset()
    if random.random() < 0.3:
      obs = self.env.reset_random_spawn()
    opponent_agent.reset()

    done = False
    ep_ret = 0.0
    opp_ret = 0.0
    ep_entropy = 0.0

    for step in range(max_steps or 500):
      a, step_entropy = self.select_action(obs[0])
      a_opponent, _ = opponent_agent.select_action(obs[1])

      actions = {0: a, 1: a_opponent}
      ep_entropy += step_entropy

      next_obs, reward, done, info = self.env.step(actions)

      # if hasattr(opponent_agent, 'replay'):
      #   opp_step_info = {
      #       "state": obs[1].copy(),
      #       "action": a_opponent,
      #       "reward": float(reward[1]),
      #       "next_state": next_obs[1].copy(),
      #       "done": bool(done),
      #   }
      #   opponent_agent.replay.push(opp_step_info)
      #   opponent_agent.global_step += 1

      #   opp_loss = opponent_agent.update()

      #   if opp_loss is not None:
      #     opp_loss_val = opp_loss

      step_info = {
          "state": obs[0].copy(),
          "action": a,
          "reward": float(reward[0]),
          "next_state": next_obs[0].copy(),
          "done": bool(done),
      }
      self.replay.push(step_info)

      ep_ret += reward[0]
      opp_ret += reward[1]
      obs = next_obs
      self.global_step += 1
      Q_loss = self.update()

      if Q_loss is not None and self.global_step % 100 == 0:
        if wandb.run is not None:
          wandb.log({
              "train/q_loss": Q_loss,
              "train/opp_q_loss": opp_loss_val,
              "train/tau": self._tau(),
              "step": self.global_step
          })

      if done:
        break

    return {
      "return": ep_ret,
      "steps": step + 1,
      "opp_return": opp_ret,
      "avg_entropy": ep_entropy / (step + 1)
    }

  def run_test_episode(self, opponent_agent, max_steps: Optional[int] = None, render: bool = False) -> Dict[str, float]:
    obs = self.env.reset()
    opponent_agent.reset()
    done = False
    ep_ret = 0.0
    opp_ret = 0.0
    ep_entropy = 0.0

    for step in range(max_steps or 500):
      a, step_entropy = self.select_action(obs[0], eval=True)
      a_opponent, _ = opponent_agent.select_action(obs[1], eval=True)

      actions = {0: a, 1: a_opponent}

      if render:
        self.env.render()
        self.heatmap_q_values(
          f"./diagrams/{self.args.folder_id}/q_heatmap_step{self.global_step + step}.png")

      next_obs, reward, done, info = self.env.step(actions)
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
      "avg_entropy": ep_entropy / (step + 1)
    }
