from typing import Deque, Dict, List, Tuple, Optional
import random
from collections import deque
from omg_args import OMGArgs

from simple_foraging_env import SimpleAgent, RandomAgent, SimpleForagingEnv
from priority_replay_buffer import PrioritizedReplayBuffer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    self.net = nn.Sequential(
        nn.Linear(self.state_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, self.action_dim),
    )

  def forward(self, batch: torch.Tensor) -> torch.Tensor:
    B, H, W, F_dim = batch.shape  # (B, H, W, F) float
    x = batch.view(B, -1)         # (B, H*W*F)
    return self.net(x)  # (B, A)


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
    # self.replay = ReplayBuffer(self.args.capacity)
    self.replay = PrioritizedReplayBuffer(self.args.capacity)

    # Schedules
    self.global_step = 0

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
      print((qvals + gumbel_noise).squeeze(0).detach().numpy())
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
      q_next = self.q_tgt(sp).max(dim=1).values
      target = r + (1.0 - done) * self.args.gamma * q_next
    return q_sa, target

  def update(self):
    if len(self.replay) < self.args.min_replay:
      return None  # not enough data yet

    if self.global_step % self.args.train_every != 0:
      return None  # only train every few steps

    batch_list, is_weights, tree_indices = self.replay.sample(self.args.batch_size)

    is_weights = torch.tensor(is_weights, dtype=torch.float32, device=self.args.device)

    # --- 1. Update the Q-Network ---
    q_sa, target = self._compute_targets(batch_list)
    with torch.no_grad():
        td_errors = (q_sa - target).cpu().numpy()
    
    loss_per_element = (q_sa - target) ** 2
    loss = (loss_per_element * is_weights).mean()

    self.opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
    self.opt.step()

    self.replay.update_priorities(tree_indices, td_errors)

    if self.global_step % self.args.target_update_every == 0:
      self.q_tgt.load_state_dict(self.q.state_dict())

    return loss.item()

  # ------------- rollout -------------

  def run_episode(self, max_steps: Optional[int] = None) -> Dict[str, float]:
    """
    Gathers a trajectory, stores future slices for subgoal selection,
    and trains the Q-network and OpponentModel.
    """
    self.opponent_agent = SimpleAgent(1, True)

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

  def run_test_episode(self, max_steps: Optional[int] = None, render: bool = False) -> Dict[str, float]:
    self.opponent_agent = SimpleAgent(1, True)
    obs = self.env.reset()
    done = False
    ep_ret = 0.0

    if render:
      SimpleForagingEnv.render_from_obs(obs[0])

    for step in range(max_steps or 500):

      a = self.select_action(obs[0], True)
      a_opponent = self.opponent_agent.select_action(obs[1])
      actions = {0: a, 1: a_opponent}
      next_obs, reward, done, info = self.env.step(actions)
      ep_ret += reward[0]
      obs = next_obs
      if render:
        SimpleForagingEnv.render_from_obs(obs[0])

      if done:
        break

    return {"return": ep_ret, "steps": step + 1}
