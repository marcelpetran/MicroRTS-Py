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

class SLnet(nn.Module):
  """
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

    self.value_head = nn.Sequential(
        nn.Linear(self.flat_dim, args.qnet_hidden),
        nn.ReLU(),
        nn.Linear(args.qnet_hidden, self.action_dim)
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
    logits = self.value_head(features)  # (B, action_dim)

    return logits


class SLBuffer:
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


class SLAgent:
  """
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
    self.q = SLnet(args).to(self.device)
    self.opt = torch.optim.Adam(self.q.parameters(), lr=self.args.lr)

    # Replay
    self.replay = SLBuffer(200_000) # extra large buffer to store all historical policies for Fictitious Play

    # Schedules
    self.global_step = 0

  def reset(self):
    pass

  # ------------- acting -------------
  @torch.no_grad()
  def select_action(self, s_t: np.ndarray, eval=False) -> Tuple[int, float]:
    """
    Samples an action from the learned average probability distribution.
    """
    self.q.eval()
    s = torch.from_numpy(s_t).float().unsqueeze(0).to(self.device)
    
    logits = self.q(s) # (1, action_dim)
    
    if eval:
        action = torch.argmax(logits, dim=1).item()
        entropy = 0.0
        return action, entropy

    dist = Categorical(logits=logits)
    action = dist.sample().item()
    entropy = dist.entropy().item()
    
    self.q.train()
    return action, entropy

  # ------------- training -------------
  def update(self):
    if len(self.replay) < self.args.min_replay:
      return None  # not enough data yet

    if self.global_step % self.args.train_every != 0:
      return None  # only train every few steps

    batch = self.replay.sample(self.args.batch_size)
    s = torch.from_numpy(np.stack([b["state"]
                         for b in batch])).float().to(self.device)

    a = torch.from_numpy(
      np.array([b["action"] for b in batch], dtype=np.int64)).to(self.device)
    
    # For SL, we treat the action as the "target" and use cross-entropy loss
    logits = self.q(s)  # (B, action_dim)
    loss = F.cross_entropy(logits, a)

    self.opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
    self.opt.step()

    return loss.item()