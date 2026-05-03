from typing import Dict, List, Tuple
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import wandb

from omg_args import OMGArgs
from buffers import ReplayBuffer, ReservoirBuffer
from networks import QNetClassic, SLnet


class FSPAgentClassic:
  """
  Unified Fictitious Self-Play Agent using classic Q-Learning.
  Contains both RL (Best Response) and SL (Average Strategy) components.
  """

  def __init__(self, env, args: OMGArgs = OMGArgs()):
    self.env = env
    self.args = args
    self.device = torch.device(args.device)

    if args.state_shape is None:
      obs = self.env.reset()
      H, W, F_dim = obs.shape
      self.args.state_shape = (H, W, F_dim)

    if not hasattr(self.env, "action_space") or self.env.action_space is None:
      raise ValueError("Env must have action_space (list or int).")
    self.args.action_dim = len(self.env.action_space) if hasattr(
      self.env.action_space, "__len__") else int(self.env.action_space)

    # RL Networks & Optimizer
    self.q = QNetClassic(args).to(self.device)
    self.q_tgt = QNetClassic(args).to(self.device)
    self.q_tgt.load_state_dict(self.q.state_dict())
    self.opt_rl = torch.optim.Adam(self.q.parameters(), lr=self.args.lr)

    # SL Network & Optimizer
    self.sl = SLnet(args).to(self.device)
    self.opt_sl = torch.optim.Adam(self.sl.parameters(), lr=self.args.lr)

    # FSP Buffers
    self.rl_replay = ReplayBuffer(self.args.capacity)
    self.sl_replay = ReservoirBuffer(self.args.sl_capacity)

    self.global_step = 0
    self.is_frozen_as_sl = False

  def reset(self):
    pass

  def freeze_as_sl_opponent(self):
    """Forces the agent to only act using the Average Strategy (for Phase 2 evaluation)."""
    self.is_frozen_as_sl = True

  # ------------- Tau Schedules --------------
  def _tau(self) -> float:
    t = min(self.global_step, self.args.tau_decay_steps)
    return self.args.tau_end + (self.args.tau_start - self.args.tau_end) * (1 - t / self.args.tau_decay_steps)

  # ------------- Action Selection Methods -------------

  def _choose_q_action(self, qvals: torch.Tensor, beta: float, eval=False) -> int:
    gumbel_noise = -beta * torch.empty_like(qvals).exponential_().log()
    if eval:
      dist = F.softmax(qvals / beta, dim=-1)
      return int(torch.multinomial(dist, num_samples=1).item())
    return int(torch.argmax(qvals + gumbel_noise))

  @torch.no_grad()
  def select_rl_action(self, s_t: np.ndarray, eval=False) -> Tuple[int, float]:
    """Calculates the Best Response action using classic Q-learning."""
    self.q.eval()
    x = torch.from_numpy(s_t).float().unsqueeze(0).to(self.device)

    qvals = self.q(x)
    tau = 0.05 if eval else self._tau()
    entropy = Categorical(logits=qvals / tau).entropy().item()

    a = self._choose_q_action(qvals, tau, eval)
    self.q.train()
    return a, entropy

  @torch.no_grad()
  def select_sl_action(self, s_t: np.ndarray, eval=False) -> Tuple[int, float]:
    """Calculates the Average Strategy action using Supervised Learning."""
    self.sl.eval()
    s = torch.from_numpy(s_t).float().unsqueeze(0).to(self.device)
    logits = self.sl(s)
    entropy = 0.0

    if eval:
      action = torch.argmax(logits, dim=1).item()
    else:
      dist = Categorical(logits=logits)
      action = dist.sample().item()
      entropy = dist.entropy().item()

    self.sl.train()
    return action, entropy

  def select_action(self, s_t: np.ndarray, eval=False) -> Tuple[int, float]:
    """Generic interface for opponents. Defaults to SL if frozen."""
    if self.is_frozen_as_sl:
      return self.select_sl_action(s_t, eval=eval)
    a, ent = self.select_rl_action(s_t, eval=eval)
    return a, ent

  # ------------- RL Update Logic -------------

  def compute_targets(self, batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
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

    q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
      q_val = self.q(sp)
      noise = torch.rand_like(q_val) * 1e-6
      best_actions = (q_val + noise).argmax(dim=1, keepdim=True)
      q_next = self.q_tgt(sp).gather(1, best_actions).squeeze(1)
      target = r + (1.0 - done) * self.args.gamma * q_next
      target = torch.clamp(target, min=-15.0, max=15.0)

    return q_sa, target

  def update_rl(self):
    """Updates the Q-network."""
    if len(self.rl_replay) < self.args.min_replay:
      return None

    batch_list = self.rl_replay.sample(self.args.batch_size)

    q_sa, target = self.compute_targets(batch_list)
    loss = F.smooth_l1_loss(q_sa, target, reduction='mean')
    loss_val = loss.item()

    self.opt_rl.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
    self.opt_rl.step()

    with torch.no_grad():
      for param, target_param in zip(self.q.parameters(), self.q_tgt.parameters()):
        target_param.lerp_(param, self.args.tau_soft)

    return loss_val

  # ------------- SL Update Logic -------------

  def update_sl(self):
    """Updates the SL Average Strategy network."""
    if len(self.sl_replay) < self.args.min_replay:
      return None

    batch = self.sl_replay.sample(self.args.batch_size)
    s = torch.from_numpy(np.stack([b["state"]
                         for b in batch])).float().to(self.device)
    a = torch.from_numpy(
      np.array([b["action"] for b in batch], dtype=np.int64)).to(self.device)

    logits = self.sl(s)
    loss = F.cross_entropy(logits, a)

    self.opt_sl.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(self.sl.parameters(), 5.0)
    self.opt_sl.step()

    return loss.item()

  # ------------- Data Generation (Rollout) -------------

  def run_fsp_episode(self, opponent_agent, eta: float, max_steps: int = 500) -> Dict[str, float]:
    """
    Executes a rollout mixing RL and SL policies using eta. 
    Stores specific transitions in RL and SL buffers.
    """
    obs = self.env.reset()
    if random.random() < 0.3:
      obs = self.env.reset_random_spawn()
    elif random.random() < 0.5:
      obs = self.env.swap_agents()
    opponent_agent.reset()

    done = False
    ep_ret, opp_ret, rl_ep_entropy, sl_ep_entropy = 0.0, 0.0, 0.0, 0.0

    for step in range(max_steps):
      # 1. Compute both RL (Best Response) and SL (Average) actions
      rl_a, rl_entropy = self.select_rl_action(obs[0])
      sl_a, sl_entropy = self.select_sl_action(obs[0])
      rl_ep_entropy += rl_entropy
      sl_ep_entropy += sl_entropy

      # 2. Mix policies based on eta (Fictitious Play)
      if random.random() < eta:
        a = rl_a
        is_rl = True
      else:
        a = sl_a
        is_rl = False

      # Opponent acts
      if hasattr(opponent_agent, 'is_frozen_as_sl') and opponent_agent.is_frozen_as_sl:
        a_opponent, _ = opponent_agent.select_sl_action(obs[1])
        opp_is_rl = False
      else:
        opp_rl_a, _ = opponent_agent.select_rl_action(obs[1])
        opp_sl_a, _ = opponent_agent.select_sl_action(obs[1])
        if random.random() < eta:
          a_opponent = opp_rl_a
          opp_is_rl = True
        else:
          a_opponent = opp_sl_a
          opp_is_rl = False

      actions = {0: a, 1: a_opponent}

      next_obs, reward, done, info = self.env.step(actions)

      # 3. Store in SL Buffer ONLY if the action was a Best Response (NFSP Standard)
      if is_rl:
        self.sl_replay.push({"state": obs[0].copy(), "action": a})
      if opp_is_rl and opponent_agent is self:
        self.sl_replay.push({"state": obs[1].copy(), "action": a_opponent})

      # 4. Store in RL Buffer for standard Q-learning
      transition = {
          "state": obs[0].copy(),
          "action": a,
          "opp_action": a_opponent,
          "reward": float(reward[0]),
          "opp_reward": float(reward[1]),
          "next_state": next_obs[0].copy(),
          "done": bool(done),
      }
      self.rl_replay.push(transition)

      ep_ret += reward[0]
      opp_ret += reward[1]
      obs = next_obs
      self.global_step += 1

      if done:
        break

    return {
        "return": ep_ret,
        "steps": step + 1,
        "opp_return": opp_ret,
        "avg_rl_entropy": rl_ep_entropy / (step + 1),
        "avg_sl_entropy": sl_ep_entropy / (step + 1)
    }

  def run_test_episode(self, opponent_agent, use_sl: bool = True, max_steps: int = 500) -> Dict[str, float]:
    """Evaluation rollout. Defaults to Average Strategy (SL) as per FSP standards."""
    obs = self.env.reset()
    opponent_agent.reset()
    done = False
    ep_ret, opp_ret, ep_entropy = 0.0, 0.0, 0.0

    for step in range(max_steps):
      if use_sl:
        a, step_entropy = self.select_sl_action(obs[0], eval=True)
      else:
        a, step_entropy = self.select_rl_action(obs[0], eval=True)

      if hasattr(opponent_agent, 'is_frozen_as_sl') and opponent_agent.is_frozen_as_sl:
        a_opponent, _ = opponent_agent.select_sl_action(obs[1], eval=True)
      else:
        a_opponent, _ = opponent_agent.select_action(obs[1], eval=True)

      actions = {0: a, 1: a_opponent}
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
        "avg_rl_entropy": ep_entropy / (step + 1) if not use_sl else 0.0,
        "avg_sl_entropy": ep_entropy / (step + 1) if use_sl else 0.0
    }
