from math import e
from typing import Deque, Dict, List, Tuple, Optional
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import wandb

from omg_args import OMGArgs
from simple_foraging_env import SimpleAgent, RandomAgent, SimpleForagingEnv

# ==========================================
# NETWORKS
# ==========================================

class QNet(nn.Module):
  """
  RL Network: Q(s, g, a)
  Learns the Best Response to the opponent's average strategy.
  """
  def __init__(self, args: OMGArgs):
    super().__init__()
    H, W, F_dim = args.state_shape
    self.state_dim = H * W * F_dim
    self.action_dim = args.action_dim
    cnn_hidden = args.cnn_hidden

    self.flat_dim = cnn_hidden * H * W
    input_channels = F_dim + 1  # State features + 1 channel for the subgoal heatmap

    self.cnn = nn.Sequential(
        nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, cnn_hidden, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(cnn_hidden, cnn_hidden, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten()
    )

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

  def forward(self, batch: torch.Tensor, g_map: torch.Tensor) -> torch.Tensor:
    s = batch.permute(0, 3, 1, 2)
    g = g_map.unsqueeze(1)
    x = torch.cat([s, g], dim=1)
    features = self.cnn(x)

    adv = self.advantage_head(features)
    val = self.value_head(features)
    q_vals = val + adv - adv.mean(dim=1, keepdim=True)
    return q_vals


class SLnet(nn.Module):
  """
  SL Network: Pi(a | s)
  Learns the agent's own average historical strategy.
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
    s = batch.permute(0, 3, 1, 2)
    features = self.cnn(s)
    logits = self.value_head(features)
    return logits

# ==========================================
# BUFFERS
# ==========================================

class ReplayBuffer:
  """Standard FIFO buffer for Q-learning (requires recent data)."""
  def __init__(self, capacity: int):
    self.capacity = capacity
    self.buf: Deque[Dict] = deque(maxlen=capacity)

  def push(self, item: Dict):
    self.buf.append(item)

  def sample(self, batch_size: int) -> List[Dict]:
    return random.sample(self.buf, batch_size)

  def __len__(self):
    return len(self.buf)

class ReservoirBuffer:
  """Reservoir Sampler for SL."""
  def __init__(self, capacity: int):
    self.capacity = capacity
    self.buf = []
    self.n_seen = 0

  def push(self, item: Dict):
    if len(self.buf) < self.capacity:
      self.buf.append(item)
    else:
      j = random.randint(0, self.n_seen)
      if j < self.capacity:
        self.buf[j] = item
    self.n_seen += 1

  def sample(self, batch_size: int) -> List[Dict]:
    return random.sample(self.buf, batch_size)

  def __len__(self):
    return len(self.buf)

# ==========================================
# FSP AGENT
# ==========================================

class FSPAgentOM:
  """
  Unified Fictitious Self-Play Agent with Opponent Modeling.
  Contains both RL (Best Response) and SL (Average Strategy) components.
  """
  def __init__(self, env, opponent_model, args: OMGArgs = OMGArgs()):
    self.env = env
    self.model = opponent_model
    self.args = args
    self.device = torch.device(args.device)

    if args.state_shape is None:
      obs = self.env.reset()
      H, W, F_dim = obs.shape
      self.args.state_shape = (H, W, F_dim)
        
    if not hasattr(self.env, "action_space") or self.env.action_space is None:
      raise ValueError("Env must have action_space (list or int).")
    self.args.action_dim = len(self.env.action_space) if hasattr(self.env.action_space, "__len__") else int(self.env.action_space)

    # RL Networks & Optimizer
    self.q = QNet(args).to(self.device)
    self.q_tgt = QNet(args).to(self.device)
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

  # ------------- Epsilon/Tau Schedules --------------
  def _eps(self) -> float:
    t = min(self.global_step, self.args.eps_decay_steps)
    return self.args.eps_end + (self.args.eps_start - self.args.eps_end) * (1 - t / self.args.eps_decay_steps)

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
  def select_rl_action(self, s_t: np.ndarray, history: Dict[str, torch.Tensor], eval=False) -> Tuple[int, torch.Tensor, float]:
    """Calculates the Best Response action using Q-learning and OM."""
    self.q.eval()
    x = torch.from_numpy(s_t).float().unsqueeze(0).to(self.device)
    
    g_logits = self.model(x, history, cached_features=True)
    g_map = F.softmax(g_logits.view(g_logits.shape[0], -1), dim=-1).view_as(g_logits)

    qvals = self.q(x, g_map)
    tau = 0.05 if eval else self._tau()
    entropy = Categorical(logits=qvals / tau).entropy().item()
    
    a = self._choose_q_action(qvals, tau, eval)
    self.q.train()
    return a, g_map.squeeze(0), entropy

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

  def select_action(self, s_t: np.ndarray, history: Dict[str, torch.Tensor] = None, eval=False) -> Tuple[int, float]:
    """Generic interface for opponents. Defaults to SL if frozen."""
    if self.is_frozen_as_sl:
      return self.select_sl_action(s_t, eval=eval)
    # If not frozen, standard routing (requires history for RL)
    a, _, ent = self.select_rl_action(s_t, history, eval=eval)
    return a, ent

  # ------------- RL Update Logic -------------
  
  def compute_targets(self, batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    s = torch.from_numpy(np.array([b["state"] for b in batch], dtype=np.float32)).to(self.device)
    sp = torch.from_numpy(np.array([b["next_state"] for b in batch], dtype=np.float32)).to(self.device)
    a = torch.from_numpy(np.array([b["action"] for b in batch], dtype=np.int64)).to(self.device)
    r = torch.from_numpy(np.array([b["reward"] for b in batch], dtype=np.float32)).to(self.device)
    done = torch.from_numpy(np.array([b["done"] for b in batch], dtype=np.float32)).to(self.device)

    g_map = torch.from_numpy(np.array([b["rollout_goal_map"] for b in batch], dtype=np.float32)).to(self.device)
    g_map_next = torch.from_numpy(np.array([b["rollout_goal_map_next"] for b in batch], dtype=np.float32)).to(self.device)

    q_sa = self.q(s, g_map).gather(1, a.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
      q_val = self.q(sp, g_map_next)
      noise = torch.rand_like(q_val) * 1e-6
      best_actions = (q_val + noise).argmax(dim=1, keepdim=True)
      q_next = self.q_tgt(sp, g_map_next).gather(1, best_actions).squeeze(1)
      target = r + (1.0 - done) * self.args.gamma * q_next
      target = torch.clamp(target, min=-5.0, max=5.0)

    return q_sa, target

  def update_rl(self):
    """Updates the Q-network and the OM Transformer."""
    if len(self.rl_replay) < self.args.min_replay:
        return None, None

    batch_list = self.rl_replay.sample(self.args.batch_size)

    # 1. Update Opponent Model
    om_batch = {
      "states": torch.from_numpy(np.array([b["state"] for b in batch_list], dtype=np.float32)).to(self.device),
      "history": {
        "state_features": torch.from_numpy(np.array([b["history"]["state_features"][0] for b in batch_list], dtype=np.float32)).to(self.device),
        "actions": torch.from_numpy(np.array([b["history"]["actions"][0] for b in batch_list], dtype=np.int64)).to(self.device),
        "mask": torch.from_numpy(np.array([b["history"]["mask"][0] for b in batch_list], dtype=bool)).to(self.device)
      },
      "true_goal_map": torch.from_numpy(np.array([b["true_goal_map"] for b in batch_list], dtype=np.float32)).to(self.device)
    }
    # cached_features=True bypasses the CNN for history (massive speedup)
    model_loss = self.model.train_step(om_batch, cached_features=True)

    # 2. Update Q-Network
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

    return loss_val, model_loss

  # ------------- SL Update Logic -------------

  def update_sl(self):
    """Updates the SL Average Strategy network."""
    if len(self.sl_replay) < self.args.min_replay:
      return None

    batch = self.sl_replay.sample(self.args.batch_size)
    s = torch.from_numpy(np.stack([b["state"] for b in batch])).float().to(self.device)
    a = torch.from_numpy(np.array([b["action"] for b in batch], dtype=np.int64)).to(self.device)

    logits = self.sl(s)
    loss = F.cross_entropy(logits, a)

    self.opt_sl.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(self.sl.parameters(), 5.0)
    self.opt_sl.step()

    return loss.item()

  # ------------- Data Generation (Rollout) -------------

  def run_fsp_episode(self, opponent_agent, eta: float, max_steps: Optional[int] = None) -> Dict[str, float]:
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
    ep_ret, opp_ret, ep_entropy = 0.0, 0.0, 0.0

    history_len = self.args.max_history_length
    rolling_feats = torch.zeros((1, history_len, self.args.d_model), device=self.device)
    rolling_actions = torch.zeros((1, history_len), dtype=torch.long, device=self.device)
    rolling_mask = torch.zeros((1, history_len), dtype=torch.bool, device=self.device)
    current_seq_len = 0

    episode_transitions = []
    H, W, _ = obs[0].shape

    for step in range(max_steps or 500):
      history_gpu = {
          "state_features": rolling_feats,
          "actions": rolling_actions,
          "mask": rolling_mask
      }

      # Compute both RL (Best Response) and SL (Average) actions
      rl_a, g_map, rl_entropy = self.select_rl_action(obs[0], history_gpu)
      sl_a, sl_entropy = self.select_sl_action(obs[0])

      # Mix policies based on eta (Fictitious Play)
      if random.random() < eta:
        a = rl_a
        step_entropy = rl_entropy
        is_rl = True
      else:
        a = sl_a
        step_entropy = sl_entropy
        is_rl = False

      # Opponent acts
      if hasattr(opponent_agent, 'is_frozen_as_sl') and opponent_agent.is_frozen_as_sl:
        a_opponent, _ = opponent_agent.select_sl_action(obs[1])
      else:
        a_opponent, _ = opponent_agent.select_action(obs[1])

      actions = {0: a, 1: a_opponent}
      ep_entropy += step_entropy

      next_obs, reward, done, info = self.env.step(actions)

      # Store in SL Buffer only if the action was a Best Response (NFSP Standard)
      if is_rl:
        self.sl_replay.push({"state": obs[0].copy(), "action": a})

      history_cpu = {
          "state_features": rolling_feats.cpu().numpy(),
          "actions": rolling_actions.cpu().numpy(),
          "mask": rolling_mask.cpu().numpy()
      }

      transition = {
          "state": obs[0].copy(),
          "action": a,
          "opp_action": a_opponent,
          "reward": float(reward[0]),
          "opp_reward": float(reward[1]),
          "next_state": next_obs[0].copy(),
          "done": bool(done),
          "rollout_goal_map": g_map.cpu().numpy(),
          "history": history_cpu
      }
      episode_transitions.append(transition)

      # Update rolling history
      state_tensor = torch.from_numpy(obs[0]).float().unsqueeze(0).to(self.device)
      with torch.no_grad():
        new_feat = self.model.inference_model.get_features(state_tensor)

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
      self.global_step += 1

      if done:
        break

    # Hindsight Relabeling for RL Buffer
    current_true_goal_pos = None
    next_map = np.zeros((H, W), dtype=np.float32)
    next_rollout_map = np.zeros((H, W), dtype=np.float32)

    if len(episode_transitions) > 0:
      final_t = episode_transitions[-1]
      if final_t["opp_reward"] == 0:
        opp_pos_arr = np.argwhere(final_t["state"][:, :, 3] == 1)
        if len(opp_pos_arr) > 0:
          current_true_goal_pos = tuple(opp_pos_arr[0])

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

      t["true_goal_map_next"] = next_map
      next_map = true_map.copy()
      del t["opp_reward"]

      t["rollout_goal_map_next"] = next_rollout_map
      next_rollout_map = t["rollout_goal_map"].copy()

    # Push to RL buffer
    for t in episode_transitions:
      self.rl_replay.push(t)

    return {
        "return": ep_ret,
        "steps": step + 1,
        "opp_return": opp_ret,
        "avg_entropy": ep_entropy / (step + 1)
    }

  def run_test_episode(self, opponent_agent, use_sl: bool = True, max_steps: Optional[int] = None) -> Dict[str, float]:
    """Evaluation rollout. Defaults to Average Strategy (SL) as per FSP standards."""
    obs = self.env.reset()
    opponent_agent.reset()
    done = False
    ep_ret, opp_ret, ep_entropy = 0.0, 0.0, 0.0

    history_len = self.args.max_history_length
    rolling_feats = torch.zeros((1, history_len, self.args.d_model), device=self.device)
    rolling_actions = torch.zeros((1, history_len), dtype=torch.long, device=self.device)
    rolling_mask = torch.zeros((1, history_len), dtype=torch.bool, device=self.device)
    current_seq_len = 0

    for step in range(max_steps or 500):
      history_gpu = {
          "state_features": rolling_feats,
          "actions": rolling_actions,
          "mask": rolling_mask
      }

      if use_sl:
        a, step_entropy = self.select_sl_action(obs[0], eval=True)
      else:
        a, _, step_entropy = self.select_rl_action(obs[0], history_gpu, eval=True)
          
      if hasattr(opponent_agent, 'is_frozen_as_sl') and opponent_agent.is_frozen_as_sl:
        a_opponent, _ = opponent_agent.select_sl_action(obs[1], eval=True)
      else:
        a_opponent, _ = opponent_agent.select_action(obs[1], eval=True)

      actions = {0: a, 1: a_opponent}
      next_obs, reward, done, info = self.env.step(actions)

      state_tensor = torch.from_numpy(obs[0]).float().unsqueeze(0).to(self.device)
      with torch.no_grad():
        new_feat = self.model.inference_model.get_features(state_tensor)
      
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
        "avg_entropy": ep_entropy / (step + 1)
    }