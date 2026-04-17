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
from networks import QNet, SLnet

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
    self.args.action_dim = len(self.env.action_space) if hasattr(
      self.env.action_space, "__len__") else int(self.env.action_space)

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
  def select_rl_action(self, s_t: np.ndarray, history: Dict[str, torch.Tensor], eval=False) -> Tuple[int, torch.Tensor, float]:
    """Calculates the Best Response action using Q-learning and OM."""
    self.q.eval()
    x = torch.from_numpy(s_t).float().unsqueeze(0).to(self.device)

    g_logits = self.model(x, history, cached_features=True)
    g_map = F.softmax(g_logits.view(
      g_logits.shape[0], -1), dim=-1).view_as(g_logits)

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
          np.array([b["history"]["state_features"][0] for b in batch], dtype=np.float32)
      ).to(self.device)
      hist_acts = torch.from_numpy(
          np.array([b["history"]["actions"][0] for b in batch], dtype=np.int64)
      ).to(self.device)
      hist_mask = torch.from_numpy(
          np.array([b["history"]["mask"][0] for b in batch])
      ).to(self.device)
      hist = {"state_features": hist_feats, "actions": hist_acts, "mask": hist_mask}

      g_logits = self.model.inference_model(s, hist, cached_features=True)
      g_map = F.softmax(g_logits.view(len(batch), -1), dim=-1).view_as(g_logits)

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
      hist_next = {"state_features": hist_feats_next, "actions": hist_acts_next, "mask": hist_mask_next}

      g_logits_next = self.model.inference_model(sp, hist_next, cached_features=True)
      g_map_next = F.softmax(g_logits_next.view(len(batch), -1), dim=-1).view_as(g_logits_next)

      self.model.inference_model.train()

    q_sa = self.q(s, g_map).gather(1, a.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
      q_val = self.q(sp, g_map_next)
      noise = torch.rand_like(q_val) * 1e-6
      best_actions = (q_val + noise).argmax(dim=1, keepdim=True)
      q_next = self.q_tgt(sp, g_map_next).gather(1, best_actions).squeeze(1)
      target = r + (1.0 - done) * self.args.gamma * q_next
      target = torch.clamp(target, min=-15.0, max=15.0)

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

    model_loss = self.model.train_step(om_batch, cached_features=True)

    return loss_val, model_loss

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

  def _apply_hindsight_relabeling(self, episode_transitions: list, H: int, W: int):
    """
    Applies Hindsight Experience Replay (HER) labeling to a trajectory.
    Modifies the transitions in-place to include 'true_goal_map'.
    """
    current_true_goal_pos = None
    next_map = np.zeros((H, W), dtype=np.float32)

    # 1. Hindsight labeling for truncated episodes
    if len(episode_transitions) > 0:
      final_t = episode_transitions[-1]
      
      if final_t["opp_reward"] == 0:
        opp_pos_arr = np.argwhere(final_t["state"][:, :, 3] == 1)
        if len(opp_pos_arr) > 0:
          current_true_goal_pos = tuple(opp_pos_arr[0])

    # 2. Walk backward through the episode to label goals
    for t in reversed(episode_transitions):

      # Did the opponent get a reward this step? (New true goal achieved)
      if t["opp_reward"] > 0:
        opp_pos_indices = np.argwhere(t["next_state"][:, :, 3] == 1)
        if len(opp_pos_indices) > 0:
          current_true_goal_pos = tuple(opp_pos_indices[0])

      # Assign the goal to this step
      true_map = np.zeros((H, W), dtype=np.float32)
      if current_true_goal_pos is not None:
        true_map[current_true_goal_pos[0], current_true_goal_pos[1]] = 1.0
      
      t["true_goal_map"] = true_map
      t["true_goal_map_next"] = next_map
      next_map = true_map.copy()
      
      del t["opp_reward"]

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

    history_len = self.args.max_history_length
    rolling_feats = torch.zeros(
      (1, history_len, self.args.d_model), device=self.device)
    rolling_actions = torch.zeros(
      (1, history_len), dtype=torch.long, device=self.device)
    rolling_mask = torch.zeros(
      (1, history_len), dtype=torch.bool, device=self.device)
    opp_rolling_feats = torch.zeros(
      (1, history_len, self.args.d_model), device=self.device)
    opp_rolling_actions = torch.zeros(
      (1, history_len), dtype=torch.long, device=self.device)
    opp_rolling_mask = torch.zeros(
      (1, history_len), dtype=torch.bool, device=self.device)
    current_seq_len = 0

    episode_transitions = []
    H, W, _ = obs[0].shape

    for step in range(max_steps):
      history_gpu = {
          "state_features": rolling_feats,
          "actions": rolling_actions,
          "mask": rolling_mask
      }
      opp_history_gpu = {
          "state_features": opp_rolling_feats,
          "actions": opp_rolling_actions,
          "mask": opp_rolling_mask
      }

      # Compute both RL (Best Response) and SL (Average) actions
      rl_a, g_map, rl_entropy = self.select_rl_action(obs[0], history_gpu)
      sl_a, sl_entropy = self.select_sl_action(obs[0])
      rl_ep_entropy += rl_entropy
      sl_ep_entropy += sl_entropy

      # Mix policies based on eta (Fictitious Play)
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
        opp_rl_a, _, _ = opponent_agent.select_rl_action(
          obs[1], opp_history_gpu)
        opp_sl_a, _ = opponent_agent.select_sl_action(obs[1])
        if random.random() < eta:
          a_opponent = opp_rl_a
          opp_is_rl = True
        else:
          a_opponent = opp_sl_a
          opp_is_rl = False

      actions = {0: a, 1: a_opponent}

      next_obs, reward, done, info = self.env.step(actions)

      # Store in SL Buffer only if the action was a Best Response (NFSP Standard)
      if is_rl:
        self.sl_replay.push({"state": obs[0].copy(), "action": a})
      if opp_is_rl and opponent_agent is self:
        self.sl_replay.push({"state": obs[1].copy(), "action": a_opponent})

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
          "history": history_cpu
      }
      episode_transitions.append(transition)

      # Update rolling history
      state_tensor = torch.from_numpy(
        obs[0]).float().unsqueeze(0).to(self.device)
      opp_state_tensor = torch.from_numpy(
        obs[1]).float().unsqueeze(0).to(self.device)
      with torch.no_grad():
        new_feat = self.model.inference_model.get_features(state_tensor)
        opp_new_feat = self.model.inference_model.get_features(
          opp_state_tensor)
      transition["next_state_feature"] = new_feat.squeeze(0).cpu().numpy()

      rolling_feats = torch.roll(rolling_feats, shifts=-1, dims=1)
      rolling_actions = torch.roll(rolling_actions, shifts=-1, dims=1)
      rolling_mask = torch.roll(rolling_mask, shifts=-1, dims=1)

      opp_rolling_feats = torch.roll(opp_rolling_feats, shifts=-1, dims=1)
      opp_rolling_actions = torch.roll(
        opp_rolling_actions, shifts=-1, dims=1)
      opp_rolling_mask = torch.roll(opp_rolling_mask, shifts=-1, dims=1)

      rolling_feats[:, -1, :] = new_feat
      rolling_actions[:, -1] = a_opponent

      opp_rolling_feats[:, -1, :] = opp_new_feat
      opp_rolling_actions[:, -1] = a

      if current_seq_len < history_len:
        current_seq_len += 1
        rolling_mask[:, -current_seq_len:] = True
        opp_rolling_mask[:, -current_seq_len:] = True

      ep_ret += reward[0]
      opp_ret += reward[1]
      obs = next_obs
      self.global_step += 1

      if done:
        break

    self._apply_hindsight_relabeling(episode_transitions, H, W)

    # Push to RL buffer
    for t in episode_transitions:
      self.rl_replay.push(t)

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
    ep_ret, opp_ret, rl_ep_entropy, sl_ep_entropy = 0.0, 0.0, 0.0, 0.0

    history_len = self.args.max_history_length
    rolling_feats = torch.zeros(
      (1, history_len, self.args.d_model), device=self.device)
    rolling_actions = torch.zeros(
      (1, history_len), dtype=torch.long, device=self.device)
    rolling_mask = torch.zeros(
      (1, history_len), dtype=torch.bool, device=self.device)
    opp_rolling_feats = torch.zeros(
      (1, history_len, self.args.d_model), device=self.device)
    opp_rolling_actions = torch.zeros(
      (1, history_len), dtype=torch.long, device=self.device)
    opp_rolling_mask = torch.zeros(
      (1, history_len), dtype=torch.bool, device=self.device)
    current_seq_len = 0

    for step in range(max_steps):
      history_gpu = {
          "state_features": rolling_feats,
          "actions": rolling_actions,
          "mask": rolling_mask
      }
      opp_history_gpu = {
          "state_features": opp_rolling_feats,
          "actions": opp_rolling_actions,
          "mask": opp_rolling_mask
      }
      sl_a, sl_entropy = self.select_sl_action(obs[0], eval=True)
      rl_a, _, rl_entropy = self.select_rl_action(
          obs[0], history_gpu, eval=True)
      if use_sl:
        a = sl_a
      else:
        a = rl_a

      if hasattr(opponent_agent, 'is_frozen_as_sl') and opponent_agent.is_frozen_as_sl:
        a_opponent, _ = opponent_agent.select_sl_action(obs[1], eval=True)
      else:
        if isinstance(opponent_agent, FSPAgentOM):
          a_opponent, _ = opponent_agent.select_action(
            obs[1], opp_history_gpu, eval=True)
        else:
          a_opponent, _ = opponent_agent.select_action(obs[1], eval=True)

      actions = {0: a, 1: a_opponent}
      next_obs, reward, done, info = self.env.step(actions)

      state_tensor = torch.from_numpy(
        obs[0]).float().unsqueeze(0).to(self.device)
      opp_state_tensor = torch.from_numpy(
        obs[1]).float().unsqueeze(0).to(self.device)
      with torch.no_grad():
        new_feat = self.model.inference_model.get_features(state_tensor)
        opp_new_feat = self.model.inference_model.get_features(
          opp_state_tensor)

      rolling_feats = torch.roll(rolling_feats, shifts=-1, dims=1)
      rolling_actions = torch.roll(rolling_actions, shifts=-1, dims=1)
      rolling_mask = torch.roll(rolling_mask, shifts=-1, dims=1)

      opp_rolling_feats = torch.roll(opp_rolling_feats, shifts=-1, dims=1)
      opp_rolling_actions = torch.roll(opp_rolling_actions, shifts=-1, dims=1)
      opp_rolling_mask = torch.roll(opp_rolling_mask, shifts=-1, dims=1)

      rolling_feats[:, -1, :] = new_feat
      rolling_actions[:, -1] = a_opponent

      opp_rolling_feats[:, -1, :] = opp_new_feat
      opp_rolling_actions[:, -1] = a

      if current_seq_len < history_len:
        current_seq_len += 1
        rolling_mask[:, -current_seq_len:] = True
        opp_rolling_mask[:, -current_seq_len:] = True

      ep_ret += reward[0]
      opp_ret += reward[1]
      obs = next_obs
      rl_ep_entropy += rl_entropy
      sl_ep_entropy += sl_entropy

      if done:
        break

    return {
        "return": ep_ret,
        "steps": step + 1,
        "opp_return": opp_ret,
        "avg_rl_entropy": rl_ep_entropy / (step + 1) if not use_sl else 0.0,
        "avg_sl_entropy": sl_ep_entropy / (step + 1) if use_sl else 0.0
    }
