from typing import Deque, Dict, List, Tuple, Optional
import random
from collections import deque
from omg_args import OMGArgs

from simple_foraging_env import SimpleAgent, RandomAgent, SimpleForagingEnv

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
    self.action_dim = args.action_dim # WARNING: might not work with complex action spaces

    hidden = args.qnet_hidden
    self.net = nn.Sequential(
        nn.Linear(self.state_dim + self.latent_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, self.action_dim),
    )

  def forward(self, batch: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    B, H, W, F_dim = batch.shape  # (B, H, W, F) float
    s = batch.view(B, H * W * F_dim)
    x = torch.cat([s, g], dim=-1)
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


class QLearningAgent:
  """
  Q(s, g, a) OMG agent that:
    • infers g_hat from history via OpponentModel.inference_model (CVAE),
    • gets g_bar from OpponentModel.subgoal_selector over future H states (VAE + value heuristic),
    • mixes them with Eq.(8), then learns with Eq.(4).

  Expected OpponentModel API:
    - inference_model(...): forward(x=current_state[None], history=...) -> (recon, mu, logvar)
    - subgoal_selector.select(vae, eval_policy, s_t, future_states) -> (mu, logvar)  # uses Eq.(6)/(7)
    - train_step(batch_dict, eval_policy) -> float  # trains the CVAE part of the OpponentModel
    - prior_model: pre-trained TransformerVAE (VAE encoder returns (mu, logvar))
  """

  def __init__(self, env, opponent_model, device="cpu", args: OMGArgs = OMGArgs()):
    self.env = env
    self.model = opponent_model
    self.args = args
    self.device = torch.device(device)
    self.opponent_agent = SimpleAgent(1)

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

  def _gmix_eps(self) -> float:
    t = min(self.global_step, self.args.gmix_eps_decay_steps)
    return self.args.gmix_eps_end + (self.args.gmix_eps_start - self.args.gmix_eps_end) * (1 - t / self.args.gmix_eps_decay_steps)

  # ------------- policy API for selector -------------

  @torch.no_grad()
  def value(self, s_t: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """
    s_t: (1, H, W, F), g: (1, latent_dim) -> Q(1, A)
    used by subgoal selector to compute V(s,g) = mean_a Q(s,g,a)
    """
    self.q.eval()
    return self.q(s_t, g)  # (1, A)

  # ------------- subgoal inference utilities -------------

  @torch.no_grad()
  def _infer_ghat(self, state_hwf: np.ndarray, history: Dict[str, List[torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (mu, logvar) for g_hat
    """
    x = torch.from_numpy(state_hwf).float().unsqueeze(
      0).to(self.device)  # (1,H,W,F)
    _, mu, logvar = self.model.inference_model(x, history)
    return mu, logvar

  @torch.no_grad()
  def _select_gbar(self, s_t: np.ndarray, future_states: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Uses SubGoalSelector over H future states. Returns (mu, logvar) for g_bar
    """
    x = torch.from_numpy(s_t).float().unsqueeze(
      0).to(self.device)    # (1,H,W,F)
    futures = torch.from_numpy(future_states).unsqueeze(
      0).float().to(self.device)  # (1,K,H,W,F)
    mu, logvar = self.model.subgoal_selector.select(
      self.model.prior_model, self, x, futures)
    return mu, logvar

  # ------------- acting -------------

  def _choose_action(self, qvals: torch.Tensor, eps: float) -> int:
    if random.random() < eps:
      # TODO: for more complex action spaces, adapt this
      return random.randrange(self.args.action_dim)
    # else choose greedy action with ties broken randomly
    qvals = qvals.squeeze(0)  # (A,)
    max_q = torch.max(qvals).item()
    max_actions = (qvals == max_q).nonzero(as_tuple=False).view(-1)
    if len(max_actions) > 1:
      return int(max_actions[torch.randint(len(max_actions), (1,))].item())
    return int(torch.argmax(qvals, dim=-1).item())

  def select_action(self, s_t: np.ndarray, history: Dict[str, List[torch.Tensor]]) -> Tuple[int, torch.Tensor]:
    """
    (interaction phase) Infer g_hat and act eps-greedily on Q(s,g_hat,*)
    """
    ghat_mu, ghat_logvar = self._infer_ghat(s_t, history)  # (1, latent_dim)
    s = torch.from_numpy(s_t).float().unsqueeze(0).to(self.device)
    qvals = self.q(s, ghat_mu)
    a = self._choose_action(qvals, self._eps())
    return a, ghat_mu.squeeze(0), ghat_logvar.squeeze(0)

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
                     dtype=torch.float32, device=self.device)               # (B,)
    done = torch.tensor([b["done"] for b in batch],
                        dtype=torch.float32, device=self.device)              # (B,)

    # Prepare g_hat (from stored inference) and g_bar (from selector over future window)
    ghat_mu = torch.stack([b["infer_mu"] for b in batch], dim=0).to(
      self.device)                           # (B,g)
    gbar_mu = []
    for b in batch:
      # Shape (K,H,W,F) for next few steps (collected during rollout)
      futures = np.stack(b["future_states"], axis=0) if b["future_states"] else np.zeros(
        (1, H, W, F_dim), dtype=np.float32)
      mu, _ = self._select_gbar(b["state"], futures)
      gbar_mu.append(mu.squeeze(0))
    gbar_mu = torch.stack(gbar_mu, dim=0)                         # (B,g)

    # Eq.(8): gt = g_hay if eta > eps_gmix else g_bar
    eps_gmix = self._gmix_eps()
    eta = torch.rand(B, device=self.device)
    use_ghat = (eta > eps_gmix).float().unsqueeze(-1)
    g_mix = use_ghat * ghat_mu + (1 - use_ghat) * gbar_mu         # (B,g)

    # Q(s,g,a) and target r + gamma * max_{a'} Q(s',g,a')  (same g)
    q_sa = self.q(s, g_mix).gather(1, a.view(-1, 1)).squeeze(1)
    with torch.no_grad():
      q_next = self.q_tgt(sp, g_mix).max(dim=1).values
      target = r + (1.0 - done) * self.args.gamma * q_next
    return q_sa, target

  def update(self):
    if len(self.replay) < self.args.min_replay:
      return (None, None)  # not enough data yet

    if self.global_step % self.args.train_every != 0:
      return (None, None)  # only train every few steps

    batch_list = self.replay.sample(self.args.batch_size)

    # --- 1. Update the Q-Network ---
    q_sa, target = self._compute_targets(batch_list)
    loss = F.mse_loss(q_sa, target)
    self.opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
    self.opt.step()

    # --- 2. Update the Opponent Model ---
    # Construct a proper batch dictionary for the OpponentModel
    om_batch = {
        # States: (B, H, W, F)
        "states": torch.stack([torch.from_numpy(b["state"]).float() for b in batch_list], dim=0),
        "history": self._collate_history([b["history"] for b in batch_list]),
        "future_states": torch.stack([torch.from_numpy(np.stack(b["future_states"])) for b in batch_list], dim=0),
        "infer_mu": torch.stack([b["infer_mu"] for b in batch_list], dim=0),
        "infer_log_var": torch.stack([b["infer_log_var"] for b in batch_list], dim=0),
    }
    model_loss = self.model.train_step(om_batch, self)

    if self.global_step % self.args.target_update_every == 0:
      self.q_tgt.load_state_dict(self.q.state_dict())

    return loss.item(), model_loss

  def _collate_history(self, histories: List[Dict]) -> Dict[str, List[torch.Tensor]]:
    """
    Helper to batch histories of variable lengths by padding shorter ones.
    """
    if not histories:
      return {"states": [], "actions": []}

    # Find the maximum history length in this batch
    max_len = 0
    for h in histories:
      if "states" in h and h["states"]:
        max_len = max(max_len, len(h["states"]))

    if max_len == 0:
      return {"states": [], "actions": []}

    # Create null tensors for padding
    null_state = torch.zeros(*self.args.state_shape, device=self.device)
    null_action = torch.zeros((), dtype=torch.long, device=self.device)

    # Pad each history to max_len
    for h in histories:
      if "states" not in h:  # Handle empty dicts
        h["states"], h["actions"] = [], []

      num_to_pad = max_len - len(h["states"])
      if num_to_pad > 0:
        h["states"].extend([null_state] * num_to_pad)
        h["actions"].extend([null_action] * num_to_pad)

    # Now that all histories are the same length, we can stack them
    collated = {"states": [], "actions": []}
    for i in range(max_len):
      collated["states"].append(torch.stack(
        [h["states"][i] for h in histories]))
      collated["actions"].append(torch.stack(
        [h["actions"][i] for h in histories]))

    return collated

  # ------------- rollout -------------

  def run_episode(self, max_steps: Optional[int] = None) -> Dict[str, float]:
    """
    Gathers a trajectory, stores future slices for subgoal selection,
    and trains the Q-network and OpponentModel.
    """
    self.opponent_agent.reset()
    obs = self.env.reset()
    done = False
    ep_ret = 0.0

    # Minimal "history" container; extend to what your CVAE expects.
    history_len = 5  # Example history length
    history = {
        "states": deque(maxlen=history_len),
        "actions": deque(maxlen=history_len)
    }

    step_buffer = deque(maxlen=self.args.horizon_H + 1)

    for step in range(max_steps or 500):
      # Convert deque to list for the model
      current_history = {k: list(v) for k, v in history.items()}

      a, ghat_mu, ghat_logvar = self.select_action(obs[0], current_history)
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
          "infer_mu": ghat_mu.detach().cpu(),
          "infer_log_var": ghat_logvar.detach().cpu(),
          "history": {k: [t.clone() for t in v] for k, v in current_history.items()}
      }
      step_buffer.append(step_info)

      # Once the buffer is full, the oldest step has its full future window
      if len(step_buffer) == self.args.horizon_H + 1:
        transition_to_store = step_buffer[0]
        future_states = [s["state"] for s in list(step_buffer)[1:]]
        transition_to_store["future_states"] = future_states
        self.replay.push(transition_to_store)

      # Update history for the next step
      history["states"].append(torch.from_numpy(obs[0]).float())
      history["actions"].append(torch.tensor(a, dtype=torch.long))

      ep_ret += reward[0]
      obs = next_obs

      self.global_step += 1
      Q_loss, model_loss = self.update()

      if Q_loss is not None and model_loss is not None and self.global_step % 100 == 0:
        print(f"Step {self.global_step}: Q_loss={Q_loss:.4f}, Model_loss={model_loss:.4f}, Eps={self._eps():.3f}, Gmix_eps={self._gmix_eps():.3f}")

      if done:
        break

    return {"return": ep_ret, "steps": step + 1}
