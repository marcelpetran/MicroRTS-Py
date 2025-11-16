from typing import Deque, Dict, List, Tuple, Optional
import random
from collections import deque

from sympy import true
from omg_args import OMGArgs

from simple_foraging_env import SimpleAgent, RandomAgent, SimpleForagingEnv
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
    # WARNING: might not work with complex action spaces
    self.action_dim = args.action_dim

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
    - subgoal_selector.select(vae, eval_policy, s_t, future_states, tau) -> (mu, logvar)  # uses Eq.(6)/(7)
    - train_step(batch_dict, eval_policy) -> float  # trains the CVAE part of the OpponentModel
    - prior_model: pre-trained TransformerVAE (VAE encoder returns (mu, logvar))
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
    # self.replay = ReplayBuffer(self.args.capacity)
    self.replay = PrioritizedReplayBuffer(self.args.capacity)

    # Schedules
    self.global_step = 0

  # ------------- epsilon schedules --------------

  def _eps(self) -> float:
    t = min(self.global_step, self.args.eps_decay_steps)
    return self.args.eps_end + (self.args.eps_start - self.args.eps_end) * (1 - t / self.args.eps_decay_steps)

  def _gmix_eps(self) -> float:
    t = min(self.global_step, self.args.gmix_eps_decay_steps)
    return self.args.gmix_eps_end + (self.args.gmix_eps_start - self.args.gmix_eps_end) * (1 - t / self.args.gmix_eps_decay_steps)

  def _tau(self) -> float:
    t = min(self.global_step, self.args.selector_tau_decay_steps)
    return self.args.selector_tau_end + (self.args.selector_tau_start - self.args.selector_tau_end) * (1 - t / self.args.selector_tau_decay_steps)

  def _beta(self) -> float:
    t = min(self.global_step, self.args.beta_decay_steps)
    return self.args.beta_end + (self.args.beta_start - self.args.beta_end) * (1 - t / self.args.beta_decay_steps)
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
      self.model.prior_model, self, x, futures, self._tau())
    return mu, logvar

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

    # This will store the max Q-value for each grid cell
    q_value_map = np.zeros((H, W))
    # This will store the best action (0:Up, 1:Down, 2:Left, 3:Right) for each cell
    policy_map = np.zeros((H, W))

    # Find the original position of our agent (agent 1, feature index 2)
    original_pos = self.env._get_agent_positions()[0]
    # Iterate over every possible cell in the grid
    for pos in self.env._get_freed_positions() + [original_pos]:
      r, c = pos
      
      self.env.set_agent_position(0, pos)
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
    self.env.set_agent_position(0, original_pos)
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

  def _choose_action(self, qvals: torch.Tensor, beta: float) -> int:
    gumbel_noise = -beta * torch.empty_like(qvals).exponential_().log()
    return int(torch.argmax(qvals + gumbel_noise))

  def select_action(self, s_t: np.ndarray, history: Dict[str, List[torch.Tensor]]) -> Tuple[int, torch.Tensor]:
    """
    (interaction phase) Infer g_hat and act eps-greedily on Q(s,g_hat,*)
    """
    ghat_mu, ghat_logvar = self._infer_ghat(s_t, history)  # (1, latent_dim)
    s = torch.from_numpy(s_t).float().unsqueeze(0).to(self.device)
    qvals = self.q(s, ghat_mu)
    a = self._choose_action(qvals, self._tau())
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
    with torch.no_grad():
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

    batch_list, is_weights, tree_indices = self.replay.sample(
      self.args.batch_size)

    is_weights = torch.tensor(
      is_weights, dtype=torch.float32, device=self.args.device)

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

    # --- 2. Update the Opponent Model ---
    # Construct a proper batch dictionary for the OpponentModel
    om_batch = {
        # States: (B, H, W, F)
        "states": torch.stack([torch.from_numpy(b["state"]).float() for b in batch_list], dim=0),
        "history": self._collate_history([b["history"] for b in batch_list]),
        "future_states": torch.stack([torch.from_numpy(np.stack(b["future_states"])) for b in batch_list], dim=0),
        "infer_mu": torch.stack([b["infer_mu"] for b in batch_list], dim=0),
        "infer_log_var": torch.stack([b["infer_log_var"] for b in batch_list], dim=0),
        "dones": torch.tensor([b["done"] for b in batch_list], dtype=torch.float32, device=self.device)
    }
    model_loss = self.model.train_step(
      om_batch, self)

    if self.global_step % self.args.target_update_every == 0:
      self.q_tgt.load_state_dict(self.q.state_dict())

    return loss.item(), model_loss

  def _collate_history(self, histories: List[Dict]) -> Dict[str, List[torch.Tensor]]:
    """
    Helper to batch histories of variable lengths by padding shorter ones.
    Creates a mask to indicate real vs padded tokens.
    Returns a dict with keys 'states', 'actions', and 'mask'.
    """
    if not histories:
      return {"states": [], "actions": []}

    true_lengths = [len(h.get("states", [])) for h in histories]
    # Find the maximum history length in this batch
    max_len = max(true_lengths) if true_lengths else 0

    if max_len == 0:
      return {"states": [], "actions": []}

    # (B, max_len) True for real tokens, False for padding.
    mask = torch.arange(max_len, device=self.device)[None, :] < torch.tensor(
      true_lengths, device=self.device)[:, None]

    padded_states_list = []
    padded_actions_list = []

    # Create null tensors for padding
    null_state = torch.zeros(*self.args.state_shape, device=self.device)
    null_action = torch.zeros((), dtype=torch.long, device=self.device)

    # Pad each history to max_len
    for h in histories:
      num_to_pad = max_len - len(h.get("states", []))
      states = [s.to(self.device) for s in h.get("states", [])]
      actions = [a.to(self.device) for a in h.get("actions", [])]
      if num_to_pad > 0:
        states.extend([null_state] * num_to_pad)
        actions.extend([null_action] * num_to_pad)

      padded_states_list.append(torch.stack(
        states, dim=0))   # List of (T, H, W, F)
      padded_actions_list.append(torch.stack(actions, dim=0))  # List of (T,)

    # list of (T,H,W,F) -> (B, T, H, W, F)
    final_padded_states = torch.stack(padded_states_list, dim=0)
    # list of (T,) -> (B, T)
    final_padded_actions = torch.stack(padded_actions_list, dim=0)

    return {
        "states": final_padded_states,
        "actions": final_padded_actions,
        "mask": mask
    }

  # ------------- rollout -------------

  def run_episode(self, max_steps: Optional[int] = None) -> Dict[str, float]:
    """
    Gathers a trajectory, stores future slices for subgoal selection,
    and trains the Q-network and OpponentModel.
    """
    # self.opponent_agent = SimpleAgent(1)
    # if random.random() < self._eps():
    #   self.opponent_agent = RandomAgent(1)
    # else:
    self.opponent_agent = SimpleAgent(1, True)

    obs = self.env.reset_random_spawn(0)
    done = False
    ep_ret = 0.0

    # History container
    history_len = self.args.max_history_length
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
      elif done and len(step_buffer) > 1:
        # If episode ends, fill the future window with remaining states
        while len(step_buffer) > 1:
          transition_to_store = step_buffer.popleft()
          future_states = [s["state"] for s in list(step_buffer)]
          # Fill the rest with copies of the terminal state
          for _ in range(self.args.horizon_H - len(future_states)):
            future_states.append(step_buffer[-1]["state"])
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
        print(f"Step {self.global_step}: Q_loss={Q_loss:.4f}, Model_loss={model_loss:.4f}, Tau={self._tau():.3f}, Gmix_eps={self._gmix_eps():.3f}")

      if done:
        break

    return {"return": ep_ret, "steps": step + 1}

  def run_test_episode(self, max_steps: Optional[int] = None, render: bool = False) -> Dict[str, float]:
    self.opponent_agent = SimpleAgent(1, True)
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

      a, ghat_mu, ghat_logvar = self.select_action(
        obs[0], current_history)
      a_opponent = self.opponent_agent.select_action(obs[1])
      actions = {0: a, 1: a_opponent}
      next_obs, reward, done, info = self.env.step(actions)

      history["states"].append(torch.from_numpy(obs[0]).float())
      history["actions"].append(torch.tensor(a, dtype=torch.long))

      ep_ret += reward[0]
      obs = next_obs
      if render and not done and (step + 1) % self.args.visualise_every_n_step == 0:
        print(f"\n--- Visualization at Step {step+1} ---")

        print("Generating Q-value heatmap...")
        self.heatmap_q_values(ghat_mu, f"./diagrams_{self.args.folder_id}/q_heatmap_step{self.global_step + step}.png")

        print("Generating subgoal visualizations...")
        with torch.no_grad():
          self.model.inference_model.eval()
          recon_logits, g_bar, _ = self.model.inference_model(
              torch.from_numpy(obs[0]).float().unsqueeze(0).to(self.device),
              current_history
          )
          # self.model.visualize_subgoal(ghat_mu.unsqueeze(0), f"./diagrams/subgoal_onehot_step{self.global_step + step}.png")
          # self.model.visualize_selected_subgoal(
          #   g_bar, obs[0], f"./diagrams_{self.args.folder_id}/selected_subgoal_step{self.global_step + step}.png")
          self.model.visualize_subgoal_logits(
            obs[0], recon_logits, f"./diagrams_{self.args.folder_id}/subgoal_logits_step{self.global_step + step}.png")

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
