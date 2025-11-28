from math import e
from typing import Dict, List, Tuple
import transformers as t
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from simple_foraging_env import RandomAgent, SimpleAgent
from q_agent import ReplayBuffer
from omg_args import OMGArgs


class SubGoalSelectorOracle:
  def __init__(self, args):
    self.args = args

  def select(self, vae, eval_policy, s_t_batch: torch.Tensor, future_states: torch.Tensor, tau: float):
    B, _, _, _ = s_t_batch.shape
    latent_dim = self.args.latent_dim
    subgoal = torch.zeros((B, latent_dim), device=s_t_batch.device)
    vae_log_var = torch.zeros((B, latent_dim), device=s_t_batch.device)
    return subgoal, vae_log_var


class OpponentModelOracle(nn.Module):
  def __init__(self, cvae: t.TransformerCVAE, vae: t.TransformerVAE, selector: SubGoalSelectorOracle, args: OMGArgs = OMGArgs()):
    super(OpponentModelOracle, self).__init__()
    self.inference_model = cvae
    self.prior_model = vae  # pre-trained VAE
    self.subgoal_selector = selector
    self.optimizer = torch.optim.Adam(cvae.parameters(), lr=args.cvae_lr)
    self.vae_optimizer = torch.optim.Adam(vae.parameters(), lr=args.vae_lr)
    self.replay = ReplayBuffer(args.capacity)
    self.device = args.device
    self.args = args
    self.projector = torch.randn(args.latent_dim, device=self.device, requires_grad=False)

    # Precompute feature weights for reconstruction loss
    splits = self.args.state_feature_splits
    weights = []
    for s in splits:
      weights.append(torch.full((s,), 1.0 / s))
    w = torch.cat(weights)
    w = w / len(splits)
    # Weights for each one-hot feature
    self.register_buffer('feature_split_weights', w.to(self.device))
    # Weights for each feature type
    self.register_buffer('feature_weights', torch.tensor(
      [1.0, 20.0, 20.0, 20.0], device=self.device))

  def forward(self, x: torch.Tensor, history: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, H, W, _ = x.shape
    g = torch.zeros((B, self.args.latent_dim), device=self.device)
    # opp start is at row 3, col 6
    opp_start = torch.tensor([3, 6], device=self.device).float()
    for b in range(B):
      food_indices = (x[b, :, :, 1] == 1).nonzero(as_tuple=False)
      opp_idx = (x[b, :, :, 3] == 1).nonzero(as_tuple=False).float()

      is_top_food = 0.0
      # if we have food and opponent is not at start position
      if len(food_indices) > 1 and not torch.all(opp_idx[0] == opp_start):
        # find closest food to opponent
        dists = torch.norm(food_indices - opp_idx[0], dim=1)
        target_idx = torch.argmin(dists)
        target_row = food_indices[target_idx][0]
        # target_row is food that is closes to opponent

        # If target row is in upper half, it's Top (1.0), else Bottom (-1.0)
        is_top_food = 1.0 if target_row < H / 2 else -1.0
      elif len(food_indices) == 1:
        target_row = food_indices[0][0]
        is_top_food = 1.0 if target_row < H / 2 else -1.0

      g[b] = is_top_food * self.projector

    return g, torch.zeros_like(g)

  def eval(self):
    """
    Set inference model to evaluation mode
    """
    self.inference_model.eval()
    self.prior_model.eval()
    return

  def reconstruct_state(self, reconstructed_state_logits, state_feature_splits=None):
    """
    Convert the reconstructed logit tensor back to one-hot encoded state.
    """
    if state_feature_splits is None:
      state_feature_splits = self.args.state_feature_splits

    if reconstructed_state_logits.dim() == 3:
      # probably got (H, W, F), add batch dim
      reconstructed_state_logits = reconstructed_state_logits.unsqueeze(0)

    reconstructed_state = torch.zeros_like(
      reconstructed_state_logits, device=reconstructed_state_logits.device)
    start_idx = 0

    B, H, W, _ = reconstructed_state.shape
    for size in state_feature_splits:
      end_idx = start_idx + size
      # Apply softmax to the logits to get probabilities
      probs = F.softmax(
        reconstructed_state_logits[:, :, :, start_idx:end_idx], dim=-1)
      # Sample indices from the probabilities
      indices = torch.multinomial(
        probs.view(-1, size), num_samples=1).view(B, H, W) + start_idx
      # or alternatively, take the argmax
      # indices = torch.argmax(probs, dim=-1) + start_idx
      # Set the corresponding one-hot feature to 1
      reconstructed_state.scatter_(3, indices.unsqueeze(-1), 1.0)
      start_idx = end_idx

    return reconstructed_state

  def visualize_subgoal_logits(self, obs: np.ndarray, reconstructed_logits: torch.Tensor, filename: str = None):
    """
    Visualizes the softmax probabilities of the reconstructed subgoal logits.
    """
    assert reconstructed_logits.dim() == 4, "Expected logits to be 4D (B, H, W, F)"
    logits = reconstructed_logits[0].detach().cpu()

    # Apply softmax to get probabilities
    probs = torch.zeros_like(logits)
    start_idx = 0
    for size in self.args.state_feature_splits:
      end_idx = start_idx + size
      probs[:, :, start_idx:end_idx] = F.softmax(
        logits[:, :, start_idx:end_idx], dim=-1)
      start_idx = end_idx

    probs = probs.numpy()  # (H, W, F)
    labels = {0: 'Empty', 1: 'Food',
              2: 'Agent 1 (Self)', 3: 'Agent 2 (Opponent)'}

    H, W, F_dim = probs.shape

    fig = plt.figure(figsize=(F_dim * 4 + 4, 8))
    gs = fig.add_gridspec(2, F_dim)

    # Plot the actual current state
    ax_obs = fig.add_subplot(gs[0, 1])

    obs_labels = np.argmax(obs, axis=-1)

    cmap_obs = plt.get_cmap('viridis', F_dim)
    im = ax_obs.imshow(obs_labels, cmap=cmap_obs, vmin=0, vmax=F_dim - 1)
    fig.colorbar(im, ax=ax_obs, ticks=np.arange(len(labels)))
    cbar = im.colorbar
    cbar.ax.set_yticklabels([labels[i] for i in range(len(labels))])

    ax_obs.set_title("Current obs (s_t)")
    ax_obs.set_xticks(np.arange(W))
    ax_obs.set_yticks(np.arange(H))

    for i in range(F_dim):
      ax = fig.add_subplot(gs[1, i])
      im = ax.imshow(probs[:, :, i], cmap='hot', vmin=0, vmax=1)
      ax.set_title(f"P({labels[i]})")
      fig.colorbar(im, ax=ax)
      ax.set_xticks(np.arange(W))
      ax.set_yticks(np.arange(H))

    fig.suptitle("Current obs and Inferred Subgoal Probabilities", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if filename is None:
      plt.show()
    else:
      plt.savefig(filename)
    plt.close()

  def get_weight_mask(self, x):
    weight_mask = torch.full_like(
      x, self.feature_weights[0], device=self.device)
    food_mask = (x[..., 1] == 1)
    agent1_mask = (x[..., 2] == 1)
    agent2_mask = (x[..., 3] == 1)

    weight_mask[..., 1][food_mask] = self.feature_weights[1]
    weight_mask[..., 2][agent1_mask] = self.feature_weights[2]
    weight_mask[..., 3][agent2_mask] = self.feature_weights[3]

    return weight_mask

  def vae_loss(self, reconstructed_x, x, mu, logvar):
    """
    Loss function for VAE combining reconstruction loss and KL divergence.
    Args:
        reconstructed_x (Tensor): Reconstructed state of shape (B, H, W, F)
        x (Tensor): Original input state of shape (B, H, W, F)
        mu (Tensor): Mean of the latent distribution (B, latent_dim)
        logvar (Tensor): Log-variance of the latent distribution (B, latent_dim)
    Returns:
        Tensor: Computed VAE loss.
    """
    recon_loss = 0
    weight_mask = self.get_weight_mask(x)

    bce = F.binary_cross_entropy_with_logits(
      reconstructed_x, x, weight=weight_mask, reduction='none')  # (B, H, W, F)
    recon_loss = bce.mean(dim=(1, 2, 3))

    # KL Divergence Loss
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss.mean() + self.args.vae_beta * kld_loss

  def train_vae(self,
                env,
                num_epochs=10000,
                save_every_n_epochs=1000,
                logg=100
                ):
    def collect_single_episode():
      if 1 / 3 < np.random.random():
        agent1 = RandomAgent(agent_id=0)
        agent2 = RandomAgent(agent_id=1)
      else:
        agent1 = SimpleAgent(agent_id=0)
        agent2 = SimpleAgent(agent_id=1)

      obs = env.reset()
      done = False
      step = 0
      ep_ret = 0.0

      while not done and (self.args.max_steps is None or step < self.args.max_steps):
        a = agent1.select_action(obs[0])
        a_opponent = agent2.select_action(obs[1])
        actions = {0: a, 1: a_opponent}
        next_obs, reward, done, info = env.step(actions)

        # store transition
        self.replay.push(
            {
                "state": obs[0].copy(),
                "action": a,
                "reward": float(reward[0]),
                "next_state": next_obs[0].copy(),
                "done": bool(done),
            }
        )

        ep_ret += reward[0]
        obs = next_obs
        step += 1
      return

    loss_collector = []
    epoch_collector = []
    avg_loss = 0.0
    for i in range(num_epochs):
      # Collect a new episode
      collect_single_episode()

      # fill the buffer so we can at least sample
      while self.replay.__len__() < self.args.batch_size:
        collect_single_episode()

      # Sample a batch from the replay buffer
      batch = self.replay.sample(self.args.batch_size)
      state_batch = torch.from_numpy(
          np.array([b["state"] for b in batch])
      ).float()  # (B, H, W, F)
      state_batch = state_batch.to(self.args.device)
      self.prior_model.train()
      reconstructed_state, mu, logvar = self.prior_model(state_batch)
      loss = self.vae_loss(
          reconstructed_state,
          state_batch,
          mu,
          logvar
      )

      self.vae_optimizer.zero_grad()
      loss.backward()
      self.vae_optimizer.step()

      avg_loss += loss.item()

      if (i + 1) % logg == 0:
        loss_collector.append(loss.item())
        epoch_collector.append(i + 1)
        print(
            f"Epoch {i + 1}/{num_epochs}, Loss: {loss.item()}, Avg Loss: {avg_loss / logg}"
        )
        avg_loss = 0.0

      if (i + 1) % save_every_n_epochs == 0:
        torch.save(self.prior_model.state_dict(),
                   f"./models_{self.args.folder_id}/vae_epoch_{i + 1}.pth")
        print(f"Model saved at epoch {i + 1}")

    loss_collector = np.array(loss_collector)
    epoch_collector = np.array(epoch_collector)

    # plt.plot(epoch_collector, loss_collector)
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.title('Training Loss over Time')
    # plt.show()

  def loss_function(
          self, reconstructed_x, x, cvae_mu, cvae_log_var,
          vae_mu, vae_log_var, infer_mu, infer_log_var, dones, eta, beta):
    return 0.0

  def train_step(self, batch, eval_policy):
    # not necessary
    return 0.0

# Helper function to plot the foraging grid


def _plot_foraging_grid(grid: np.ndarray, filename: str):
  """
  Creates a plot of a single (H, W, F) foraging state.
  """

  grid_labels = np.argmax(grid, axis=-1)

  cmap = plt.get_cmap('viridis', 4)
  labels = {0: 'Empty', 1: 'Food',
            2: 'Agent 1 (Self)', 3: 'Agent 2 (Opponent)'}

  fig, ax = plt.subplots()
  mat = ax.matshow(grid_labels, cmap=cmap)

  # Create a color bar with labels
  cbar = plt.colorbar(mat, ticks=np.arange(len(labels)))
  cbar.ax.set_yticklabels([labels[i] for i in range(len(labels))])

  plt.title("Reconstructed Subgoal State")
  plt.savefig(filename)
  plt.close()
