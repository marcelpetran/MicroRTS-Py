from typing import Dict, Tuple
import transformers as t
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from simple_foraging_env import RandomAgent, SimpleAgent
from q_agent import ReplayBuffer
from omg_args import OMGArgs


class SubGoalSelector:
  def __init__(self, args):
    self.args = args

  def gumbel_sample(self, logits, beta):
    """
    Samples from a categorical distribution using the Gumbel-max/min trick.
    Args:
        logits (Tensor): Logits of shape (K,) representing the unnormalized log probabilities.
    Returns:
        int: Index of the sampled category.
    """
    # gumbel_noise = np.random.gumbel(0, beta, logits.shape)
    # inverse CDF method X = mu - beta * log(-log(U)) -> mu = 0, U ~ Uniform(0,1) -> X = - beta * log(U)
    gumbel_noise = -beta * torch.empty_like(logits).exponential_().log()

  def select(self, vae, eval_policy, s_t_batch: torch.Tensor, future_states: torch.Tensor, tau: float):
    """
    Selects a subgoal by encoding future states to latent space s_g = encode(s_t+K), 
    selecting suitable future state by policy argmin Q-value(s_t, s_g), 
    and encoding choosen state to latent space
    Args:
        vae (VAE): Pre-trained VAE model
        policy (Policy): Policy model with Q-value function
        s_t_batch (Tensor): Current state of shape (B, H, W, F)
        future_states (Tensor): Future states of shape (B, K, H, W, F) - K is the horizon
    Returns:
        Tensor: latent subgoal of shape (latent_dim)
    """
    vae.eval()
    with torch.no_grad():
      assert s_t_batch.dim(
      ) == 4, f"s_t_ba should be 4D (B, H, W, F), got {s_t_batch.shape}"
      assert future_states.dim(
      ) == 5, f"future_states should be 5D (B, K, H, W, F), got {future_states.shape}"

      B, K, H, W, F_dim = future_states.shape

      # (B, K, H, W, F) -> (B*K, H, W, F)
      all_future_states = future_states.view(B * K, H, W, F_dim)

      # (B, H, W, F) -> (B, 1, H, W, F) -> (B*K, H, W, F)
      s_t_expanded = s_t_batch.unsqueeze(1).repeat(
        1, K, 1, 1, 1).view(B * K, H, W, F_dim)

      mu, logvar = vae.encode(all_future_states)  # (B*K, latent_dim)

      # s_t_expanded: (B*K, H, W, F), mu: (B*K, latent_dim) -> values_flat: (B*K, A)
      values_flat = eval_policy.value(s_t_expanded, mu)

      # reshape back to (B, K, ...)
      mu = mu.view(B, K, self.args.latent_dim)
      logvar = logvar.view(B, K, self.args.latent_dim)
      values = values_flat.view(B, K, self.args.action_dim)

      # Calculate V(s, g) = Expected Q-value over actions (B, K)
      probs = F.softmax(values / tau, dim=-1)  # (B, K, A)
      expected_values = (probs * values).sum(dim=-1)  # (B, K)

      # best_idx is (B,) containing the index (0 to K-1) for each item
      gumbel_noise = -tau * \
          torch.empty_like(expected_values).exponential_().log()

      best_idx = torch.argmin(expected_values - gumbel_noise, dim=1)

      # Gather the Selected Latent Vector
      # We need to gather the selected indices from the K dimension
      # best_idx is (B,). We need to transform it to (B, 1, latent_dim) for gather.

      # Create indexing mask (B, 1, latent_dim)
      idx_mask = best_idx.unsqueeze(-1).unsqueeze(-1).repeat(1,
                                                             1, self.args.latent_dim)

      # Gather the selected mu and logvar
      subgoal = torch.gather(mu, 1, idx_mask).squeeze(1)  # (B, latent_dim)
      vae_log_var = torch.gather(
        logvar, 1, idx_mask).squeeze(1)  # (B, latent_dim)

    return subgoal, vae_log_var


class OpponentModel(nn.Module):
  def __init__(self, args: OMGArgs = OMGArgs()):
    super(OpponentModel, self).__init__()
    self.inference_model = None  # TODO SpatialOpponentModel
    self.optimizer = torch.optim.Adam(
      self.inference_model.parameters(), lr=args.lr)
    self.replay = ReplayBuffer(args.capacity)
    self.device = args.device
    self.args = args

  def forward(self, x: torch.Tensor, history: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the forward pass, using the inference model (CVAE) 
    to predict the reconstructed state and the latent distribution.

    Args:
        x (Tensor): Current state s_t (B, H, W, F).
        history (Dict): Historical trajectory (states/actions).

    Returns:
        Tuple[recon, mu, logvar]
    """
    self.inference_model.eval()
    return self.inference_model.encode(x, history)

  def eval(self):
    """
    Set inference model to evaluation mode
    """
    self.inference_model.eval()
    self.prior_model.eval()
    return

  def loss_function(
          self, reconstructed_x, x, cvae_mu, cvae_log_var, infer_mu, infer_log_var, dones, eta, beta):
    """
    Calculates the full OMG loss for the CVAE.
    """
    # --- 1. Reconstruction Loss ---
    # This forces the CVAE to represent the current state
    recon_loss = 0

    weight_mask = self.get_weight_mask(x)

    bce = F.binary_cross_entropy_with_logits(
      reconstructed_x, x, weight=weight_mask, reduction='none')  # (B, H, W, F)
    recon_loss = bce.mean(dim=[1, 2, 3])

    target_mu, target_log_var = infer_mu.detach(), infer_log_var.detach()

    # KL Divergence between CVAE's prediction and the target distribution
    kl_div_per_example = -0.5 * torch.sum(1 + cvae_log_var - target_log_var -
                                          (((target_mu - cvae_mu) ** 2 +
                                            cvae_log_var.exp()) / target_log_var.exp()),
                                          dim=-1)

    omg_loss = kl_div_per_example.mean()

    # KL Divergence loss || Gaussian KL Divergence for regularization
    kld_loss = -0.5 * torch.mean(1 + cvae_log_var -
                                 cvae_mu.pow(2) - cvae_log_var.exp())

    # --- 3. Total Loss ---
    # + self.args.vae_beta * kld_loss
    total_loss = recon_loss.mean() + beta * omg_loss
    return total_loss

  def train_step(self, batch, agent):
    x = batch['states']
    history = batch['history']
    # (B, H, W) Ground Truth from Hindsight
    target_map = batch['true_goal_map']

    pred_logits = self.forward(x, history)  # (B, H, W)

    # Flatten spatial dimensions for Cross Entropy
    pred_flat = pred_logits.view(pred_logits.shape[0], -1)  # (B, H*W)
    target_indices = target_map.view(
      target_map.shape[0], -1).argmax(dim=1)  # (B,)

    loss = F.cross_entropy(pred_flat, target_indices)

    agent.model_optimizer.zero_grad()
    loss.backward()
    agent.model_optimizer.step()

    return loss.item()


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
