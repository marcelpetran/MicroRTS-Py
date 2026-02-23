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


class OpponentModelOracle(nn.Module):
  def __init__(self, args: OMGArgs = OMGArgs()):
    super(OpponentModelOracle, self).__init__()
    self.inference_model = t.SpatialOpponentModel(args)
    self.optimizer = torch.optim.Adam(
      self.inference_model.parameters(), lr=args.lr)
    self.replay = ReplayBuffer(args.capacity)
    self.device = args.device
    self.args = args

  def forward(self, x: torch.Tensor, history: Dict) -> torch.Tensor:  # Notice the fixed return type
    B, H, W, _ = x.shape

    # 1. Initialize with large negative numbers so Softmax pushes them to ~0%
    g_logits = torch.full((B, H, W), -10.0, device=self.device)

    # Opponent start position (Row 3, Col 6 for 7x7 grid, adjust if env_size changes)
    opp_start = torch.tensor([3, 6], device=self.device).float()

    for b in range(B):
      food_indices = (x[b, :, :, 1] == 1).nonzero(as_tuple=False)
      opp_indices = (x[b, :, :, 3] == 1).nonzero(as_tuple=False).float()

      if len(food_indices) == 0:
        continue  # No food left, return uniform -10.0 map

      ambiguous = False

      # If there's multiple foods and opponent has moved from start
      if len(food_indices) > 1 and len(opp_indices) > 0 and not torch.all(opp_indices[0] == opp_start):
        opp_idx = opp_indices[0]
        # find closest food to opponent
        dists = torch.norm(food_indices.float() - opp_idx, dim=1)

        # Sort distances to check the gap between closest and second closest
        sorted_dists, _ = torch.sort(dists)
        diff = sorted_dists[1] - sorted_dists[0]

        # If the difference is small, it is ambiguous
        if diff < 0.1:
          ambiguous = True
        else:
          # Not ambiguous: Snap to the closest
          min_idx = torch.argmin(dists)
          target_coords = food_indices[min_idx].long()
          # High logit -> ~100% prob
          g_logits[b, target_coords[0], target_coords[1]] = 10.0

      elif len(food_indices) > 1:
        # At start position with multiple foods -> Ambiguous
        ambiguous = True
      elif len(food_indices) == 1:
        # Only one food left -> Obvious target
        target_coords = food_indices[0].long()
        g_logits[b, target_coords[0], target_coords[1]] = 10.0

      # Handle the Ambiguous Case safely
      if ambiguous:
        # Give equal high logits to ALL remaining foods (e.g., 50/50 split after Softmax)
        for f_idx in food_indices:
          g_logits[b, f_idx[0].long(), f_idx[1].long()] = 10.0

    return g_logits

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

  def train_step(self, batch):
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
