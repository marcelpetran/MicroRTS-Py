from typing import Dict, Tuple, List
import random
import math
from scipy import spatial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from omg_args import OMGArgs
from tqdm import tqdm
import wandb


class OpponentModel(nn.Module):
  def __init__(self, model, args: OMGArgs = OMGArgs()):
    super(OpponentModel, self).__init__()
    self.inference_model = model
    # self.inference_model = torch.compile(self.inference_model)
    self.optimizer = torch.optim.Adam(
      self.inference_model.parameters(), lr=args.lr)
    self.device = args.device
    self.args = args

  def collate_history(self, histories: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Pads history sequences to max_len within the batch.
    """
    if not histories:
      return {}

    true_lengths = [len(h.get("states", [])) for h in histories]
    max_len = self.args.max_history_length

    if max_len == 0:
      return {
          "states": torch.empty(0).to(self.device),
          "actions": torch.empty(0).to(self.device),
          "mask": torch.empty(0).to(self.device)
      }

    B = len(histories)
    H, W, F_dim = self.args.state_shape

    padded_states_np = np.zeros((B, max_len, H, W, F_dim), dtype=np.float32)
    padded_actions_np = np.zeros((B, max_len), dtype=np.int64)

    for i, h in enumerate(histories):
      state_seq = h.get("states", [])
      action_seq = h.get("actions", [])

      s_len = len(state_seq)
      if s_len > 0:
        padded_states_np[i, -s_len:] = state_seq

      a_len = len(action_seq)
      if a_len > 0:
        flat_actions = np.array(action_seq, dtype=np.int64).flatten()
        valid_len = min(len(flat_actions), max_len)
        padded_actions_np[i, -valid_len:] = flat_actions[-valid_len:]

    final_padded_states = torch.from_numpy(padded_states_np).to(self.device)
    final_padded_actions = torch.from_numpy(padded_actions_np).to(self.device)

    # Fast mask generation (driven by state length, which is mathematically correct)
    true_lengths_np = np.array(true_lengths, dtype=np.int64)
    lengths_tensor = torch.from_numpy(
      true_lengths_np).to(self.device).unsqueeze(1)
    mask = torch.arange(max_len, device=self.device).expand(
      B, max_len) >= (max_len - lengths_tensor)

    return {
        "states": final_padded_states,
        "actions": final_padded_actions,
        "mask": mask
    }

  def heatmap_kl_divergence(self, g_map: torch.Tensor, true_goal_map: torch.Tensor) -> float:
    """
    Evaluates how closely the inferred subgoal distribution matches the true intent distribution
    using Kullback-Leibler Divergence. Lower is better (0.0 is perfect).

    Args:
        g_map (torch.Tensor): Inferred subgoal heatmap as logits, shape (B, H, W)
        true_goal_map (torch.Tensor): Ground truth distribution over subgoals, shape (B, H, W)
    """
    B = g_map.shape[0]
    g_map_flat = g_map.view(B, -1)  # (B, H*W)
    true_goal_flat = true_goal_map.view(B, -1)  # (B, H*W)

    # Convert logits to log-probabilities for PyTorch's kl_div
    log_probs = F.log_softmax(g_map_flat, dim=-1)

    # Compute KL Divergence
    kl_div = F.kl_div(log_probs, true_goal_flat, reduction='batchmean')

    return kl_div.item()

  def top1_spatial_error(self, g_map: torch.Tensor, true_goal_map: torch.Tensor) -> float:
    """
    Measures the Manhattan distance between the model's most confident prediction 
    and the closest valid ground-truth target.

    Args:
        g_map (torch.Tensor): Inferred subgoal heatmap as logits, shape (B, H, W)
        true_goal_map (torch.Tensor): Ground truth distribution over subgoals, shape (B, H, W)
    """
    B, H, W = g_map.shape
    g_map_flat = g_map.view(B, -1)

    # Get coordinates of highest predicted probability
    pred_idx = torch.argmax(g_map_flat, dim=-1)
    pred_r, pred_c = pred_idx // W, pred_idx % W

    total_error = 0.0
    for b in range(B):
        # Get all valid true targets for this batch item (where probability > 0)
      true_targets = torch.nonzero(true_goal_map[b] > 0)

      if len(true_targets) > 0:
        # Calculate Manhattan distances from the predicted point to all valid true targets
        dists = torch.abs(
          true_targets[:, 0] - pred_r[b]) + torch.abs(true_targets[:, 1] - pred_c[b])
        # Distance to the nearest valid target
        total_error += torch.min(dists).item()

    return total_error / B

  def pretrain(self, dataset, epochs=10, batch_size=128):
    """
    Enhanced pretraining loop with logging and progress bars.
    """
    print(f"Starting pretraining for {epochs} epochs on {self.device}...")
    step = 0
    for epoch in range(epochs):
      random.shuffle(dataset)
      epoch_losses = []
      epoch_kl_divs = []
      epoch_spatial_errors = []

      pbar = tqdm(range(0, len(dataset), batch_size),
                  desc=f"Epoch {epoch + 1}/{epochs}")

      for i in pbar:
        batch_data = dataset[i: i + batch_size]

        # Prepare batch data
        om_batch = {
          "states": torch.from_numpy(np.stack([b["state"] for b in batch_data], dtype=np.float32)).to(self.device, non_blocking=True),
          "history": self.collate_history([b["history"] for b in batch_data]),
          "true_goal_map": torch.from_numpy(np.stack([b["true_goal_map"] for b in batch_data], dtype=np.float32)).to(self.device, non_blocking=True),
          "true_opp_heatmap": torch.from_numpy(np.stack([b["true_opp_heatmap"] for b in batch_data], dtype=np.float32)).to(self.device, non_blocking=True),
        }

        loss, kl_error, spatial_error = self.pretrain_step(om_batch, step=step, epoch=epoch)
        epoch_losses.append(loss)
        epoch_kl_divs.append(kl_error)
        epoch_spatial_errors.append(spatial_error)
        step += 1

        # Update progress bar suffix with current loss
        pbar.set_postfix({"loss": f"{loss:.4f}"})

      avg_loss = sum(epoch_losses) / len(epoch_losses)
      avg_kl_div = sum(epoch_kl_divs) / len(epoch_kl_divs)
      avg_spatial_error = sum(epoch_spatial_errors) / len(epoch_spatial_errors)
      print(f"  => Average Loss: {avg_loss:.6f}")

      # Log epoch-level metrics
      wandb.log({
        "pretrain/epoch_loss": avg_loss,
        "pretrain/epoch_kl_divergence": avg_kl_div,
        "pretrain/epoch_top1_spatial_error": avg_spatial_error,
        "epoch": epoch
      })

  def forward(self, x: torch.Tensor, history: Dict, cached_features=True) -> torch.Tensor:
    """
    Calculates the forward pass, using the inference model 
    to predict the opponent's subgoal.

    Args:
        x (Tensor): Current state s_t (B, H, W, F).
        history (Dict): Historical trajectory (states/opp_actions).

    Returns:
        Heatmap (B, H, W) of the predicted subgoal location.
    """
    return self.inference_model(x, history, cached_features=cached_features)

  def _generate_soft_targets(self, target_map: torch.Tensor, sigma: float = 1.0):
    """
    Applies a Gaussian filter directly on the GPU using PyTorch Conv2d.
    This makes model learn faster and maybe even avoids getting stuck in local minima 
    as it provides a smoother gradient signal compared to a hard one-hot target. 
    The sigma parameter controls how much smoothing is applied, 
    with higher values creating a wider "hill" around the true target location.
    target_map: (B, H, W)
    """
    kernel_size = int(2 * math.ceil(2 * sigma) + 1)

    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, dtype=torch.float32,
                     device=target_map.device)
    x = x - kernel_size // 2
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Create 2D Gaussian kernel via outer product
    kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, K, K)

    # Reshape target map for convolution: (B, C, H, W) where C=1
    target_reshaped = target_map.unsqueeze(1)

    # Apply padding to maintain spatial dimensions
    padding = kernel_size // 2
    soft_targets = F.conv2d(target_reshaped, kernel_2d, padding=padding)

    # Re-normalize each map in the batch so the peak is exactly 1.0
    # Flatten spatial dims to find max per batch item
    batch_size = soft_targets.shape[0]
    max_vals = soft_targets.view(batch_size, -1).max(dim=1)[0]

    # Avoid division by zero for empty targets
    max_vals = torch.clamp(max_vals, min=1e-8)
    soft_targets = soft_targets / max_vals.view(batch_size, 1, 1, 1)

    return soft_targets.squeeze(1)  # Return to (B, H, W)
  
  def pretrain_step(self, batch, step=0, epoch=0):
    x = batch['states']
    history = batch['history']
    # (B, H, W) Ground Truth from Hindsight
    target_map = batch['true_goal_map']
    self.inference_model.train()
    pred_logits = self.forward(x, history, cached_features=False)  # (B, H, W)

    # Generate soft targets with Gaussian smoothing
    soft_targets = self._generate_soft_targets(target_map, sigma=1.0)

    loss = F.binary_cross_entropy_with_logits(
        pred_logits.view(pred_logits.shape[0], -1),
        soft_targets.view(soft_targets.shape[0], -1)
    )
    loss_val = loss.item()
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    
    opp_heatmap = batch['true_opp_heatmap'].to(self.device)
    g_map = F.softmax(pred_logits.view(pred_logits.shape[0], -1),
                      dim=-1).view_as(pred_logits) # (B, H, W)
    kl_div = self.heatmap_kl_divergence(g_map, opp_heatmap)
    spatial_error = self.top1_spatial_error(pred_logits, opp_heatmap)
    wandb.log({
      "train/batch_loss": loss_val,
      "train/kl_divergence": kl_div,
      "train/top1_spatial_error": spatial_error,
      "step": step,
      "epoch": epoch
    })

    return loss_val, kl_div, spatial_error

  def train_step(self, batch, cached_features=True):
    x = batch['states']
    history = batch['history']
    # (B, H, W) Ground Truth from Hindsight
    target_map = batch['true_goal_map']
    self.inference_model.train()
    pred_logits = self.forward(x, history, cached_features)  # (B, H, W)

    # Generate soft targets with Gaussian smoothing
    soft_targets = self._generate_soft_targets(target_map, sigma=1.0)

    loss = F.binary_cross_entropy_with_logits(
        pred_logits.view(pred_logits.shape[0], -1),
        soft_targets.view(soft_targets.shape[0], -1)
    )
    loss_val = loss.item()
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return loss_val
