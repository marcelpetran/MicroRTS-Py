from typing import Dict, Tuple, List
import random
import math
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
    self.inference_model = torch.compile(self.inference_model)
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
      return {"states": torch.empty(0).to(self.device), "actions": torch.empty(0).to(self.device), "mask": torch.empty(0).to(self.device)}

    B = len(histories)
    H, W, F_dim = self.args.state_shape

    padded_states_np = np.zeros((B, max_len, H, W, F_dim), dtype=np.float32)
    padded_actions_np = np.zeros((B, max_len), dtype=np.int64)

    for i, h in enumerate(histories):
      seq_len = true_lengths[i]
      if seq_len > 0:
        padded_states_np[i, :seq_len] = h["states"]
        padded_actions_np[i, :seq_len] = h["actions"]

    final_padded_states = torch.from_numpy(padded_states_np).to(self.device)
    final_padded_actions = torch.from_numpy(padded_actions_np).to(self.device)

    # Fast mask generation
    true_lengths_np = np.array(true_lengths, dtype=np.int64)
    mask = torch.arange(max_len, device=self.device).expand(
      B, max_len) < torch.from_numpy(true_lengths_np).to(self.device).unsqueeze(1)
    return {
        "states": final_padded_states,
        "actions": final_padded_actions,
        "mask": mask
    }

  def pretrain(self, dataset, epochs=10, batch_size=128, use_wandb=False, writer=None):
    """
    Enhanced pretraining loop with logging and progress bars.
    """
    print(f"Starting pretraining for {epochs} epochs on {self.device}...")
    
    for epoch in range(epochs):
      random.shuffle(dataset)
      epoch_losses = []
      
      pbar = tqdm(range(0, len(dataset), batch_size), desc=f"Epoch {epoch+1}/{epochs}")
      
      for i in pbar:
        batch_data = dataset[i : i + batch_size]
        
        # Prepare batch data
        om_batch = {
          "states": torch.from_numpy(np.stack([b["state"] for b in batch_data], dtype=np.float32)).to(self.device, non_blocking=True),
          "history": self.collate_history([b["history"] for b in batch_data]),
          "true_goal_map": torch.from_numpy(np.stack([b["true_goal_map"] for b in batch_data], dtype=np.float32)).to(self.device, non_blocking=True)
        }

        loss = self.train_step(om_batch)
        epoch_losses.append(loss)
        
        # Update progress bar suffix with current loss
        pbar.set_postfix({"loss": f"{loss:.4f}"})

        # Log individual steps if using wandb or TB
        step = epoch * (len(dataset) // batch_size) + (i // batch_size)
        if use_wandb:
          wandb.log({"train/batch_loss": loss, "step": step})
        if writer:
          writer.add_scalar("Loss/batch", loss, step)

      avg_loss = sum(epoch_losses) / len(epoch_losses)
      print(f"  => Average Loss: {avg_loss:.6f}")
      
      # Log epoch-level metrics
      if use_wandb:
        wandb.log({"train/epoch_loss": avg_loss, "epoch": epoch})
      if writer:
        writer.add_scalar("Loss/epoch", avg_loss, epoch)

  def forward(self, x: torch.Tensor, history: Dict) -> torch.Tensor:
    """
    Calculates the forward pass, using the inference model 
    to predict the opponent's subgoal.

    Args:
        x (Tensor): Current state s_t (B, H, W, F).
        history (Dict): Historical trajectory (states/opp_actions).

    Returns:
        Heatmap (B, H, W) of the predicted subgoal location.
    """
    return self.inference_model(x, history)

  def _generate_soft_targets(self, target_map: torch.Tensor, sigma: float = 1.0):
    """
    Applies a Gaussian filter directly on the GPU using PyTorch Conv2d.
    target_map: (B, H, W)
    """
    kernel_size = int(2 * math.ceil(2 * sigma) + 1)
    
    # Create 1D Gaussian kernel
    x = torch.arange(kernel_size, dtype=torch.float32, device=target_map.device)
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
    
    return soft_targets.squeeze(1) # Return to (B, H, W)

  def train_step(self, batch):
    x = batch['states']
    history = batch['history']
    # (B, H, W) Ground Truth from Hindsight
    target_map = batch['true_goal_map']
    self.inference_model.train()
    pred_logits = self.forward(x, history)  # (B, H, W)

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
