from typing import Dict, Tuple, List
import transformers as t
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from simple_foraging_env import RandomAgent, SimpleAgent
from omg_args import OMGArgs


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

  def train_step(self, batch):
    x = batch['states']
    history = batch['history']
    # (B, H, W) Ground Truth from Hindsight
    target_map = batch['true_goal_map']
    self.inference_model.train()
    pred_logits = self.forward(x, history)  # (B, H, W)

    # Flatten spatial dimensions for Cross Entropy
    pred_flat = pred_logits.view(pred_logits.shape[0], -1)  # (B, H*W)
    target_indices = target_map.view(
      target_map.shape[0], -1).argmax(dim=1)  # (B,)

    loss = F.cross_entropy(pred_flat, target_indices)
    loss_val = loss.item()
    # Only backprop if loss is significant to save time
    if loss_val > self.args.aux_loss_threshold:
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

    return loss_val
