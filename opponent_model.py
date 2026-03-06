from typing import Dict, Tuple
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
    self.optimizer = torch.optim.Adam(
      self.inference_model.parameters(), lr=args.lr)
    self.device = args.device
    self.args = args

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

    # Only backprop if loss is significant to save time
    if loss.item() > self.args.aux_loss_threshold:
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

    return loss.item()
