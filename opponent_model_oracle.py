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

  def train_step(self, batch):
    # not necessary
    return 0.0
