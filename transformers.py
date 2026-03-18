from json import decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from omg_args import OMGArgs
from typing import Dict, List, Optional


class PositionalEncoding(nn.Module):
  """
  Standard positional encoding as used in the original Transformer paper.
  Adds sine and cosine functions of different frequencies to the input embeddings.
  """

  def __init__(self, d_model: int, seq_len: int, dropout: float):
    super(PositionalEncoding, self).__init__()
    self.d_model = d_model
    self.seq_len = seq_len
    self.dropout = nn.Dropout(dropout)
    # Create the positional encoding matrix (seq_len, d_model)
    pe = torch.zeros(seq_len, d_model)
    # Create a vector of shape (seq_len, 1)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    # Apply sine to even indices and cosine to odd indices
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    # Reshape to (1, seq_len, d_model) for broadcasting
    pe = pe.unsqueeze(0)
    # Register as a buffer so it is not a parameter but still part of the model's state
    self.register_buffer("pe", pe)

  def forward(self, x):
    """
    Args:
        x (Tensor): Input tensor of shape (Batch, Seq_len, d_model)
    Returns:
        Tensor: Output tensor of the same shape as input with positional encoding added
    """
    # x is expected to be (Batch, Seq_len, d_model)
    # We don't want to train the positional encodings -> gradient=False
    # Positional encodings are added to the input embeddings
    x = x + (self.pe[:, : x.size(1), :]).requires_grad_(False)
    return self.dropout(x)



class SpatialOpponentModel(nn.Module):
  def __init__(self, args: OMGArgs):
    super().__init__()
    self.args = args
    H, W, F_dim = args.state_shape

    # CNN feature extractor to embed each (H, W, F) state into a d_model vector
    # Projects (H, W, F) -> d_model
    self.feature_extractor = nn.Sequential(
        nn.Conv2d(F_dim, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16 * H * W, args.d_model)
    )

    self.character_embedder = nn.Sequential(
        nn.Conv2d(F_dim, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * H * W, args.d_model)
    )

    self.action_embedder = nn.Embedding(args.action_dim, args.d_model)

    self.pos_encoder = PositionalEncoding(
      args.d_model, seq_len=args.max_history_length + 1, dropout=args.dropout)

    # Transformer Encoder
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=args.d_model,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        batch_first=True
    )
    self.transformer = nn.TransformerEncoder(
      encoder_layer, num_layers=args.num_encoder_layers)

    # Spatial Head to predict heatmap of opponent location: d_model -> H*W
    self.spatial_head = nn.Sequential(
        nn.Linear(args.d_model*2, 128),
        nn.ReLU(),
        nn.Linear(128, H * W)
    )

  def forward(self, x: torch.Tensor, history: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    x: (B, H, W, F) Current state
    history: Dict containing padded 'states' (B, T, H, W, F) and 'mask' (B, T)
    """
    B, H, W, F_dim = x.shape

    # Heuristic fallback if no history is available: predict opponent on food tiles
    if not history or 'states' not in history or history['states'].numel() == 0:
      food_channel = x[:, :, :, 1]
      logits = torch.where(food_channel > 0.5,
                           torch.tensor(10.0, device=x.device),
                           torch.tensor(-10.0, device=x.device))
      return logits

    # Embed current state
    x_flat = x.permute(0, 3, 1, 2)  # (B, F, H, W)
    x_feat = self.feature_extractor(x_flat).unsqueeze(1)  # (B, 1, d_model)

    # Embed History
    hist_states = history['states']  # (B, T, H, W, F)
    hist_actions = history['actions']  # (B, T)
    hist_mask = history['mask']      # (B, T) True for valid tokens
    T = hist_states.shape[1]

    hist_flat = hist_states.view(B * T, H, W, F_dim).permute(0, 3, 1, 2)
    hist_feat = self.feature_extractor(
      hist_flat).view(B, T, -1)  # (B, T, d_model)

    hist_action_feat = self.action_embedder(hist_actions)  # (B, T, d_model)

    hist_feat = hist_feat + hist_action_feat  # (B, T, d_model)

    char_feats = self.character_embedder(hist_flat).view(B, T, -1)
    char_mask = hist_mask.unsqueeze(-1).float() # (B, T, 1)
    char_embed = (char_feats * char_mask).sum(dim=1) / (char_mask.sum(dim=1) + 1e-8) # (B, d_model)

    # Prepend current state
    seq_feats = torch.cat([x_feat, hist_feat], dim=1)  # (B, 1 + T, d_model)

    # Index 0 current state x is always valid
    x_mask = torch.ones((B, 1), dtype=torch.bool, device=x.device)
    full_mask = torch.cat([x_mask, hist_mask], dim=1)  # (B, 1 + T)

    # Positional encoding
    seq_feats = seq_feats * np.sqrt(self.args.d_model)
    seq_feats = self.pos_encoder(seq_feats)

    # Transformer pass
    # src_key_padding_mask expects True for PADDING
    src_key_padding_mask = ~full_mask
    memory = self.transformer(
      seq_feats, src_key_padding_mask=src_key_padding_mask)

    # Extract summary and predict
    final_memory = memory[:, 0, :]

    combined_feat = torch.cat([final_memory, char_embed], dim=-1)  # (B, 2*d_model)

    logits = self.spatial_head(combined_feat)  # (B, H*W)
    heatmap_logits = logits.view(B, H, W)

    return heatmap_logits
