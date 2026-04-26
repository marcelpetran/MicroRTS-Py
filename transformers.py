from json import decoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from omg_args import OMGArgs
from typing import Dict, List, Optional
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
        nn.Linear(args.d_model, 128),
        nn.ReLU(),
        nn.Linear(128, H * W)
    )

  def visualize_action_embeddings(self):
    # Extract the weights from the embedding layer: shape (4, d_model)
    action_weights = self.action_embedder.weight.detach().cpu().numpy()
    
    # Reduce to 2 dimensions using PCA
    pca = PCA(n_components=2)
    actions_2d = pca.fit_transform(action_weights)
    
    action_labels = ['Up', 'Down', 'Left', 'Right']
    
    plt.figure(figsize=(6, 6))
    plt.scatter(actions_2d[:, 0], actions_2d[:, 1], color='red', s=100)
    
    for i, label in enumerate(action_labels):
        plt.annotate(label, (actions_2d[i, 0], actions_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=12)
        
    plt.title("PCA of Action Embeddings")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

  def get_features(self, x: torch.Tensor) -> torch.Tensor:
    """Helper function to extract features from a single state tensor (B, H, W, F)."""
    x_flat = x.permute(0, 3, 1, 2)
    return self.feature_extractor(x_flat)

  def forward(self, x: torch.Tensor, history: Dict[str, torch.Tensor], cached_features: bool = True) -> torch.Tensor:
    """
    x: (B, H, W, F) Current state
    history: Dict containing padded 'states' (B, T, H, W, F) and 'mask' (B, T)
    """
    B, H, W, F_dim = x.shape

    # Embed current state
    x_feat = self.get_features(x).unsqueeze(1)  # (B, 1, d_model)

    # Embed History
    hist_actions = history['actions']  # (B, T)
    hist_mask = history['mask']      # (B, T) True for valid tokens
    T = hist_actions.shape[1]

    if cached_features:
      hist_feat = history['state_features']  # (B, T, d_model)
    else:
      hist_states = history['states']  # (B, T, H, W, F)
      hist_flat = hist_states.reshape(B * T, H, W, F_dim)
      hist_feat = self.get_features(hist_flat).reshape(B, T, -1)

    hist_action_feat = self.action_embedder(hist_actions)  # (B, T, d_model)

    hist_feat = hist_feat + hist_action_feat  # (B, T, d_model)

    # Prepend current state
    seq_feats = torch.cat([x_feat, hist_feat], dim=1)  # (B, 1 + T, d_model)

    # Index 0 current state x is always valid
    x_mask = torch.ones((B, 1), dtype=torch.bool, device=x.device)
    full_mask = torch.cat([x_mask, hist_mask], dim=1)  # (B, 1 + T)

    # Positional encoding
    seq_feats = seq_feats * np.sqrt(self.args.d_model)
    seq_feats = self.pos_encoder(seq_feats) # (B, 1 + T, d_model)

    # Transformer pass
    # src_key_padding_mask expects True for PADDING
    src_key_padding_mask = ~full_mask
    memory = self.transformer(
      seq_feats, src_key_padding_mask=src_key_padding_mask)

    # Extract summary and predict
    final_memory = memory[:, 0, :] # (B, d_model) - summary token corresponding to current state

    logits = self.spatial_head(final_memory)  # (B, H*W)
    heatmap_logits = logits.view(B, H, W)

    return heatmap_logits
