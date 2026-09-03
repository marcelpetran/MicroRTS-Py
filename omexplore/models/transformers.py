import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from omexplore.utils.omg_args import OMGArgs


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

        # CNN feature extractor to embed each (H, W, F) state into a d_model
        # vector. 8x downsampling via AvgPool (NOT strided convs): stride-2
        # convolutions fall off the MKLDNN fast path and run ~4x slower per
        # FLOP, and pooling is anti-aliased (stride-2 convs alias). AvgPool
        # (not Max) preserves the sign of the difference channels
        # (x - x_prev), where -1 marks a disappearance under fog of war.
        self.feature_extractor = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(2 * F_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (H // 8) * (W // 8), args.d_model),
        )

        self.pos_encoder = PositionalEncoding(
            args.d_model, seq_len=args.max_history_length + 1, dropout=args.dropout
        )

        # Transformer Encoder. enable_nested_tensor=False: the nested-tensor
        # fast path is prototype-stage and unsupported on some backends (MPS).
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.d_model,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=args.num_encoder_layers,
            enable_nested_tensor=False,
        )

        # Spatial Head to predict heatmap of opponent location: d_model -> H*W
        self.spatial_head = nn.Sequential(
            nn.Linear(args.d_model, 128), nn.ReLU(), nn.Linear(128, H * W)
        )

    def get_features(self, x: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
        """x, x_prev: (B, H, W, F). x_prev=None means 'no previous frame' (zeros)."""
        # x_pair = torch.cat([x, x_prev], dim=-1)        # (B, H, W, 2F)
        x_pair = torch.cat(
            [x, x - x_prev], dim=-1
        )  # explicit motion encoding (B, H, W, 2F)
        x = x_pair.permute(0, 3, 1, 2).contiguous(memory_format=torch.channels_last)
        return self.feature_extractor(x)

    def forward(
        self,
        x: torch.Tensor,
        history: Dict[str, torch.Tensor],
        cached_features: bool = False,
    ) -> torch.Tensor:
        """
        x: (B, H, W, F) Current state
        history: Dict containing padded 'states' (B, T, H, W, F) and 'mask' (B, T)
        """
        B, H, W, F_dim = x.shape

        # Embed History
        hist_mask = history["mask"]  # (B, T) True for valid tokens
        T = hist_mask.shape[1]

        if cached_features:
            hist_feat = history["state_features"]  # (B, T, d_model), paired at rollout
            x_prev = history.get("prev_obs", torch.zeros_like(x))  # (B, H, W, F)
        else:
            hist_states = history["states"]  # (B, T, H, W, F)
            prev_states = torch.zeros_like(hist_states)
            prev_states[:, 1:] = hist_states[:, :-1]
            # True predecessor of the oldest window frame. For a full window
            # (start > 0) this is the real previous observation; for a partial
            # window it is zeros, matching the rollout's episode-start pairing.
            prev_first = history.get("prev_first")
            if prev_first is None:
                prev_first = torch.zeros_like(hist_states[:, 0])
            prev_states[:, 0] = prev_first
            valid = hist_mask.reshape(-1)
            hist_flat = hist_states.reshape(B * T, H, W, F_dim)[valid]
            prev_flat = prev_states.reshape(B * T, H, W, F_dim)[valid]
            feats_valid = self.get_features(hist_flat, prev_flat)
            hist_feat = torch.zeros(
                B * T, self.args.d_model, device=x.device, dtype=feats_valid.dtype
            )
            hist_feat[valid] = feats_valid
            hist_feat = hist_feat.reshape(B, T, -1)
            x_prev = hist_states[:, -1] * hist_mask[:, -1].view(B, 1, 1, 1).float()

        x_feat = self.get_features(x, x_prev).unsqueeze(1)

        # Prepend current state
        seq_feats = torch.cat([x_feat, hist_feat], dim=1)  # (B, 1 + T, d_model)

        # Index 0 current state x is always valid
        x_mask = torch.ones((B, 1), dtype=torch.bool, device=x.device)
        full_mask = torch.cat([x_mask, hist_mask], dim=1)  # (B, 1 + T)

        # Positional encoding
        seq_feats = seq_feats * np.sqrt(self.args.d_model)
        seq_feats = self.pos_encoder(seq_feats)  # (B, 1 + T, d_model)

        # Transformer pass
        # src_key_padding_mask expects True for PADDING
        src_key_padding_mask = ~full_mask
        memory = self.transformer(seq_feats, src_key_padding_mask=src_key_padding_mask)

        # Extract summary and predict
        # (B, d_model) - summary token corresponding to current state
        final_memory = memory[:, 0, :]

        logits = self.spatial_head(final_memory)  # (B, H*W)
        heatmap_logits = logits.view(B, H, W)

        return heatmap_logits
