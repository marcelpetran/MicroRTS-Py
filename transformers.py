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


class StateEmbeddings(nn.Module):
  """
  This module handles the embedding of the complex, multi-feature game state.
  It takes the (B, H, W, F) tensor, flattens it, embeds each feature group,
  sums them, and adds positional encoding.
  """

  def __init__(self, H, W, state_feature_splits, d_model, state_token):
    super().__init__()
    self.h = H
    self.w = W
    self.seq_len = H * W
    self.state_feature_splits = state_feature_splits
    self.d_model = d_model
    self.state_token = state_token

    # Separate linear layer for each one-hot feature group
    self.feature_embedders = nn.ModuleList(
        [nn.Linear(size, d_model, bias=False) for size in state_feature_splits]
    )

  def forward(self, state_tensor):
    """
    Args:
        state_tensor (Tensor): Input state of shape (B, H, W, F)
    Returns:
        Tensor: Embedded state of shape (B, H*W, d_model)
    """
    if state_tensor.dim() == 5 and state_tensor.shape[1] == 1:
      state_tensor = state_tensor[:, 0]  # (B,1,H,W,F) -> (B,H,W,F)

    assert state_tensor.dim() == 4, (
        f"Expected (B,H,W,F), got {tuple(state_tensor.shape)}"
    )

    B = state_tensor.shape[0]

    # Flatten spatial dimensions: (B, H, W, F) -> (B, H*W, F)
    state_flat = state_tensor.view(B, self.seq_len, -1)

    F_dim = state_flat.shape[-1]
    assert sum(self.state_feature_splits) == F_dim, (
        f"state_feature_splits must sum to F={F_dim}, got {self.state_feature_splits}"
    )

    # Split the features along the last dimension
    split_features = torch.split(state_flat, self.state_feature_splits, dim=-1)
    # Now I have a list of tensors, each (B, H*W, feature_size)

    # Embed each feature group and sum them up.
    # We initialize with zeros and add each embedding.
    embedded = torch.zeros(
        B, self.seq_len, self.d_model, device=state_tensor.device
    )
    # For each cell in the grid, sum the embeddings of each feature group
    for i, feature_tensor in enumerate(split_features):
      # TODO: WARNING, it may not be good ideat to sum embeddings like this
      embedded += self.feature_embedders[i](feature_tensor.float())
    # embedded: (B, H*W, d_model)

    # Add token type embedding
    embedded += self.state_token

    return embedded


class TrajectoryEmbedder(nn.Module):
  def __init__(self, H, W, state_feature_splits, d_model, state_token):
    super().__init__()
    self.seq_len_per_state = H * W
    self.F_dim = sum(state_feature_splits)
    self.d_model = d_model
    self.state_feature_splits = state_feature_splits
    self.state_token = state_token

    # Feature Embedders
    self.feature_embedders = nn.ModuleList(
        [nn.Linear(size, d_model, bias=False) for size in state_feature_splits]
    )

  def forward(self, trajectory_tensor):
    """
    Args:
        trajectory_tensor (Tensor): Input tensor of shape (B, T, H, W, F)
    Returns:
        Tensor: Embedded trajectory of shape (B, T*H*W, d_model)
    """
    assert trajectory_tensor.dim() == 5, (
        f"Expected (B,T,H,W,F), got {tuple(trajectory_tensor.shape)}"
    )
    B, T, _, _, _ = trajectory_tensor.shape
    flat_for_embedding = trajectory_tensor.view(
      B, -1, self.F_dim)  # (B, T*H*W, d_model)
    split_features = torch.split(
      flat_for_embedding, self.state_feature_splits, dim=-1)
    embedded_features = torch.zeros(
        B, T * self.seq_len_per_state, self.d_model, device=trajectory_tensor.device
    )
    for i, feature_tensor in enumerate(split_features):
      embedded_features += self.feature_embedders[i](feature_tensor.float())

    # Add token type embedding
    embedded_features += self.state_token

    return embedded_features


class DiscreteActionEmbedder(nn.Module):
  """Embeds simple discrete actions into a d_model vector."""

  def __init__(self, num_actions, d_model, action_token_type):
    super().__init__()
    self.embedding = nn.Embedding(num_actions, d_model)
    self.d_model = d_model
    self.action_token_type = action_token_type

  def forward(self, actions):
    """
    Args:
        actions (Tensor): A tensor of action indices, shape (B,).
    Returns:
        Tensor: Embedded actions, shape (B, d_model).
    """
    # (B,) -> (B, d_model)
    return self.embedding(actions) + self.action_token_type


class ActionEmbeddings(StateEmbeddings):
  """
  Handles embedding of grid-like action tensors.
  Inherits from StateEmbeddings since the logic is identical.
  """

  def __init__(self, h, w, action_feature_splits, d_model, action_token_type):
    super().__init__(h, w, action_feature_splits, d_model, action_token_type)


# Main CVAE Model using Transformer architecture


class TransformerCVAE(nn.Module):
  """
  Conditional Variational Autoencoder with Transformer architecture.
  The encoder takes both the input state and a conditioning trajectory.
  """

  def __init__(self, args: OMGArgs):
    super().__init__()
    self.seq_len = args.H * args.W
    self.dropout = args.dropout
    self.args = args
    self.state_token_type = nn.Parameter(torch.randn(1, 1, args.d_model))
    self.action_token_type = nn.Parameter(torch.randn(1, 1, args.d_model))
    self.cls_token = nn.Parameter(torch.randn(1, 1, args.d_model))
    self.empty_history_token = nn.Parameter(torch.randn(1, 1, args.d_model))

    # --- Positional Encoding for history + s_t + CLS token ---
    self.seq_pos_encoder = PositionalEncoding(
      args.d_model,
      seq_len=1 + (args.max_history_length + 1) *
        (self.seq_len + (1 if args.action_dim else 0)),
      dropout=args.dropout)

    # --- Feature Embedding ---
    self.state_embedder = StateEmbeddings(
        args.H, args.W, args.state_feature_splits, args.d_model, self.state_token_type
    )
    self.trajectory_embedder = TrajectoryEmbedder(
        args.H, args.W, args.state_feature_splits, args.d_model, self.state_token_type
    )
    if args.action_dim is None:
      if args.action_feature_splits is None:
        raise ValueError(
            "Either action_dim or action_feature_splits must be provided."
        )
      self.action_embedder = ActionEmbeddings(
          args.H, args.W, args.action_feature_splits, args.d_model, self.action_token_type
      )
    else:
      self.action_embedder = DiscreteActionEmbedder(
        args.action_dim, args.d_model, self.action_token_type)

    # --- Encoder ---
    encoder_layer = nn.TransformerEncoderLayer(
        args.d_model, args.nhead, args.dim_feedforward, batch_first=True
    )
    self.transformer_encoder = nn.TransformerEncoder(
        encoder_layer, args.num_encoder_layers
    )
    self.fc_mu = nn.Linear(args.d_model, args.latent_dim)
    self.fc_logvar = nn.Linear(args.d_model, args.latent_dim)

    # --- Decoder ---
    self.latent_to_decoder_input = nn.Linear(
        args.latent_dim, self.seq_len * args.d_model
    )
    self.decoder_pos_encoder = PositionalEncoding(
        args.d_model, seq_len=self.seq_len, dropout=self.dropout
    )
    decoder_layer = nn.TransformerDecoderLayer(
        args.d_model, args.nhead, args.dim_feedforward, batch_first=True
    )
    self.transformer_decoder = nn.TransformerDecoder(
        decoder_layer, args.num_decoder_layers
    )
    # unconditioned_decoder_layer = nn.TransformerEncoderLayer(
    #     args.d_model, args.nhead, args.dim_feedforward, batch_first=True
    # )
    # self.unconditioned_decoder = nn.TransformerEncoder(
    #     unconditioned_decoder_layer, args.num_decoder_layers
    # )

    # --- Output Projection ---
    self.output_projectors = nn.ModuleList(
        [nn.Linear(args.d_model, size) for size in args.state_feature_splits]
    )

    # Set xavier initialization for all linear layers
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
          nn.init.zeros_(m.bias)

  def get_history_seq(self, history, B):
    """
    Concatenates the embedded history into a single sequence tensor.
    Args:
        history (dict): Dictionary containing 'states', 'actions' and 'mask'.
    Returns:
        Tensor: Concatenated history sequence of shape (B, total_seq_len, d_model)
    """
    states = history["states"]  # (B, T, H, W, F), where T is longest history
    actions = history["actions"]  # (B, T) discrete actions
    # opp_actions = history["opp_actions"]  # (B, T) discrete actions
    # (B, T) boolean mask for real vs padded tokens
    mask = history.get("mask", None)

    if isinstance(states, list):
      if not states:
        return torch.empty(B, 0, self.args.d_model, device=self.args.device), \
            torch.empty(B, 0, dtype=torch.bool, device=self.args.device)

      # evaluation mode - states are in format list of (H, W, F) of length T
      states = torch.stack(states, dim=0).unsqueeze(0)  # (1, T, H, W, F)
      actions = torch.stack(actions, dim=0).unsqueeze(0)  # (1, T)

    states = states.to(self.args.device)
    actions = actions.to(self.args.device)

    T = states.shape[1]
    if mask is None:
      # during collection or evaluation -> all tokens are valid
      mask = torch.ones(B, T, dtype=torch.bool, device=self.args.device)

    state_mask = mask.unsqueeze(-1).repeat(1, 1, self.seq_len)  # (B, T, H*W)
    action_mask = mask.unsqueeze(-1)  # (B, T, 1)
    final_mask = torch.cat([state_mask, action_mask],
                           dim=2).view(B, -1)  # (B, T*(H*W + 1))

    state_embeddings = self.trajectory_embedder(states)  # (B, T*H*W, d_model)
    action_embeddings = self.action_embedder(actions)  # (B, T, d_model)
    state_embeddings = state_embeddings.view(
      B, T, self.seq_len, self.args.d_model)  # (B, T, H*W, d_model)
    action_embeddings = action_embeddings.unsqueeze(2)  # (B, T, 1, d_model)

    # Interleave state and action embeddings
    # (B, T, H*W + 1, d_model)
    interleaved_seq = torch.cat([state_embeddings, action_embeddings], dim=2)
    concatenated_seq = interleaved_seq.view(
      B, -1, self.args.d_model)  # (B, T * (H*W + 1), d_model)

    # return self.history_pos_encoder(concatenated_seq), final_mask
    return concatenated_seq, final_mask

  def encode(self, x, history, is_history_seq=False):
    """
    Encodes the input state x conditioned on the historical trajectory.
    Args:
        x (Tensor): Input state of shape (B, H, W, F)
        history (tensor or dict): The embedded history sequence or raw history dict.
        is_history_seq (bool): If True, 'history' is already an embedded sequence.
    Returns:
        Tensor: Mean of the latent distribution (B, latent_dim)
        Tensor: Log-variance of the latent distribution (B, latent_dim)
    """
    x = x.to(self.args.device)
    x_embedded = self.state_embedder(x)  # (B, H*W, d_model)
    B = x_embedded.shape[0]

    if not is_history_seq:
      condition_seq, condition_mask = self.get_history_seq(history, B)
    else:
      condition_seq, condition_mask = history
    x_mask = torch.ones(
      B, x_embedded.shape[1], dtype=torch.bool, device=self.args.device)

    cls_tokens = self.cls_token.repeat(B, 1, 1)
    cls_mask = torch.ones(B, 1, dtype=torch.bool, device=self.args.device)

    combined_mask = torch.cat(
      [cls_mask, x_mask, condition_mask], dim=1)  # (B, total_seq_len)

    # (B, 1+T*(H*W+1)+H*W, d_model)
    combined_seq = torch.cat([cls_tokens, x_embedded, condition_seq], dim=1)
    combined_seq = self.seq_pos_encoder(combined_seq)
    # PyTorch's mask expects True for padded tokens, so we invert our boolean mask
    encoder_output = self.transformer_encoder(
        combined_seq,
        src_key_padding_mask=~combined_mask
    )  # (B, total_seq_len, d_model)
    aggregated_output = encoder_output[:, 0, :]  # (B, d_model)

    mu = self.fc_mu(aggregated_output)
    logvar = self.fc_logvar(aggregated_output)
    return mu, logvar  # (B, latent_dim), (B, latent_dim)

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

  def decode(self, z, history_seq, history_mask):
    """
    Decodes a latent vector z conditioned on an embedded history sequence.

    Args:
        z (Tensor): The latent vector. Shape: (B, latent_dim)
        history (Tensor or dict): The embedded history sequence or raw history dict.
        is_history_seq (bool): If True, 'history' is already an embedded sequence.
    Returns:
        Tensor: Reconstructed state of shape (B, H, W, F)
    """
    z = z.to(self.args.device)

    B = z.shape[0]

    decoder_input = self.latent_to_decoder_input(z).view(B, self.seq_len, -1)
    tgt = self.decoder_pos_encoder(decoder_input)

    tokens_per_step = (self.args.H * self.args.W)
    if self.args.action_dim:
      tokens_per_step += 1

    # Check if history has at least one step to remove
    if history_seq.shape[1] >= tokens_per_step * 2:
      # Remove the last `tokens_per_step` tokens from the sequence and mask
      memory_seq = history_seq[:, :-tokens_per_step, :]
      memory_mask = history_mask[:, :-tokens_per_step]
    else:
      # History is empty or shorter than one step, just pass it along
      memory_seq = self.empty_history_token.repeat(B, 1, 1)
      memory_mask = torch.ones(B, 1, dtype=torch.bool, device=self.args.device)

    self.seq_pos_encoder(memory_seq)
    decoder_output = self.transformer_decoder(
      tgt=tgt,
      memory=memory_seq,
      memory_key_padding_mask=~memory_mask
    )
    # decoder_output = self.unconditioned_decoder(tgt)

    reconstructed_features = [
        proj(decoder_output) for proj in self.output_projectors
    ]
    reconstructed_x_flat = torch.cat(reconstructed_features, dim=-1)
    reconstructed_x = reconstructed_x_flat.view(
        B, self.args.H, self.args.W, -1
    )
    return reconstructed_x

  def forward(self, x, history):
    """
    Args:
        x (Tensor): Input state of shape (B, H, W, F)
        c (Tensor): Conditioning trajectory of shape (B, H, W, F)
    Returns:
        Tensor: Reconstructed state of shape (B, H, W, F)
        Tensor: Mean of the latent distribution (B, latent_dim)
        Tensor: Log-variance of the latent distribution (B, latent_dim)
    """
    x = x.to(self.args.device)
    history_seq, mask = self.get_history_seq(history, x.shape[0])
    history_seq = history_seq.to(self.args.device)
    mu, logvar = self.encode(x, (history_seq, mask), is_history_seq=True)
    z = self.reparameterize(mu, logvar)
    reconstructed_x = self.decode(z, history_seq, mask)
    return reconstructed_x, mu, logvar


class TransformerVAE(nn.Module):
  def __init__(self, args: OMGArgs):
    super().__init__()
    self.seq_len = args.H * args.W
    self.dropout = args.dropout
    self.args = args
    self.state_token_type = nn.Parameter(torch.randn(1, 1, args.d_model))
    self.cls_token = nn.Parameter(torch.randn(1, 1, args.d_model))

    # --- Feature Embedding ---
    self.embedd = StateEmbeddings(
        args.H, args.W, args.state_feature_splits, args.d_model, self.state_token_type
    )
    self.pos_encoder = PositionalEncoding(
        args.d_model, seq_len=self.seq_len * 2, dropout=self.dropout
    )

    # --- Encoder ---
    encoder_layer = nn.TransformerEncoderLayer(
        args.d_model, args.nhead, args.dim_feedforward, batch_first=True
    )
    self.transformer_encoder = nn.TransformerEncoder(
        encoder_layer, args.num_encoder_layers
    )
    self.fc_mu = nn.Linear(args.d_model, args.latent_dim)
    self.fc_logvar = nn.Linear(args.d_model, args.latent_dim)

    # --- Decoder ---
    self.latent_to_decoder_input = nn.Linear(
        args.latent_dim, self.seq_len * args.d_model
    )
    self.decoder_pos_encoder = PositionalEncoding(
        args.d_model, seq_len=self.seq_len, dropout=self.dropout
    )
    # In a VAE, we don't have a memory sequence to attend to
    # we only have the latent vector z to reconstruct from.
    # TransformerEncoder is simply a stack of self-attention blocks
    # which is exactly what we need to reconstruct a sequence from a starting point
    decoder_layer = nn.TransformerEncoderLayer(
        args.d_model, args.nhead, args.dim_feedforward, batch_first=True
    )
    self.transformer_decoder = nn.TransformerEncoder(
        decoder_layer, args.num_decoder_layers
    )

    # --- Output Projection ---
    self.output_projectors = nn.ModuleList(
        [nn.Linear(args.d_model, size) for size in args.state_feature_splits]
    )

    # set xavier initialization for all linear layers
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
          nn.init.zeros_(m.bias)

  def encode(self, x):
    x = x.to(self.args.device)
    x_embedded = self.embedd(x)
    B = x_embedded.shape[0]
    cls_tokens = self.cls_token.repeat(B, 1, 1)
    x_seq = torch.cat([cls_tokens, x_embedded], dim=1)
    x_seq = self.pos_encoder(x_seq)
    encoder_output = self.transformer_encoder(x_seq)
    # we need single summary vector for mu and logvar
    aggregated_output = encoder_output[:, 0, :]

    mu = self.fc_mu(aggregated_output)
    logvar = self.fc_logvar(aggregated_output)
    return mu, logvar

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

  def decode(self, z):
    B = z.shape[0]
    decoder_input = self.latent_to_decoder_input(z).view(B, self.seq_len, -1)
    tgt = self.decoder_pos_encoder(decoder_input)

    decoder_output = self.transformer_decoder(tgt)

    reconstructed_features = [
        proj(decoder_output) for proj in self.output_projectors
    ]
    reconstructed_x_flat = torch.cat(reconstructed_features, dim=-1)
    reconstructed_x = reconstructed_x_flat.view(
      B, self.args.H, self.args.W, -1)
    return reconstructed_x

  def forward(self, x):
    """
    Args:
        x (Tensor): Input state of shape (B, H, W, F)
    Returns:
        Tensor: Reconstructed state of shape (B, H, W, F)
        Tensor: Mean of the latent distribution (B, latent_dim)
        Tensor: Log-variance of the latent distribution (B, latent_dim)
    """
    mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    reconstructed_x = self.decode(z)
    return reconstructed_x, mu, logvar


class VAE(nn.Module):
  def __init__(self, args: OMGArgs):
    super().__init__()
    self.args = args
    input_dim = sum(args.state_feature_splits)
    hidden_dim = args.dim_feedforward
    latent_dim = args.latent_dim

    # Encoder
    self.encoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
    )
    self.fc_mu = nn.Linear(hidden_dim, latent_dim)
    self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    # Decoder
    self.decoder_input = nn.Linear(latent_dim, hidden_dim)
    self.decoder = nn.Sequential(
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, input_dim),
    )

  def encode(self, x):
    x = x.to(self.args.device)
    h = self.encoder(x)
    mu = self.fc_mu(h)
    logvar = self.fc_logvar(h)
    return mu, logvar

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

  def decode(self, z):
    z = z.to(self.args.device)
    h = self.decoder_input(z)
    x_recon = self.decoder(h)
    return x_recon.view(-1, self.args.H, self.args.W, sum(self.args.state_feature_splits))

  def forward(self, x):
    mu, logvar = self.encode(x)
    z = self.reparameterize(mu, logvar)
    reconstructed_x = self.decode(z)
    return reconstructed_x, mu, logvar


class SpatialOpponentModel(nn.Module):
  def __init__(self, args: OMGArgs):
    super().__init__()
    self.args = args
    H, W, F_dim = args.state_shape

    # 1. Feature Extractor to embed each (H, W, F) state into a d_model vector
    # Projects (H, W, F) -> d_model
    self.feature_extractor = nn.Sequential(
        nn.Conv2d(F_dim, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(16 * H * W, args.d_model)
    )

    self.pos_encoder = PositionalEncoding(
      args.d_model, max_len=args.max_history_length)

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

  def forward(self, history: Dict[str, torch.Tensor]) -> torch.Tensor:
    states = history['states']  # (B, T, H, W, F)
    B, T, H, W, F_dim = states.shape

    # Embed the States
    # (B*T, F, H, W)
    x = states.view(B * T, H, W, F_dim).permute(0, 3, 1, 2)
    feats = self.feature_extractor(x)  # (B*T, d_model)

    # Reshape to (B, T, d_model)
    seq_feats = feats.view(B, T, -1)

    # Add Positional Encoding
    seq_feats = seq_feats * np.sqrt(self.args.d_model)
    seq_feats = self.pos_encoder(seq_feats)

    # C. Pass through Transformer
    # We need a mask for padding if history is variable length!
    # Assuming history['mask'] exists from your collate_fn
    # mask shape: (B, T), True where valid, False where padding
    # Transformer expects: key_padding_mask (B, T) where True is PADDING (ignore)
    # Your collate usually gives True for Valid. Check this!
    src_key_padding_mask = ~history['mask'] if 'mask' in history else None

    memory = self.transformer(
      seq_feats, src_key_padding_mask=src_key_padding_mask)

    # D. Pooling / Summary
    # If using padding, taking -1 is risky (it might be padding).
    # Better to take the embedding at the index of the last *valid* token.
    # For simplicity (assuming batch_first=True):
    if 'mask' in history:
      # Find last True index for each batch
      last_indices = history['mask'].sum(dim=1) - 1  # (B,)
      # Gather: memory[b, last_indices[b], :]
      final_memory = memory[torch.arange(B), last_indices.long(), :]
    else:
      final_memory = memory[:, -1, :]

    # E. Predict Map
    logits = self.spatial_head(final_memory)  # (B, H*W)
    heatmap_logits = logits.view(B, H, W)

    # Standardize output for Agent (add channel dim if needed by QNet)
    # (B, H, W) is fine if your QNet expects it that way.

    return heatmap_logits  # Return logits for Loss, softmax for Agent

  def train_step(self, batch, agent):
    # 1. Get Prediction
    pred_heatmap = self.forward(batch['history'])  # (B, H, W)

    # 2. Get Target (Ground Truth Future Location)
    # We need to create a mask from the future states in the batch
    # Assuming batch['future_states'] contains the actual path
    # You can select the LAST state in the horizon as the target
    future_states = batch['future_states']  # (B, Horizon, H, W, F)
    target_state = future_states[:, -1]    # Take the state at H

    # Extract food channel (index 1) or Agent 2 channel (index 3) from target?
    # Since we want to predict WHERE the opponent ends up:
    target_mask = target_state[:, :, :, 3]  # Channel 3 is Opponent

    # 3. Compute Loss (Cross Entropy)
    # Flatten for loss: Pred (B, H*W), Target (B, H*W) indices
    target_indices = target_mask.view(
      pred_heatmap.shape[0], -1).argmax(dim=1)
    loss = F.cross_entropy(pred_heatmap.view(
      pred_heatmap.shape[0], -1), target_indices)

    # Optimization
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    return loss.item()
