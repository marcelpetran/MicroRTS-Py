from typing import final
from git import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from omg_args import OMGArgs


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

  def __init__(self, H, W, state_feature_splits, d_model, dropout, state_token: Optional[torch.Tensor] = None):
    super().__init__()
    self.h = H
    self.w = W
    self.seq_len = H * W
    self.state_feature_splits = state_feature_splits
    self.d_model = d_model
    self.dropout = dropout
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
    if self.state_token is not None:
      embedded += self.state_token
    return embedded * math.sqrt(self.d_model)


class TrajectoryEmbedder(nn.Module):
  def __init__(self, H, W, state_feature_splits, d_model, dropout, state_token: Optional[torch.Tensor] = None):
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
    flat_for_embedding = trajectory_tensor.view(-1, self.F_dim)
    split_features = torch.split(
      flat_for_embedding, self.state_feature_splits, dim=-1)
    embedded_features = torch.zeros(
        B * T * self.seq_len_per_state, self.d_model, device=trajectory_tensor.device
    )
    for i, feature_tensor in enumerate(split_features):
      embedded_features += self.feature_embedders[i](feature_tensor.float())

    # (B, T*H*W, d_model)
    embedded_features = embedded_features.view(
      B, T * self.seq_len_per_state, self.d_model)

    # Add token type embedding
    if self.state_token is not None:
      embedded_features += self.state_token

    return embedded_features * math.sqrt(self.d_model)


class DiscreteActionEmbedder(nn.Module):
  """Embeds simple discrete actions into a d_model vector."""

  def __init__(self, num_actions, d_model):
    super().__init__()
    self.embedding = nn.Embedding(num_actions, d_model)
    self.d_model = d_model

  def forward(self, actions):
    """
    Args:
        actions (Tensor): A tensor of action indices, shape (B,).
    Returns:
        Tensor: Embedded actions, shape (B, d_model).
    """
    # (B,) -> (B, d_model)
    return self.embedding(actions) * math.sqrt(self.d_model)


class ActionEmbeddings(StateEmbeddings):
  """
  Handles embedding of grid-like action tensors.
  Inherits from StateEmbeddings since the logic is identical.
  """

  def __init__(self, h, w, action_feature_splits, d_model, dropout):
    super().__init__(h, w, action_feature_splits, d_model, dropout)


# Main CVAE Model using Transformer architecture


class TransformerCVAE(nn.Module):
  """
  Conditional Variational Autoencoder with Transformer architecture.
  The encoder takes both the input state and a conditioning trajectory.
  """

  def __init__(self, args: OMGArgs):
    super().__init__()
    self.seq_len = args.H * args.W
    self.droput = args.dropout
    self.args = args
    self.state_token_type = nn.Parameter(torch.randn(1, 1, args.d_model))
    self.action_token_type = nn.Parameter(torch.randn(1, 1, args.d_model))

    # --- Positional Encoding for history + s_t ---
    self.seq_pos_encoder = PositionalEncoding(
      args.d_model,
      seq_len=(args.max_history_length + 1) *
        (self.seq_len + (1 if args.action_dim else 0)),
      dropout=args.dropout)

    # --- Feature Embedding ---
    self.state_embedder = StateEmbeddings(
        args.H, args.W, args.state_feature_splits, args.d_model, args.dropout, self.state_token_type
    )
    self.trajectory_embedder = TrajectoryEmbedder(
        args.H, args.W, args.state_feature_splits, args.d_model, args.dropout
    )
    if args.action_dim is None:
      if args.action_feature_splits is None:
        raise ValueError(
            "Either action_dim or action_feature_splits must be provided."
        )
      self.action_embedder = ActionEmbeddings(
          args.H, args.W, args.action_feature_splits, args.d_model, args.dropout
      )
    else:
      self.action_embedder = DiscreteActionEmbedder(
        args.action_dim, args.d_model)

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
        args.d_model, seq_len=self.seq_len, dropout=self.droput
    )
    decoder_layer = nn.TransformerDecoderLayer(
        args.d_model, args.nhead, args.dim_feedforward, batch_first=True
    )
    self.transformer_decoder = nn.TransformerDecoder(
        decoder_layer, args.num_decoder_layers
    )
    unconditioned_decoder_layer = nn.TransformerEncoderLayer(
        args.d_model, args.nhead, args.dim_feedforward, batch_first=True
    )
    self.unconditioned_decoder = nn.TransformerEncoder(
        unconditioned_decoder_layer, args.num_decoder_layers
    )

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

  def get_history_seq(self, history):
    """
    Concatenates the embedded history into a single sequence tensor.
    Args:
        history (dict): Dictionary containing 'states', 'actions' and 'mask'.
    Returns:
        Tensor: Concatenated history sequence of shape (B, total_seq_len, d_model)
    """
    states = history["states"]  # (B, T, H, W, F), where T is longest history
    actions = history["actions"]  # (B, T) discrete actions
    # (B, T) boolean mask for real vs padded tokens
    mask = history.get("mask", None)

    if isinstance(states, list):
      if not states:
        return torch.empty(1, 0, self.args.d_model, device=self.args.device), \
            torch.empty(1, 0, dtype=torch.bool, device=self.args.device)

      # evaluation mode - states are in format list of (H, W, F) of length T
      states = torch.stack(states, dim=0).unsqueeze(0)  # (1, T, H, W, F)
      actions = torch.stack(actions, dim=0).unsqueeze(0)  # (1, T)

    states = states.to(self.args.device)
    actions = actions.to(self.args.device)

    B, T, _, _, _ = states.shape
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
    action_embeddings = action_embeddings.view(
      B, T, 1, self.args.d_model)  # (B, T, 1, d_model)

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
      condition_seq, condition_mask = self.get_history_seq(history)
    else:
      condition_seq, condition_mask = history
    x_mask = torch.ones(
      B, x_embedded.shape[1], dtype=torch.bool, device=self.args.device)

    combined_mask = torch.cat([condition_mask, x_mask], dim=1)
    combined_seq = torch.cat([condition_seq, x_embedded], dim=1)
    combined_seq = self.seq_pos_encoder(combined_seq)
    # PyTorch's mask expects True for padded tokens, so we invert our boolean mask
    encoder_output = self.transformer_encoder(
        combined_seq,
        src_key_padding_mask=~combined_mask
    )
    aggregated_output = encoder_output[:, 0, :]

    mu = self.fc_mu(aggregated_output)
    logvar = self.fc_logvar(aggregated_output)
    return mu, logvar

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

  def decode(self, z, history, is_history_seq=False):
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
    if not is_history_seq:
      history_seq, mask = self.get_history_seq(history)
    else:
      history_seq, mask = history

    B = z.shape[0]
    memory = self.seq_pos_encoder(history_seq)

    decoder_input = self.latent_to_decoder_input(z).view(B, self.seq_len, -1)
    tgt = self.decoder_pos_encoder(decoder_input)

    if history_seq.shape[1] > 0:
      decoder_output = self.transformer_decoder(
          tgt=tgt,
          memory=memory,
          memory_key_padding_mask=~mask
      )
    else:
      # If history is empty, use the unconditioned VAE-style decoder
      decoder_output = self.unconditioned_decoder(tgt)

    reconstructed_features = [
        proj(decoder_output) for proj in self.output_projectors
    ]
    reconstructed_x_flat = torch.cat(reconstructed_features, dim=-1)
    reconstructed_x = reconstructed_x_flat.view(
        B, self.state_embedder.h, self.state_embedder.w, -1
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
    history_seq, mask = self.get_history_seq(history)
    history_seq = history_seq.to(self.args.device)
    mu, logvar = self.encode(x, (history_seq, mask), is_history_seq=True)
    z = self.reparameterize(mu, logvar)
    reconstructed_x = self.decode(z, (history_seq, mask), is_history_seq=True)
    return reconstructed_x, mu, logvar


class TransformerVAE(nn.Module):
  def __init__(self, args: OMGArgs):
    super().__init__()
    self.seq_len = args.H * args.W
    self.dropout = args.dropout
    self.args = args

    # --- Feature Embedding ---
    self.embedd = StateEmbeddings(
        args.H, args.W, args.state_feature_splits, args.d_model, args.dropout
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
    x_embedded = self.pos_encoder(self.embedd(x))
    encoder_output = self.transformer_encoder(x_embedded)
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
      B, self.embedd.h, self.embedd.w, -1)
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
