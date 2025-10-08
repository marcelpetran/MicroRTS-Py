from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from q_agent import ReplayBuffer, OMGArgs
from simple_foraging_env import RandomAgent, SimpleAgent
from matplotlib import pyplot as plt


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
    div_term = torch.exp(torch.arange(0, d_model, 2).float()
                         * (-math.log(10000.0) / d_model))
    # Apply sine to even indices and cosine to odd indices
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    # Reshape to (1, seq_len, d_model) for broadcasting
    pe = pe.unsqueeze(0)
    # Register as a buffer so it is not a parameter but still part of the model's state
    self.register_buffer('pe', pe)

  def forward(self, x):
    """
    Args:
        x (Tensor): Input tensor of shape (Batch, Seq_len, d_model)
    Returns:
        Tensor: Output tensor of the same shape as input with positional encoding added
    """
    # x is expected to be (Batch, Seq_len, d_model)
    # We don't want to train the positional encodings
    x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False)
    return self.dropout(x)


class StateEmbeddings(nn.Module):
  """
  This module handles the embedding of the complex, multi-feature game state.
  It takes the (B, H, W, F) tensor, flattens it, embeds each feature group,
  sums them, and adds positional encoding.
  """

  def __init__(self, h, w, state_feature_splits, d_model, dropout):
    super().__init__()
    self.h = h
    self.w = w
    self.seq_len = h * w
    self.state_feature_splits = state_feature_splits
    self.d_model = d_model
    self.dropout = dropout

    # Create a separate linear layer for each one-hot feature group
    self.feature_embedders = nn.ModuleList([
        nn.Linear(size, d_model, bias=False) for size in state_feature_splits
    ])

    # Positional encoding
    self.position_encoder = PositionalEncoding(
      d_model, seq_len=self.seq_len * 2, dropout=self.dropout)

  def forward(self, state_tensor):
    """
    Args:
        state_tensor (Tensor): Input state of shape (B, H, W, F)
    Returns:
        Tensor: Embedded state of shape (B, H*W, d_model)
    """
    if state_tensor.dim() == 5 and state_tensor.shape[1] == 1:
      state_tensor = state_tensor[:, 0]  # (B,1,H,W,F) -> (B,H,W,F)

    assert state_tensor.dim(
    ) == 4, f"Expected (B,H,W,F), got {tuple(state_tensor.shape)}"

    B = state_tensor.shape[0]

    # Flatten spatial dimensions: (B, H, W, F) -> (B, H*W, F)
    state_flat = state_tensor.view(B, self.seq_len, -1)

    F_dim = state_flat.shape[-1]
    assert sum(self.state_feature_splits) == F_dim, \
        f"state_feature_splits must sum to F={F_dim}, got {self.state_feature_splits}"

    # Split the features along the last dimension
    split_features = torch.split(state_flat, self.state_feature_splits, dim=-1)

    # Embed each feature group and sum them up.
    # We initialize with zeros and add each embedding.
    embedded = torch.zeros(B, self.seq_len, self.d_model,
                           device=state_tensor.device)
    for i, feature_tensor in enumerate(split_features):
      # Ensure float for linear layer
      embedded += self.feature_embedders[i](feature_tensor.float())

    # Add positional encoding
    return self.position_encoder(embedded)


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
        Tensor: Embedded actions, shape (B, 1, d_model) ready for sequence concatenation.
    """
    # (B,) -> (B, d_model) -> (B, 1, d_model)
    return self.embedding(actions).unsqueeze(1)


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
    self.null_condition = torch.zeros(
      1, args.H, args.W, sum(args.state_feature_splits))

    # --- Feature Embedding ---
    self.state_embedder = StateEmbeddings(
      args.H, args.W, args.state_feature_splits, args.d_model, args.dropout)
    if args.action_dim is None:
      if args.action_feature_splits is None:
        raise ValueError(
          "Either action_dim or action_feature_splits must be provided.")
      self.action_embedder = ActionEmbeddings(
        args.H, args.W, args.action_feature_splits, args.d_model, args.dropout)
    else:
      self.action_embedder = DiscreteActionEmbedder(
        args.action_dim, args.d_model)

    # --- Encoder ---
    encoder_layer = nn.TransformerEncoderLayer(
      args.d_model, args.nhead, args.dim_feedforward, batch_first=True)
    self.transformer_encoder = nn.TransformerEncoder(
      encoder_layer, args.num_encoder_layers)
    self.fc_mu = nn.Linear(args.d_model, args.latent_dim)
    self.fc_logvar = nn.Linear(args.d_model, args.latent_dim)

    # --- Decoder ---
    self.latent_to_decoder_input = nn.Linear(
      args.latent_dim, self.seq_len * args.d_model)
    self.decoder_pos_encoder = PositionalEncoding(
      args.d_model, seq_len=self.seq_len, dropout=self.droput)
    decoder_layer = nn.TransformerDecoderLayer(
      args.d_model, args.nhead, args.dim_feedforward, batch_first=True)
    self.transformer_decoder = nn.TransformerDecoder(
      decoder_layer, args.num_decoder_layers)

    # --- Output Projection ---
    self.output_projectors = nn.ModuleList([
        nn.Linear(args.d_model, size) for size in args.state_feature_splits
    ])

    # Set xavier initialization for all linear layers
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
          nn.init.zeros_(m.bias)

  def get_history_embeddings(self, history):
    """
    Aggregates and embeds the historical trajectory into a single sequence.
    Expected keys: 'states', 'actions'
    history['states'] is a list of tensors, each either (H, W, F) or (B, H, W, F)
    history['actions'] is a list of tensors, each either () or (B,)
    """
    def to_bhwf(x):
      # Accept (H,W,F) or (B,H,W,F) and return (B,H,W,F)
      if x.dim() == 3:
        return x.unsqueeze(0)
      if x.dim() == 4:
        return x
      raise ValueError(
        f"state has to be (H,W,F) or (B,H,W,F); got {tuple(x.shape)}")

    def to_b(x):
      # Accept scalar () or vector (B,) and return (B,)
      if x.dim() == 0:
        return x.unsqueeze(0)
      if x.dim() == 1:
        return x
      return x.view(x.shape[0])  # be forgiving if shaped (B,1)

    states = history.get("states", [])
    actions = history.get("actions", None)
    have_actions = isinstance(actions, list) and len(actions) == len(states)

    history_embeddings = []
    for t, s_t in enumerate(states):
      s_t = to_bhwf(s_t)                          # (B,H,W,F)
      s_t_embedded = self.state_embedder(s_t)     # (B, H*W, d_model)

      if have_actions:
        a_t = to_b(actions[t])                  # (B,)
        a_t_embedded = self.action_embedder(a_t)  # (B, 1, d_model)
        history_embeddings.append(
          torch.cat([s_t_embedded, a_t_embedded], dim=1))  # (B, H*W+1, d_model)
      else:
        history_embeddings.append(s_t_embedded)

    return history_embeddings
  
  def get_history_seq(self, history):
    """
    Concatenates the embedded history into a single sequence tensor.
    Args:
        history (dict): Dictionary containing 'states' and optionally 'actions'.
    Returns:
        Tensor: Concatenated history sequence of shape (B, total_seq_len, d_model)
    """
    history_embeddings = self.get_history_embeddings(history)
    if history_embeddings is None or len(history_embeddings) == 0:
      # If no history, use a null condition state, therefore it should behave like a regular VAE
      history_embeddings = [self.state_embedder(
        self.null_condition.to(next(self.parameters()).device))]
      
      # Concatenate all history elements into a single long sequence
    return torch.cat(history_embeddings, dim=1)  # (B, total_seq_len, d_model)

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
    x_embedded = self.state_embedder(x)

    if not is_history_seq:
      condition_seq = self.get_history_seq(history)
    else:
      condition_seq = history

    combined_seq = torch.cat([x_embedded, condition_seq], dim=1)
    encoder_output = self.transformer_encoder(combined_seq)
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
    if not is_history_seq:
      history_seq = self.get_history_seq(history)
    else:
      history_seq = history

    B = z.shape[0]
    memory = history_seq

    decoder_input = self.latent_to_decoder_input(z).view(B, self.seq_len, -1)
    tgt = self.decoder_pos_encoder(decoder_input)

    decoder_output = self.transformer_decoder(tgt=tgt, memory=memory)

    reconstructed_features = [proj(decoder_output)
                              for proj in self.output_projectors]
    reconstructed_x_flat = torch.cat(reconstructed_features, dim=-1)
    reconstructed_x = reconstructed_x_flat.view(
      B, self.state_embedder.h, self.state_embedder.w, -1)
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
    history_seq = self.get_history_seq(history)
    mu, logvar = self.encode(x, history_seq, is_history_seq=True)
    z = self.reparameterize(mu, logvar)
    reconstructed_x = self.decode(z, history_seq, is_history_seq=True)
    return reconstructed_x, mu, logvar


class TransformerVAE(nn.Module):
  def __init__(self, args: OMGArgs):
    super().__init__()
    self.seq_len = args.H * args.W
    self.dropout = args.dropout
    self.args = args

    # --- Feature Embedding ---
    self.embedd = StateEmbeddings(
      args.H, args.W, args.state_feature_splits, args.d_model, args.dropout)

    # --- Encoder ---
    encoder_layer = nn.TransformerEncoderLayer(
      args.d_model, args.nhead, args.dim_feedforward, batch_first=True)
    self.transformer_encoder = nn.TransformerEncoder(
      encoder_layer, args.num_encoder_layers)
    self.fc_mu = nn.Linear(args.d_model, args.latent_dim)
    self.fc_logvar = nn.Linear(args.d_model, args.latent_dim)

    # --- Decoder ---
    self.latent_to_decoder_input = nn.Linear(
      args.latent_dim, self.seq_len * args.d_model)
    self.decoder_pos_encoder = PositionalEncoding(
      args.d_model, seq_len=self.seq_len, dropout=self.dropout)
    # In a VAE, we don't have a memory sequence to attend to
    # we only have the latent vector z to reconstruct from.
    # TransformerEncoder is simply a stack of self-attention blocks
    # which is exactly what we need to reconstruct a sequence from a starting point
    decoder_layer = nn.TransformerEncoderLayer(
      args.d_model, args.nhead, args.dim_feedforward, batch_first=True)
    self.transformer_decoder = nn.TransformerEncoder(
      decoder_layer, args.num_decoder_layers)

    # --- Output Projection ---
    self.output_projectors = nn.ModuleList([
        nn.Linear(args.d_model, size) for size in args.state_feature_splits
    ])

    # set xavier initialization for all linear layers
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
          nn.init.zeros_(m.bias)

  def encode(self, x):
    x_embedded = self.embedd(x)
    encoder_output = self.transformer_encoder(x_embedded)
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

    reconstructed_features = [proj(decoder_output)
                              for proj in self.output_projectors]
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

# Loss function for the VAE


def vae_loss(reconstructed_x, x, mu, logvar, state_feature_splits, beta=1.0):
  """
  Loss function for VAE combining reconstruction loss and KL divergence.
  Args:
      reconstructed_x (Tensor): Reconstructed state of shape (B, H, W, F)
      x (Tensor): Original input state of shape (B, H, W, F)
      mu (Tensor): Mean of the latent distribution (B, latent_dim)
      logvar (Tensor): Log-variance of the latent distribution (B, latent_dim)
      state_feature_splits (List[int]): List of sizes for each one-hot feature group.
      beta (float): Weighting factor for the KL divergence term.
  Returns:
      Tensor: Computed VAE loss.
  """
  # Reconstruction Loss
  # We use Binary Cross Entropy with Logits for each one-hot feature group
  # as it's suitable for multi-label classification style outputs
  recon_loss = 0

  batch_size = x.shape[0]

  # Flatten inputs for easier loss calculation
  recon_flat = reconstructed_x.view(-1, sum(state_feature_splits))
  x_flat = x.view(-1, sum(state_feature_splits))

  # Calculate loss for each feature group separately and sum them up
  # This is more stable than calculating on the concatenated tensor
  x_split = torch.split(x_flat, state_feature_splits, dim=-1)
  recon_split = torch.split(recon_flat, state_feature_splits, dim=-1)

  for recon_group, x_group in zip(recon_split, x_split):
    recon_loss += F.binary_cross_entropy_with_logits(
      recon_group, x_group, reduction='mean')

  # KL Divergence Loss
  # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

  return recon_loss + beta * kld_loss


def reconstruct_state(reconstructed_state_logits, state_feature_splits):
  """
  Convert the reconstructed logit tensor back to one-hot encoded state.
  """
  reconstructed_state = torch.zeros_like(
    reconstructed_state_logits, device=reconstructed_state_logits.device)
  start_idx = 0
  B, H, W, _ = reconstructed_state.shape
  for size in state_feature_splits:
    end_idx = start_idx + size
    # Apply softmax to the logits to get probabilities
    probs = F.softmax(
      reconstructed_state_logits[:, :, :, start_idx:end_idx], dim=-1)
    # Sample indices from the probabilities
    indices = torch.multinomial(probs.view(-1, size), num_samples=1).view(B, H, W) + start_idx
    # or alternatively, take the argmax
    # indices = torch.argmax(probs, dim=-1) + start_idx
    # Set the corresponding one-hot feature to 1
    reconstructed_state.scatter_(3, indices.unsqueeze(-1), 1.0)
    start_idx = end_idx

  return reconstructed_state


def train_vae(env, model: TransformerVAE, replay: ReplayBuffer, optimizer, num_epochs=10000, save_every_n_epochs=1000, batch_size=32, max_steps=None, logg=100):
  def collect_single_episode():
    if 1/3 < np.random.random():
      agent1 = RandomAgent(agent_id=0)
      agent2 = RandomAgent(agent_id=1)
    else:
      agent1 = SimpleAgent(agent_id=0)
      agent2 = SimpleAgent(agent_id=1)

    obs = env.reset()
    done = False
    step = 0
    ep_ret = 0.0

    while not done and (max_steps is None or step < max_steps):
      a = agent1.select_action(obs[0])
      a_opponent = agent2.select_action(obs[1])
      actions = {0: a, 1: a_opponent}
      next_obs, reward, done, info = env.step(actions)

      # store transition
      replay.push({
          "state": obs[0].copy(),
          "action": a,
          "reward": float(reward[0]),
          "next_state": next_obs[0].copy(),
          "done": bool(done),
      })

      ep_ret += reward[0]
      obs = next_obs
      step += 1
    return
  
  loss_collector = []
  epoch_collector = []
  avg_loss = 0.0
  for i in range(num_epochs):
    # Collect a new episode
    collect_single_episode()

    # fill the buffer so we can at least sample
    while (replay.__len__() < batch_size):
      collect_single_episode()

    # Sample a batch from the replay buffer
    batch = replay.sample(batch_size)
    state_batch = torch.from_numpy(
      np.array([b['state'] for b in batch])).float()  # (B, H, W, F)

    model.train()
    reconstructed_state, mu, logvar = model(state_batch)
    loss = vae_loss(reconstructed_state, state_batch, mu,
                    logvar, model.embedd.state_feature_splits, model.args.beta)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_loss += loss.item()

    if (i + 1) % logg == 0:
      loss_collector.append(loss.item())
      epoch_collector.append(i + 1)
      print(f"Epoch {i+1}/{num_epochs}, Loss: {loss.item()}, Avg Loss: {avg_loss/logg}")
      avg_loss = 0.0

    if (i + 1) % save_every_n_epochs == 0:
      torch.save(model.state_dict(),
                 f"Trained_VAE/vae_epoch_{i+1}.pth")
      print(f"Model saved at epoch {i+1}")

  loss_collector = np.array(loss_collector)
  epoch_collector = np.array(epoch_collector)

  # plt.plot(epoch_collector, loss_collector)
  # plt.xlabel('Iteration')
  # plt.ylabel('Loss')
  # plt.title('Training Loss over Time')
  # plt.show()
  return model


def generate_data(batch_size, h, w, state_feature_splits):
  """
  Generates dummy data for testing the TransformerVAE model.
  Data format: (B, H, W, F) where F is sum of state_feature_splits. Each feature group is one-hot encoded.
  """
  f_sum = sum(state_feature_splits)
  x = torch.zeros(batch_size, h, w, f_sum)

  current_f_offset = 0
  for size in state_feature_splits:
    # For each pixel, choose a random index within this feature group
    indices = torch.randint(0, size, (batch_size, h, w, 1))
    # Place 1 at that random index
    x.scatter_(3, indices + current_f_offset, 1)
    current_f_offset += size

  return x


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser(
    description="Simple test of VAE transformer.")

  parser.add_argument('--batch-size', type=int, default=4,
                      help='Batch size for training')
  parser.add_argument('--h', type=int, default=4, help='Height of the map')
  parser.add_argument('--w', type=int, default=4, help='Width of the map')
  parser.add_argument('--dataset-size', type=int, default=1024,
                      help='Size of the dataset to generate')
  parser.add_argument('--latent-dim', type=int, default=8,
                      help='Dimensionality of the latent space')
  parser.add_argument('--d-model', type=int, default=128,
                      help='Dimensionality of the transformer model')
  parser.add_argument('--nhead', type=int, default=2,
                      help='Number of attention heads')
  parser.add_argument('--num-encoder-layers', type=int,
                      default=2, help='Number of encoder layers')
  parser.add_argument('--num-decoder-layers', type=int,
                      default=2, help='Number of decoder layers')
  parser.add_argument('--dim-feedforward', type=int, default=512,
                      help='Dimensionality of the feedforward network')
  parser.add_argument('--dropout', type=float,
                      default=0.01, help='Dropout rate')
  parser.add_argument('--epochs', type=int, default=3_000,
                      help='Number of training epochs')
  parser.add_argument('--logg', type=int, default=10, help='Logging interval')
  args = parser.parse_args()

  # --- Example Usage of VAE transformer ---
  B = args.batch_size   # Batch size
  H = args.h       # Map height
  W = args.w       # Map width

  # Example: F is composed of a 3-class one-hot vector and a 4-class one-hot vector
  FEATURE_SPLITS = [3, 4]

  # --- Create Dataset ---
  from torch.utils.data import TensorDataset, DataLoader
  full_dataset_x = generate_data(args.dataset_size, H, W, FEATURE_SPLITS)
  test_data_x = generate_data(args.dataset_size//2, H, W, FEATURE_SPLITS)
  print("Generated dataset with shape:", full_dataset_x[0])
  train_dataset = TensorDataset(full_dataset_x)
  test_dataset = TensorDataset(test_data_x)
  train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=B, shuffle=True)

  print("Dataset created. Starting training...")
  om_args = OMGArgs(
      H=H, W=W, state_feature_splits=FEATURE_SPLITS,
      latent_dim=args.latent_dim,
      d_model=args.d_model,
      nhead=args.nhead,
      num_encoder_layers=args.num_encoder_layers,
      num_decoder_layers=args.num_decoder_layers,
      dim_feedforward=args.dim_feedforward,
      dropout=args.dropout
  )
  # --- Model Init ---
  model = TransformerVAE(
      om_args
  )
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

  loss_collector = []
  x_collector = []
  # --- Train Loop ---
  epochs = args.epochs
  logg = args.logg
  for epoch in range(epochs):
    for batch in train_loader:
      x = batch[0]  # Get the input state tensor (B, H, W, F)
      model.train()

      # --- Forward Pass and Loss Calculation ---
      reconstructed_state, mu, logvar = model(x)
      loss = vae_loss(reconstructed_state, x, mu, logvar, FEATURE_SPLITS, model.args.beta)

      # --- Backward Pass ---
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    # --- Logging every 'logg' epochs ---
    if (epoch + 1) % logg == 0:
      loss_collector.append(loss.item())
      x_collector.append(epoch + 1)
      print(f"Epoch {epoch+1}/{epochs}")
      print("Input state shape:", x.shape)
      print("Reconstructed state shape:", reconstructed_state.shape)
      print("Mu shape:", mu.shape)
      print("Logvar shape:", logvar.shape)
      print("Calculated Loss:", loss.item())

  # --- Test the model ---
  model.eval()
  with torch.no_grad():
    for batch in test_loader:
      x = batch[0]
      reconstructed_state, mu, logvar = model(x)
      test_loss = vae_loss(reconstructed_state, x, mu, logvar, FEATURE_SPLITS, model.args.beta)
      print("Test Loss:", test_loss.item())
      break  # Just test on one batch for brevity
  print("Training complete.")

  # --- Plotting the loss curve ---

  loss_collector = np.array(loss_collector)
  x_collector = np.array(x_collector)

  plt.plot(x_collector, loss_collector)
  plt.xlabel('Iteration')
  plt.ylabel('Loss')
  plt.title('Training Loss over Time')
  plt.show()
