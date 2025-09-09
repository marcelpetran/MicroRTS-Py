import re
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as torch_f
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """
    Standard positional encoding as used in the original Transformer paper.
    Adds sine and cosine functions of different frequencies to the input embeddings.
    """
    def __init__(self, d_model: int , seq_len: int, dropout: float):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create the positional encoding matrix (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
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
        x = x + (self.pe[:, :x.size(1), :]).requires_grad_(False) # We don't want to train the positional encodings
        return self.dropout(x)

class StateEmbeddings(nn.Module):
    """
    This module handles the embedding of the complex, multi-feature game state.
    It takes the (B, H, W, F) tensor, flattens it, embeds each feature group,
    sums them, and adds positional encoding.
    """
    def __init__(self, h, w, feature_split_sizes, d_model, dropout):
        super().__init__()
        self.h = h
        self.w = w
        self.seq_len = h * w
        self.feature_split_sizes = feature_split_sizes
        self.d_model = d_model
        self.dropout = dropout

        # Create a separate linear layer for each one-hot feature group
        self.feature_embedders = nn.ModuleList([
            nn.Linear(size, d_model, bias=False) for size in feature_split_sizes
        ])

        # Positional encoding
        self.position_encoder = PositionalEncoding(d_model, seq_len=self.seq_len * 2, dropout=self.dropout)

    def forward(self, state_tensor):
        """
        Args:
            state_tensor (Tensor): Input state of shape (B, H, W, F)
        Returns:
            Tensor: Embedded state of shape (B, H*W, d_model)
        """
        B = state_tensor.shape[0]
        
        # Flatten spatial dimensions: (B, H, W, F) -> (B, H*W, F)
        state_flat = state_tensor.view(B, self.seq_len, -1)
        
        # Split the features along the last dimension
        split_features = torch.split(state_flat, self.feature_split_sizes, dim=-1)
        
        # Embed each feature group and sum them up.
        # We initialize with zeros and add each embedding.
        embedded = torch.zeros(B, self.seq_len, self.d_model, device=state_tensor.device)
        for i, feature_tensor in enumerate(split_features):
            embedded += self.feature_embedders[i](feature_tensor)

        # Add positional encoding
        return self.position_encoder(embedded)
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
    def __init__(self, h, w, state_feature_splits, action_feature_splits , latent_dim, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.seq_len = h * w
        self.droput = dropout

        # --- Feature Embedding ---
        self.state_embedder = StateEmbeddings(h, w, state_feature_splits, d_model, dropout)
        self.action_embedder = ActionEmbeddings(h, w, action_feature_splits, d_model, dropout)

        # --- Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

        # --- Decoder ---
        self.latent_to_decoder_input = nn.Linear(latent_dim, self.seq_len * d_model)
        self.decoder_pos_encoder = PositionalEncoding(d_model, seq_len=self.seq_len, dropout=self.droput)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # --- Output Projection ---
        self.output_projectors = nn.ModuleList([
            nn.Linear(d_model, size) for size in state_feature_splits
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
        Args:
            history (dict): A dictionary containing historical states and actions.
                            Expected keys: 'states', 'actions', 'opp_actions' (if applicable)
        Returns:
            List[Tensor]: A list of embedded tensors forming the history sequence.
        """
        history_embeddings = []
        for i in range(len(history['states'])):
            s_t = history['states'][i]
            a_t = history['actions'][i]
            # A_t = history['opp_actions'][i]

            # Embed state and action and add to our history sequence
            s_t_embedded = self.state_embedder(s_t)
            a_t_embedded = self.action_embedder(a_t)
            history_embeddings.extend([s_t_embedded, a_t_embedded])
        return history_embeddings

    def encode(self, x, history):
        """
        Encodes the input state x conditioned on the historical trajectory.
        Args:
            x (Tensor): Input state of shape (B, H, W, F)
            history (dict): A dictionary containing historical states and actions.
                            Expected keys: 'states', 'actions', 'opp_actions' (if applicable)
        Returns:
            Tensor: Mean of the latent distribution (B, latent_dim)
            Tensor: Log-variance of the latent distribution (B, latent_dim)
        """
        x_embedded = self.state_embedder(x)

        # Process the historical trajectory to form the condition
        history_embeddings = self.get_history_embeddings(history)
        
        # Concatenate all history elements into a single long sequence
        condition_seq = torch.cat(history_embeddings, dim=1)
        
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

    def decode(self, z, history_seq):
        """
        Decodes a latent vector z conditioned on an embedded history sequence.
        
        Args:
            z (Tensor): The latent vector. Shape: (B, latent_dim)
            history_seq (Tensor): The pre-embedded and concatenated history sequence
                                  that acts as memory for the decoder.
        Returns:
            Tensor: Reconstructed state of shape (B, H, W, F)
        """
        B = z.shape[0]
        memory = history_seq
        
        decoder_input = self.latent_to_decoder_input(z).view(B, self.seq_len, -1)
        tgt = self.decoder_pos_encoder(decoder_input)
        
        decoder_output = self.transformer_decoder(tgt=tgt, memory=memory)
        
        reconstructed_features = [proj(decoder_output) for proj in self.output_projectors]
        reconstructed_x_flat = torch.cat(reconstructed_features, dim=-1)
        reconstructed_x = reconstructed_x_flat.view(B, self.state_embedder.h, self.state_embedder.w, -1)
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
        history_seq = torch.cat(self.get_history_embeddings(history), dim=1)
        mu, logvar = self.encode(x, history)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode(z, history_seq)
        return reconstructed_x, mu, logvar

class TransformerVAE(nn.Module):
    def __init__(self, h, w, feature_split_sizes, latent_dim, d_model=256, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.seq_len = h * w
        self.dropout = dropout

        # --- Feature Embedding ---
        self.embedd = StateEmbeddings(h, w, feature_split_sizes, d_model, dropout)

        # --- Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

        # --- Decoder ---
        self.latent_to_decoder_input = nn.Linear(latent_dim, self.seq_len * d_model)
        self.decoder_pos_encoder = PositionalEncoding(d_model, seq_len=self.seq_len, dropout=self.dropout)
        # In a VAE, we don't have a memory sequence to attend to
        # we only have the latent vector z to reconstruct from
        # TransformerEncoder is simply a stack of self-attention blocks
        # which is exactly what we need to reconstruct a sequence from a starting point
        decoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerEncoder(decoder_layer, num_decoder_layers)
        
        # --- Output Projection ---
        self.output_projectors = nn.ModuleList([
            nn.Linear(d_model, size) for size in feature_split_sizes
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
        
        reconstructed_features = [proj(decoder_output) for proj in self.output_projectors]
        reconstructed_x_flat = torch.cat(reconstructed_features, dim=-1)
        reconstructed_x = reconstructed_x_flat.view(B, self.embedd.h, self.embedd.w, -1)
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
    
def simple_cvae_loss(predicted_mu, target_mu):
    """
    Very simple loss function to train encoder to match the latent distribution of a pre-trained VAE.
    Uses Mean Squared Error between the predicted and target means.
    We do not need to use decoder, or we simply copy the weights from the pre-trained VAE.
    Not usable for OM! -> We also need KL
    Args:
        predicted_mu (Tensor): Predicted mean from the CVAE encoder. Shape: (B, latent_dim)
        target_mu (Tensor): Target mean from the pre-trained VAE encoder. Shape: (B, latent_dim)
    Returns:
        Tensor: Computed MSE loss.
    """
    return torch_f.mse_loss(predicted_mu, target_mu)
    
# Loss function for the VAE
def vae_loss(reconstructed_x, x, mu, logvar, feature_split_sizes):
    """
    Loss function for VAE combining reconstruction loss and KL divergence.
    Args:
        reconstructed_x (Tensor): Reconstructed state of shape (B, H, W, F)
        x (Tensor): Original input state of shape (B, H, W, F)
        mu (Tensor): Mean of the latent distribution (B, latent_dim)
        logvar (Tensor): Log-variance of the latent distribution (B, latent_dim)
        feature_split_sizes (List[int]): List of sizes for each one-hot feature group.
    Returns:
        Tensor: Computed VAE loss.
    """
    # Reconstruction Loss
    # We use Binary Cross Entropy with Logits for each one-hot feature group
    # as it's suitable for multi-label classification style outputs
    recon_loss = 0
    
    # Flatten inputs for easier loss calculation
    recon_flat = reconstructed_x.view(-1, sum(feature_split_sizes))
    x_flat = x.view(-1, sum(feature_split_sizes))
    
    # Calculate loss for each feature group separately and sum them up
    # This is more stable than calculating on the concatenated tensor
    x_split = torch.split(x_flat, feature_split_sizes, dim=-1)
    recon_split = torch.split(recon_flat, feature_split_sizes, dim=-1)
    
    for recon_group, x_group in zip(recon_split, x_split):
        recon_loss += torch_f.binary_cross_entropy_with_logits(recon_group, x_group, reduction='sum')

    # KL Divergence Loss
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kld_loss

def reconstruct_state(reconstructed_state_logits, feature_split_sizes):
    """
    Convert the reconstructed logit tensor back to one-hot encoded state.
    """
    reconstructed_state = torch.zeros_like(reconstructed_state_logits, device=reconstructed_state_logits.device)
    start_idx = 0
    for size in feature_split_sizes:
        end_idx = start_idx + size
        # Apply softmax to the logits to get probabilities
        probs = torch_f.softmax(reconstructed_state_logits[:, :, :, start_idx:end_idx], dim=-1)
        # Sample indices from the probabilities
        # indices = torch.multinomial(probs.view(-1, size), num_samples=1).view(B, h, w) + start_idx
        # or alternatively, take the argmax
        indices = torch.argmax(probs, dim=-1) + start_idx
        # Set the corresponding one-hot feature to 1
        reconstructed_state.scatter_(3, indices.unsqueeze(-1), 1.0)
        start_idx = end_idx
    
    return reconstructed_state

# experimental data generation for testing CVAE
def generate_conditional_data(batch_size, h, w):
    """
    Generates a batch of (target, condition) pairs with a learnable rule.
    - Condition 'c' has one "goal" pixel.
    - Target 'x' has one "player" pixel near the goal.
    """
    # For simplicity, we'll use a single feature plane (F=1)
    # The concept extends to your multi-feature setup
    
    # --- Create the Condition 'c' ---
    c = torch.zeros(batch_size, h, w, 1)
    # Pick a random "goal" location for each item in the batch
    goal_coords_h = torch.randint(0, h, (batch_size,))
    goal_coords_w = torch.randint(0, w, (batch_size,))
    # Place the "goal" pixel
    c[torch.arange(batch_size), goal_coords_h, goal_coords_w, 0] = 1

    # --- Create the Target 'x' ---
    x = torch.zeros(batch_size, h, w, 1)
    # Player position is a small random offset from the goal
    offset_h = torch.randint(-1, 2, (batch_size,)) # Offset of -1, 0, or 1
    offset_w = torch.randint(-1, 2, (batch_size,))
    
    # Calculate player coordinates and clamp them to stay within the grid
    player_coords_h = torch.clamp(goal_coords_h + offset_h, 0, h - 1)
    player_coords_w = torch.clamp(goal_coords_w + offset_w, 0, w - 1)
    
    # Place the "player" pixel
    x[torch.arange(batch_size), player_coords_h, player_coords_w, 0] = 1
    
    return x, c

def generate_data(batch_size, h, w, feature_split_sizes):
    """
    Generates dummy data for testing the TransformerVAE model.
    Data format: (B, H, W, F) where F is sum of feature_split_sizes. Each feature group is one-hot encoded.
    """
    F = sum(feature_split_sizes)
    x = torch.zeros(batch_size, h, w, F)
    
    current_f_offset = 0
    for size in feature_split_sizes:
        # For each pixel, choose a random index within this feature group
        indices = torch.randint(0, size, (batch_size, h, w, 1))
        # Place 1 at that random index
        x.scatter_(3, indices + current_f_offset, 1)
        current_f_offset += size
        
    return x

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Simple test of VAE transformer.")

    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--h', type=int, default=4, help='Height of the map')
    parser.add_argument('--w', type=int, default=4, help='Width of the map')
    parser.add_argument('--dataset-size', type=int, default=32, help='Size of the dataset to generate')
    parser.add_argument('--latent-dim', type=int, default=8, help='Dimensionality of the latent space')
    parser.add_argument('--d-model', type=int, default=32, help='Dimensionality of the transformer model')
    parser.add_argument('--nhead', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--num-encoder-layers', type=int, default=1, help='Number of encoder layers')
    parser.add_argument('--num-decoder-layers', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--dim-feedforward', type=int, default=64, help='Dimensionality of the feedforward network')
    parser.add_argument('--dropout', type=float, default=0.01, help='Dropout rate')
    parser.add_argument('--epochs', type=int, default=30000, help='Number of training epochs')
    parser.add_argument('--logg', type=int, default=100, help='Logging interval')
    args = parser.parse_args()

    # --- Example Usage of VAE transformer ---
    B = args.batch_size   # Batch size
    H = args.h       # Map height
    W = args.w       # Map width
    
    # Example: F is composed of a 3-class one-hot vector and a 4-class one-hot vector
    FEATURE_SPLITS = [3, 4] 
    F = sum(FEATURE_SPLITS)
    
    # --- Create Dataset ---
    from torch.utils.data import TensorDataset, DataLoader
    full_dataset_x = generate_data(args.dataset_size, H, W, FEATURE_SPLITS)
    print("Generated dataset with shape:", full_dataset_x[0])
    train_dataset = TensorDataset(full_dataset_x)
    train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True)
    
    print("Dataset created. Starting training...")

    # --- Model Init ---
    model = TransformerVAE(
        h=H,
        w=W,
        feature_split_sizes=FEATURE_SPLITS,
        latent_dim=args.latent_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
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
        loss = vae_loss(reconstructed_state, x, mu, logvar, FEATURE_SPLITS)

        # --- Backward Pass ---
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

      # --- Logging every 'logg' epochs ---
      if (epoch + 1) % logg == 0:
        loss_collector.append(loss.item())
        x_collector.append(epoch+1)
        print(f"Epoch {epoch+1}/{epochs}")
        print("Input state shape:", x.shape)
        print("Reconstructed state shape:", reconstructed_state.shape)
        print("Mu shape:", mu.shape)
        print("Logvar shape:", logvar.shape)
        print("Calculated Loss:", loss.item())
    
    # --- Test the model ---
    model.eval()
    with torch.no_grad():
        test_x = generate_data(B, H, W, FEATURE_SPLITS)
        reconstructed_x_logits, mu, logvar = model(test_x)
        reconstructed_state = reconstruct_state(reconstructed_x_logits, FEATURE_SPLITS)
        print("Test Input State:\n", test_x[0])
        print("Reconstructed State:\n", reconstructed_state[0])
        print("Reconstructed logits:\n", reconstructed_x_logits[0])
        print("Mu:", mu)
        print("Logvar:", logvar)
        print("Reconstruction Loss:", vae_loss(reconstructed_x_logits, test_x, mu, logvar, FEATURE_SPLITS).item())
    print("Training complete.")

    # --- Plotting the loss curve ---
    from matplotlib import pyplot as plt

    loss_collector = np.array(loss_collector)
    x_collector = np.array(x_collector)

    plt.plot(x_collector, loss_collector)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss over Time')
    plt.show()