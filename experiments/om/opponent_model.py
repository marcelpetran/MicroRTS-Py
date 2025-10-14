import transformers as t
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class SubGoalSelector:
  def __init__(self, args):
    self.args = args

  def gumbel_max_sample(self, logits):
    """
    Samples from a categorical distribution using the Gumbel-max trick.
    Args:
        logits (Tensor): Logits of shape (N,) representing the unnormalized log probabilities.
    Returns:
        int: Index of the sampled category.
    """
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    if self.args.selector_mode == "optimistic":
      return torch.argmax(logits + gumbel_noise)
    elif self.args.selector_mode == "conservative":
      return torch.argmin(logits + gumbel_noise)
    else:
      raise ValueError(
        f"Unknown selector_mode: {self.args.selector_mode},\nchoose from ['optimistic', 'conservative']")

  def select(self, vae, eval_policy, s_t: torch.Tensor, future_states: torch.Tensor):
    """
    Selects a subgoal by encoding future states to latent space s_g = encode(s_t+K), 
    selecting suitable future state by policy argmin Q-value(s_t, s_g), 
    and encoding choosen state to latent space
    Args:
        vae (VAE): Pre-trained VAE model
        policy (Policy): Policy model with Q-value function
        s_t (Tensor): Current state of shape (1, H, W, F)
        future_states (Tensor): Future states of shape (B, K, H, W, F) - K is the horizon
    Returns:
        Tensor: latent subgoal of shape (latent_dim)
    """
    vae.eval()
    with torch.no_grad():
      subgoal_batch = []
      vae_log_var_batch = []
      # iterate over batch
      for i, horizon in enumerate(future_states):
        # (B, K, H, W, F) -> (K, H, W, F)
        mu, _ = vae.encode(horizon)  # (K, latent_dim)
        # policy expects a batch dim for state and subgoal
        s_t_batch = s_t if s_t.dim() == 4 else s_t.unsqueeze(0)  # (1, H, W, F)
        mu_batch = mu.unsqueeze(0)  # (1, K, latent_dim)

        # s_t: (1, D_s), mu: (K, D_g) -> Q(K, A)
        # print(s_t.shape, future_states.shape) # (1, H, W, F), (K, H, W, F)
        s_t_expanded = s_t_batch[i].repeat(
          1, mu.shape[0], 1, 1, 1).view(-1, *s_t.shape[1:])
        mu_expanded = mu_batch.view(-1, mu.shape[-1])

        values_flat = eval_policy.value(s_t_expanded, mu_expanded)  # (K, A)
        values = values_flat.mean(dim=-1)  # V(s,g) = mean_a Q(s,g,a)

        # Gumbel-max trick for differentiable argmin
        best_idx = self.gumbel_max_sample(values)
        best_future_state = horizon[best_idx].unsqueeze(0)  # (1, H, W, F)
        # (1, latent_dim), (1, latent_dim)
        subgoal, vae_log_var = vae.encode(best_future_state)
        subgoal_batch.append(subgoal)
        vae_log_var_batch.append(vae_log_var)

      # concat all subgoals and log_vars back to batch dimension
      subgoal = torch.cat(subgoal_batch, dim=0)
      vae_log_var = torch.cat(vae_log_var_batch, dim=0)
      # (B, latent_dim), (B, latent_dim)
    return subgoal, vae_log_var


class OpponentModel(nn.Module):
  def __init__(self, cvae: t.TransformerCVAE, vae: t.TransformerVAE, selector: SubGoalSelector, optimizer, device, args):
    super(OpponentModel, self).__init__()
    self.inference_model = cvae
    self.prior_model = vae  # pre-trained VAE
    self.subgoal_selector = selector
    self.optimizer = optimizer
    self.device = device
    self.args = args
    self.mse_loss = nn.MSELoss()

    # Precompute feature weights for reconstruction loss
    splits = self.args.state_feature_splits
    weights = []
    for s in splits:
      weights.append(torch.full((s,), 1.0 / s))
    w = torch.cat(weights)
    w = w / len(splits)
    self.register_buffer('feature_weights', w.to(self.device))  # (F,)

  def eval(self):
    """
    Set inference model to evaluation mode
    """
    self.inference_model.eval()
    self.prior_model.eval()
    return

  def reconstruct_state(self, reconstructed_state_logits, state_feature_splits=None):
    """
    Convert the reconstructed logit tensor back to one-hot encoded state.
    """
    if state_feature_splits is None:
      state_feature_splits = self.args.state_feature_splits

    if reconstructed_state_logits.dim() == 3:
      # probably got (H, W, F), add batch dim
      reconstructed_state_logits = reconstructed_state_logits.unsqueeze(0)

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
      indices = torch.multinomial(
        probs.view(-1, size), num_samples=1).view(B, H, W) + start_idx
      # or alternatively, take the argmax
      # indices = torch.argmax(probs, dim=-1) + start_idx
      # Set the corresponding one-hot feature to 1
      reconstructed_state.scatter_(3, indices.unsqueeze(-1), 1.0)
      start_idx = end_idx

    return reconstructed_state

  def visualize_subgoal(self, ghat_mu: torch.Tensor, filename: str = "./subgoal_visualization.png"):
    """
    Reconstructs visualization of a subgoal state from a latent vector.

    Args:
        ghat_mu (torch.Tensor): The mean of the inferred latent distribution (ĝt), shape (B, latent_dim).
    """
    print(f"Visualizing subgoal and saving to {filename}...")
    self.prior_model.eval()  # Use the frozen, pre-trained VAE for reconstruction

    with torch.no_grad():
      reconstructed_logits = self.prior_model.decode(ghat_mu)
      reconstructed_state_one_hot = self.reconstruct_state(
          reconstructed_logits, self.args.state_feature_splits
      )
    # Shape: (H, W, F)
    state_to_plot = reconstructed_state_one_hot[0].cpu().numpy()

    _plot_foraging_grid(state_to_plot, filename)

    return

  def visualize_subgoal_logits(self, obs: np.ndarray, reconstructed_logits: torch.Tensor, state_feature_splits: tuple, filename: str = "subgoal_logits.png"):
    """
    Visualizes the softmax probabilities of the reconstructed subgoal logits.
    """
    assert reconstructed_logits.dim() == 4, "Expected logits to be 4D (B, H, W, F)"
    logits = reconstructed_logits[0].detach().cpu()

    # Apply softmax to get probabilities
    probs = torch.zeros_like(logits)
    start_idx = 0
    for size in state_feature_splits:
      end_idx = start_idx + size
      probs[:, :, start_idx:end_idx] = F.softmax(
        logits[:, :, start_idx:end_idx], dim=-1)
      start_idx = end_idx

    probs = probs.numpy()  # (H, W, F)
    labels = {0: 'Empty', 1: 'Food',
              2: 'Agent 1 (Self)', 3: 'Agent 2 (Opponent)'}

    H, W, F_dim = probs.shape

    fig = plt.figure(figsize=(F_dim * 4 + 4, 8))
    gs = fig.add_gridspec(2, F_dim)

    # Plot the actual current state
    ax_obs = fig.add_subplot(gs[0, 1])

    obs_labels = np.argmax(obs, axis=-1)

    cmap_obs = plt.get_cmap('viridis', F_dim)
    im = ax_obs.imshow(obs_labels, cmap=cmap_obs, vmin=0, vmax=F_dim - 1)
    fig.colorbar(im, ax=ax_obs, ticks=np.arange(len(labels)))
    cbar = im.colorbar
    cbar.ax.set_yticklabels([labels[i] for i in range(len(labels))])

    ax_obs.set_title("Current obs (s_t)")
    ax_obs.set_xticks(np.arange(W))
    ax_obs.set_yticks(np.arange(H))

    for i in range(F_dim):
      ax = fig.add_subplot(gs[1, i])
      im = ax.imshow(probs[:, :, i], cmap='hot', vmin=0, vmax=1)
      ax.set_title(f"P({labels[i]})")
      fig.colorbar(im, ax=ax)
      ax.set_xticks(np.arange(W))
      ax.set_yticks(np.arange(H))

    fig.suptitle("Current obs and Inferred Subgoal Probabilities", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(filename)
    plt.close()

  def loss_function(
          self, reconstructed_x, x, cvae_mu, cvae_log_var,
          vae_mu, vae_log_var, infer_mu, infer_log_var, dones, eta):
    """
    Calculates the full OMG loss for the CVAE.
    """
    mask = (1.0 - dones.float())
    denom = mask.sum().clamp_min(1.0)
    # --- 1. Reconstruction Loss ---
    # This forces the CVAE to represent the current state
    recon_loss = 0

    # recon_flat = reconstructed_x.view(-1, sum(self.args.state_feature_splits))
    # x_flat = x.view(-1, sum(self.args.state_feature_splits))

    # x_split = torch.split(x_flat, self.args.state_feature_splits, dim=-1)
    # recon_split = torch.split(
    #   recon_flat, self.args.state_feature_splits, dim=-1)

    # for recon_group, x_group in zip(recon_split, x_split):
    #   bce_loss = F.binary_cross_entropy_with_logits(recon_group, x_group, reduction='none').mean(dim=1)
    #   print(bce_loss.shape, mask.shape)  # (B*H*W,), (B,1)
    #   recon_loss += (bce_loss * mask).sum() / mask.sum()

    bce = F.binary_cross_entropy_with_logits(
      reconstructed_x, x, reduction='none')  # (B, H, W, F)
    per_cell = (bce * self.feature_weights).sum(dim=-1)
    per_ex = per_cell.mean(dim=(1, 2))
    recon_loss = (per_ex * mask).sum() / denom

    # --- 2. Inference Loss ---
    # This forces the CVAE latent vector g_hat to point towards high-probability future
    if eta < np.random.random():  # eta goes to 0 over time (eq. (8) in the paper)
      # Use the model's own inference (g_hat) as the target
      target_mu, target_log_var = infer_mu.detach(), infer_log_var.detach()
    else:
      # Use the pre-trained VAE's output (g_bar) as the target
      target_mu, target_log_var = vae_mu.detach(), vae_log_var.detach()

    # KL Divergence between CVAE's prediction and the target distribution
    kl_div_per_example = -0.5 * torch.sum(1 + cvae_log_var - target_log_var -
                                          (((target_mu - cvae_mu) ** 2 +
                                            cvae_log_var.exp()) / target_log_var.exp()),
                                          dim=-1)

    omg_loss = (kl_div_per_example * mask).sum() / denom

    # --- 3. Total Loss ---
    total_loss = recon_loss + self.args.beta * omg_loss
    return total_loss

  def train_step(self, batch, eval_policy):
    """
    Performs a single training step for the opponent model.
    Args:
        batch (dict): A batch of data from the replay buffer containing:
            - 'states': Tensor of shape (B, H, W, F)
            - 'history': Dict of lists of Tensors for historical data
            - 'future_states': Tensor of shape (B, K, H, W, F)
            - 'infer_mu': Tensor of shape (B, latent_dim)
            - 'infer_log_var': Tensor of shape (B, latent_dim)
        eval_policy (Policy): The current evaluation policy used for subgoal selection.
    Returns:
        float: The computed loss for the batch.
    """
    # Unpack the batch from the replay buffer
    x = batch['states'].to(self.device)  # (B, H, W, F)
    history = batch['history']  # A dict of state/action lists up to s_t-1
    future_states = batch['future_states'].to(self.device)  # (B, K, H, W, F)
    infer_mu = batch['infer_mu'].to(self.device)
    infer_log_var = batch['infer_log_var'].to(self.device)
    dones = batch['dones'].to(self.device)

    assert x.dim(
    ) == 4, f"Expected states to be 4D (B, H, W, F), got {x.shape}"
    assert future_states.dim(
    ) == 5, f"Expected future_states to be 5D (B, K, H, W, F), got {future_states.shape}"

    self.inference_model.train()
    self.optimizer.zero_grad()
    # transformers.py, line 321, in forward
    #   mu, logvar = self.encode(x, history)
    # File transformers.py, line 265, in encode
    #   combined_seq = torch.cat([x_embedded, condition_seq], dim=1)
    # Sizes of tensors must match except in dimension 1. Expected size 4 but got size 1 for tensor number 1 in the list.
    # this happens when history is empty
    # TODO: fix this properly
    reconstructed_x, cvae_mu, cvae_log_var = self.inference_model(x, history)

    with torch.no_grad():
      vae_mu, vae_log_var = self.subgoal_selector.select(self.prior_model, eval_policy,
                                                         x, future_states)

    loss = self.loss_function(reconstructed_x, x, cvae_mu, cvae_log_var,
                              vae_mu, vae_log_var, infer_mu, infer_log_var, dones, eval_policy._gmix_eps())

    loss.backward()
    self.optimizer.step()

    return loss.item()

# Helper function to plot the foraging grid


def _plot_foraging_grid(grid: np.ndarray, filename: str):
  """
  Creates a plot of a single (H, W, F) foraging state.
  """

  # Convert the one-hot grid to a grid of integer labels for coloring
  # e.g., empty=0, food=1, agent1=2, agent2=3
  grid_labels = np.argmax(grid, axis=-1)

  # Define colors and labels for the plot
  # Make sure this matches the feature order in your environment!
  cmap = plt.get_cmap('viridis', 4)
  labels = {0: 'Empty', 1: 'Food',
            2: 'Agent 1 (Self)', 3: 'Agent 2 (Opponent)'}

  fig, ax = plt.subplots()
  mat = ax.matshow(grid_labels, cmap=cmap)

  # Create a color bar with labels
  cbar = plt.colorbar(mat, ticks=np.arange(len(labels)))
  cbar.ax.set_yticklabels([labels[i] for i in range(len(labels))])

  plt.title("Reconstructed Subgoal State")
  plt.savefig(filename)
  plt.close()
