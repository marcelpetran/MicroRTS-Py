import transformers as t
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from simple_foraging_env import RandomAgent, SimpleAgent
from q_agent import ReplayBuffer


class SubGoalSelector:
  def __init__(self, args):
    self.args = args

  def gumbel_sample(self, logits, beta):
    """
    Samples from a categorical distribution using the Gumbel-max/min trick.
    Args:
        logits (Tensor): Logits of shape (K,) representing the unnormalized log probabilities.
    Returns:
        int: Index of the sampled category.
    """
    # gumbel_noise = np.random.gumbel(0, beta, logits.shape)
    # inverse CDF method X = mu - beta * log(-log(U)) -> mu = 0, U ~ Uniform(0,1) -> X = - beta * log(U)
    gumbel_noise = -beta * torch.empty_like(logits).exponential_().log()
    if self.args.selector_mode == "optimistic":
      return torch.argmax(logits + gumbel_noise)
    elif self.args.selector_mode == "conservative":
      return torch.argmin(logits - gumbel_noise)
    else:
      raise ValueError(
        f"Unknown selector_mode: {self.args.selector_mode},\nchoose from ['optimistic', 'conservative']")

  def select(self, vae, eval_policy, s_t_batch: torch.Tensor, future_states: torch.Tensor, tau: float):
    """
    Selects a subgoal by encoding future states to latent space s_g = encode(s_t+K), 
    selecting suitable future state by policy argmin Q-value(s_t, s_g), 
    and encoding choosen state to latent space
    Args:
        vae (VAE): Pre-trained VAE model
        policy (Policy): Policy model with Q-value function
        s_t_batch (Tensor): Current state of shape (B, H, W, F)
        future_states (Tensor): Future states of shape (B, K, H, W, F) - K is the horizon
    Returns:
        Tensor: latent subgoal of shape (latent_dim)
    """
    vae.eval()
    with torch.no_grad():
      subgoal_batch = []
      vae_log_var_batch = []
      assert s_t_batch.dim(
      ) == 4, f"s_t_ba should be 4D (B, H, W, F), got {s_t_batch.shape}"
      assert future_states.dim(
      ) == 5, f"future_states should be 5D (B, K, H, W, F), got {future_states.shape}"

      # iterate over batch
      for i, horizon in enumerate(future_states):
        # (B, K, H, W, F) -> (K, H, W, F)
        mu, _ = vae.encode(horizon)  # (K, latent_dim)
        # policy expects a batch dim for state and subgoal
        s_t = s_t_batch[i]
        K = horizon.shape[0]

        # s_t: (D_s) -> (K, D_s)
        s_t_expanded = s_t.unsqueeze(0).repeat(K, 1, 1, 1)  # (K, H, W, F)

        # s_t_expanded: (K, H, W, F), mu: (K, latent_dim) -> values_flat: (K, A)
        values_flat = eval_policy.value(s_t_expanded, mu)  # (K, A)

        # old approach: take max Q-value over actions
        # values = values_flat.max(dim=-1)

        # Boltzmann exploration over Q-values
        probs = F.softmax(-values_flat / tau, dim=-1)  # (K, A)
        values = (probs * values_flat).sum(dim=-1)  # (K,) Expected Q-value

        # Gumbel-max trick for differentiable argmin or argmax
        best_idx = self.gumbel_sample(values, tau)
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
    # Weights for each one-hot feature
    self.register_buffer('feature_split_weights', w.to(self.device))
    # Weights for each feature type
    self.register_buffer('feature_weights', torch.tensor(
      [1.0, 10.0, 20.0, 20.0], device=self.device))

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

  def visualize_subgoal_logits(self, obs: np.ndarray, reconstructed_logits: torch.Tensor, filename: str = "subgoal_logits.png"):
    """
    Visualizes the softmax probabilities of the reconstructed subgoal logits.
    """
    assert reconstructed_logits.dim() == 4, "Expected logits to be 4D (B, H, W, F)"
    logits = reconstructed_logits[0].detach().cpu()

    # Apply softmax to get probabilities
    probs = torch.zeros_like(logits)
    start_idx = 0
    for size in self.args.state_feature_splits:
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

  def visualize_selected_subgoal(self, gbar_mu: torch.Tensor, original_obs: np.ndarray, filename: str = "selected_subgoal.png"):
    """
    Reconstructs and visualizes the state corresponding to the selected subgoal (g_bar).
    """
    self.prior_model.eval()
    with torch.no_grad():
      reconstructed_logits = self.prior_model.decode(gbar_mu)

    self.visualize_subgoal_logits(
      original_obs, reconstructed_logits, filename)

  # Loss function for the VAE

  def vae_loss(self, reconstructed_x, x, mu, logvar):
    """
    Loss function for VAE combining reconstruction loss and KL divergence.
    Args:
        reconstructed_x (Tensor): Reconstructed state of shape (B, H, W, F)
        x (Tensor): Original input state of shape (B, H, W, F)
        mu (Tensor): Mean of the latent distribution (B, latent_dim)
        logvar (Tensor): Log-variance of the latent distribution (B, latent_dim)
    Returns:
        Tensor: Computed VAE loss.
    """
    recon_loss = 0
    weight_mask = torch.full_like(x, self.feature_weights[0], device=self.device)
    food_mask = (x[..., 1] == 1)
    agent1_mask = (x[..., 2] == 1)
    agent2_mask = (x[..., 3] == 1)

    weight_mask[..., 1][food_mask] = self.feature_weights[1] # Weight for food
    weight_mask[..., 2][agent1_mask] = self.feature_weights[2] # Weight for agent 1
    weight_mask[..., 3][agent2_mask] = self.feature_weights[3] # Weight for agent 2

    # weight_mask = x * self.feature_weights
    # weight_mask = weight_mask + 1.0

    bce = F.binary_cross_entropy_with_logits(
      reconstructed_x, x, weight=weight_mask, reduction='none')  # (B, H, W, F)
    per_cell = (bce * self.feature_split_weights).sum(dim=-1)
    recon_loss = per_cell.mean(dim=(1, 2))
    # per_cell = bce.mean(dim=(1, 2, 3))
    # recon_loss = per_cell.mean()

    # KL Divergence Loss
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss.mean() + self.args.vae_beta * kld_loss

  def train_vae(self,
                env,
                replay: ReplayBuffer,
                optimizer,
                num_epochs=10000,
                save_every_n_epochs=1000,
                max_steps=None,
                logg=100
                ):
    def collect_single_episode():
      if 1 / 3 < np.random.random():
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
        replay.push(
            {
                "state": obs[0].copy(),
                "action": a,
                "reward": float(reward[0]),
                "next_state": next_obs[0].copy(),
                "done": bool(done),
            }
        )

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
      while replay.__len__() < self.args.batch_size:
        collect_single_episode()

      # Sample a batch from the replay buffer
      batch = replay.sample(self.args.batch_size)
      state_batch = torch.from_numpy(
          np.array([b["state"] for b in batch])
      ).float()  # (B, H, W, F)
      state_batch = state_batch.to(self.args.device)
      self.prior_model.train()
      reconstructed_state, mu, logvar = self.prior_model(state_batch)
      loss = self.vae_loss(
          reconstructed_state,
          state_batch,
          mu,
          logvar
      )

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      avg_loss += loss.item()

      if (i + 1) % logg == 0:
        loss_collector.append(loss.item())
        epoch_collector.append(i + 1)
        print(
            f"Epoch {i + 1}/{num_epochs}, Loss: {loss.item()}, Avg Loss: {avg_loss / logg}"
        )
        avg_loss = 0.0

      if (i + 1) % save_every_n_epochs == 0:
        torch.save(self.prior_model.state_dict(),
                   f"./trained_vae_{self.args.folder_id}/vae_epoch_{i + 1}.pth")
        print(f"Model saved at epoch {i + 1}")

    loss_collector = np.array(loss_collector)
    epoch_collector = np.array(epoch_collector)

    # plt.plot(epoch_collector, loss_collector)
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.title('Training Loss over Time')
    # plt.show()

  def loss_function(
          self, reconstructed_x, x, cvae_mu, cvae_log_var,
          vae_mu, vae_log_var, infer_mu, infer_log_var, dones, eta, beta):
    """
    Calculates the full OMG loss for the CVAE.
    """
    # mask = (1.0 - dones.float())
    # denom = mask.sum().clamp_min(1.0)
    # --- 1. Reconstruction Loss ---
    # This forces the CVAE to represent the current state
    recon_loss = 0

    weight_mask = torch.full_like(x, self.feature_weights[0], device=self.device)
    food_mask = (x[..., 1] == 1)
    agent1_mask = (x[..., 2] == 1)
    agent2_mask = (x[..., 3] == 1)

    weight_mask[..., 1][food_mask] = self.feature_weights[1] # Weight for food
    weight_mask[..., 2][agent1_mask] = self.feature_weights[2] # Weight for agent 1
    weight_mask[..., 3][agent2_mask] = self.feature_weights[3] # Weight for agent 2

    # # added weight mask for critical features, to reduce problem majority problem
    # weight_mask = x * self.feature_weights
    # # base for all cells so empty cells aren't ignored completely
    # weight_mask = weight_mask + 1.0

    bce = F.binary_cross_entropy_with_logits(
      reconstructed_x, x, weight=weight_mask, reduction='none')  # (B, H, W, F)
    per_cell = (bce * self.feature_split_weights).sum(dim=-1)
    recon_loss = per_cell.mean(dim=(1, 2))
    # recon_loss = (per_ex * mask).sum() / denom

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

    # omg_loss = (kl_div_per_example * mask).sum() / denom
    omg_loss = kl_div_per_example.mean()

    # KL Divergence loss
    kld_loss = -0.5 * torch.mean(1 + cvae_log_var -
                                 cvae_mu.pow(2) - cvae_log_var.exp())

    # --- 3. Total Loss ---
    total_loss = recon_loss.mean() + beta * omg_loss + self.args.vae_beta * kld_loss
    return total_loss

  def train_step(self, batch, eval_policy, tau, beta):
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
    history = batch['history']  # A dict of state/action/mask lists up to s_t-1
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

    reconstructed_x, cvae_mu, cvae_log_var = self.inference_model(x, history)

    with torch.no_grad():
      vae_mu, vae_log_var = self.subgoal_selector.select(self.prior_model, eval_policy,
                                                         x, future_states, tau)

    loss = self.loss_function(reconstructed_x, x, cvae_mu, cvae_log_var,
                              vae_mu, vae_log_var, infer_mu, infer_log_var, dones, eval_policy._gmix_eps(), beta)

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
