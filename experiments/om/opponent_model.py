import experiments.om.transformers as t
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
            raise ValueError(f"Unknown selector_mode: {self.args.selector_mode},\nchoose from ['optimistic', 'conservative']")
    
    def select(self, vae, eval_policy, s_t: torch.Tensor, future_states: torch.Tensor):
        """
        Selects a subgoal by encoding future states to latent space s_g = encode(s_t+K), 
        selecting suitable future state by policy argmin Q-value(s_t, s_g), 
        and encoding choosen state to latent space
        Args:
            vae (VAE): Pre-trained VAE model
            policy (Policy): Policy model with Q-value function
            s_t (Tensor): Current state of shape (1, H, W, F)
            future_states (Tensor): Future states of shape (K, H, W, F) - K is the horizon
        Returns:
            Tensor: latent subgoal of shape (latent_dim)
        """
        vae.eval()
        with torch.no_grad():
            if future_states.dim() == 5: # (B, K, H, W, F)
                future_states = future_states.squeeze(0) # Assume B=1 for selection

            mu, _ = self.vae.encode(future_states) # (K, g_dim)
            #  The policy expects a batch dim for state and subgoal
            # The policy expects a batch dim for state and subgoal
            s_t_batch = s_t if s_t.dim() == 4 else s_t.unsqueeze(0) # (1, H, W, F)
            mu_batch = mu.unsqueeze(0) # (1, K, g_dim)
            
            # The Q-net needs to broadcast s_t over all K subgoals
            # s_t: (1, 1, D_s), mu: (1, K, D_g) -> Q(1, K, A)
            s_t_expanded = s_t_batch.repeat(1, mu.shape[0], 1, 1, 1).view(-1, *s_t.shape[1:])
            mu_expanded = mu_batch.view(-1, mu.shape[-1])
            
            values_flat = eval_policy.value(s_t_expanded, mu_expanded) # (K, A)
            values = values_flat.mean(dim=-1) # V(s,g) = mean_a Q(s,g,a)

            # Gumbel-max trick for differentiable argmin
            best_idx = self.gumbel_max_sample(values)
            best_future_state = future_states[best_idx].unsqueeze(0)
            subgoal, vae_log_var = vae.encode(best_future_state)
        return subgoal, vae_log_var

class OpponentModel:
    def __init__(self, cvae: t.TransformerCVAE, vae: t.TransformerVAE, selector: SubGoalSelector, optimizer, device, args):
        self.inference_model = cvae
        self.prior_model = vae # The pre-trained VAE
        self.subgoal_selector = selector
        self.optimizer = optimizer
        self.device = device
        self.args = args
        self.mse_loss = nn.MSELoss()
    
    def reconstruct_state(reconstructed_state_logits, state_feature_splits):
      """
      Convert the reconstructed logit tensor back to one-hot encoded state.
      """
      reconstructed_state = torch.zeros_like(reconstructed_state_logits, device=reconstructed_state_logits.device)
      start_idx = 0
      for size in state_feature_splits:
          end_idx = start_idx + size
          # Apply softmax to the logits to get probabilities
          probs = F.softmax(reconstructed_state_logits[:, :, :, start_idx:end_idx], dim=-1)
          # Sample indices from the probabilities
          # indices = torch.multinomial(probs.view(-1, size), num_samples=1).view(B, h, w) + start_idx
          # or alternatively, take the argmax
          indices = torch.argmax(probs, dim=-1) + start_idx
          # Set the corresponding one-hot feature to 1
          reconstructed_state.scatter_(3, indices.unsqueeze(-1), 1.0)
          start_idx = end_idx
      
      return reconstructed_state

    def loss_function(self, reconstructed_x, x, cvae_mu, cvae_log_var, vae_mu, vae_log_var, infer_mu, infer_log_var):
        """
        Calculates the full OMG loss for the CVAE.
        """
        # --- 1. Reconstruction Loss ---
        # This part trains the CVAE's decoder
        # This forces the CVAE to represent the current state
        recon_loss = 0
        recon_flat = reconstructed_x.view(-1, sum(self.args.state_feature_splits))
        x_flat = x.view(-1, sum(self.args.state_feature_splits))
        
        x_split = torch.split(x_flat, self.args.state_feature_splits, dim=-1)
        recon_split = torch.split(recon_flat, self.args.state_feature_splits, dim=-1)
        
        for recon_group, x_group in zip(recon_split, x_split):
            recon_loss += F.binary_cross_entropy_with_logits(recon_group, x_group, reduction='mean')

        # --- 2. Inference Loss ---
        # This part trains the CVAE's encoder
        # This forces the CVAE latent vector g_hat to point towards high-probability future
        if self.args.eta < np.random.random(): # eta goes to 0 over time (eq. (8) in the paper)
             # Use the model's own inference (g_hat) as the target
            target_mu, target_log_var = infer_mu.detach(), infer_log_var.detach()
        else:
            # Use the pre-trained VAE's output (g_bar) as the target
            target_mu, target_log_var = vae_mu.detach(), vae_log_var.detach()
        
        # KL Divergence between CVAE's prediction and the target distribution
        omg_loss = -0.5 * torch.sum(1 + cvae_log_var - target_log_var - 
                                    (((target_mu - cvae_mu) ** 2 + cvae_log_var.exp()) / target_log_var.exp()), 
                                    dim=-1)
        
        # The final loss is a weighted sum of reconstruction and inference loss
        total_loss = recon_loss.mean() + self.args.alpha * omg_loss.mean()
        return total_loss
    
    def train_step(self, batch, eval_policy):
        """
        Performs a single training step for the opponent model.
        Args:
            batch (dict): A batch of data from the replay buffer containing:
                - 'state': Tensor of shape (B, T, H, W, F)
                - 'history': Dict of lists of Tensors for historical data
                - 'future_states': Tensor of shape (B, K, H, W, F)
                - 'infer_mu': Tensor of shape (B, latent_dim)
                - 'infer_log_var': Tensor of shape (B, latent_dim)
            eval_policy (Policy): The current evaluation policy used for subgoal selection.
        Returns:
            float: The computed loss for the batch.
        """
        # Unpack the batch from the replay buffer
        states = batch['state'].to(self.device) # (B, T, H, W, F)
        history = batch['history']  # A dict of state/action lists up to s_t-1
        future_states = batch['future_states'].to(self.device) # (B, K, H, W, F)
        infer_mu = batch['infer_mu'].to(self.device)
        infer_log_var = batch['infer_log_var'].to(self.device)
        
        self.inference_model.train()
        self.optimizer.zero_grad()

        x = states[:, 0:, ...]
        reconstructed_x, cvae_mu, cvae_log_var = self.inference_model(x, history)

        with torch.no_grad():
            
            vae_mu, vae_log_var = self.subgoal_selector.select(self.prior_model, eval_policy, 
                                                              x, future_states)

        loss = self.loss_function(reconstructed_x, x, cvae_mu, cvae_log_var, 
                                  vae_mu, vae_log_var, infer_mu, infer_log_var)

        loss.backward()
        self.optimizer.step()

        return loss.item()