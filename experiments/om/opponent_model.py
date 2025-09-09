import re
import tranformers as t
import torch
import torch.nn as nn
import torch.nn.functional as torch_f
import numpy as np

class SubGoalSelector:
    def __init__(self):
        pass
    
    def select_subgoal(self, vae, eval_policy, s_t: torch.Tensor, future_states: torch.Tensor):
        """
        Selects a subgoal by encoding future states to latent space s_g = encode(s_t+K), 
        selecting suitable future state by policy argmin Q-value(s_t, s_g), 
        and encoding choosen state to latent space
        Args:
            vae (VAE): Pre-trained VAE model
            policy (Policy): Policy model with Q-value function
            s_t (Tensor): Current state of shape (1, H, W, F)
            future_states (Tensor): Future states of shape (K, H, W, F)
        Returns:
            Tensor: latent subgoal of shape (latent_dim)
        """
        self.model.eval()
        with torch.no_grad():
            mu, _ = self.model.encode(future_states)
            values = eval_policy.q_value(s_t, mu) # TODO: when policy is implemented, check compatibility
            best_idx = torch.argmin(values)
            best_future_state = future_states[best_idx].unsqueeze(0)
            subgoal, vae_log_var = vae.encode(best_future_state)
        return subgoal, vae_log_var

class OpponentModel:
    def __init__(self, cvae: t.TransformerCVAE, vae: t.TransformerVAE, selector: SubGoalSelector, optimizer, device, max_history=5, alpha=1.0, eta=1.0):
        self.inference_model = cvae
        self.target_prior = vae
        self.subgoal_selector = selector
        self.optimizer = optimizer
        self.device = device
        self.max_history = max_history
        self.alpha = alpha
        self.eta = eta
        self.mse_loss = nn.MSELoss()
    
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

    def inference_loss(self, batch, eval_policy):
      """
      Loss function to train encoder to match the latent distribution of a pre-trained VAE.
      Uses Mean Squared Error between the predicted and target means.
      We do not need to use decoder, or we simply copy the weights from the pre-trained VAE.
      Args:
          predicted_mu (Tensor): Predicted mean from the CVAE encoder. Shape: (B, latent_dim)
          target_mu (Tensor): Target mean from the pre-trained VAE encoder. Shape: (B, latent_dim)
      Returns:
          Tensor: Computed MSE loss.
      """
      # TODO: simplify the code below, remove unnecessary parts, put them in train_step if not needed here
      terminated = batch["terminated"][:, :-1].float().squeeze(-1)
      infer_mu = batch["infer_mu"][:, :-1].float().squeeze(-1)
      infer_log_var = batch["infer_log_var"][:, :-1].float().squeeze(-1)
      mask = batch["filled"][:, :-1].float().squeeze(-1)
      mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

      vae_mu, vae_log_var = self.subgoal_selector.select_subgoal(self.target_prior, eval_policy, batch["states"][:, :-1], batch["future_states"][:, :-1])
      cvae_input = batch["states"][:, 1:].float()
      cvae_recon, cvae_mu, cvae_log_var = self.inference_model(batch["states"][:, :-1].float(), batch["history"])

      recons_loss = self.mse_loss(cvae_recon, cvae_input, reduction='none').mean(dim=-1)
      recons_loss = recons_loss * mask

      if self.args.eta < np.random.random():
          omg_loss = -0.5 * torch.sum(1 + cvae_log_var - vae_log_var.detach() - ((vae_mu.detach() - cvae_mu) ** 2 + cvae_log_var.exp())/vae_log_var.detach().exp(), dim=-1)
      else:
          omg_loss = -0.5 * torch.sum(1 + cvae_log_var - infer_log_var.detach() - ((infer_mu.detach() - cvae_mu) ** 2 + cvae_log_var.exp())/infer_log_var.detach().exp(), dim=-1)
      omg_loss = omg_loss * mask

      loss = recons_loss.mean() + self.args.omg_cvae_alpha * omg_loss.mean()
      return loss
    
    def train_step(self, batch, eval_policy):
        """
        TODO
        Performs a single training step for the opponent model.
        Args:
            x (Tensor): Input state of shape (1, H, W, F)
            history (dict): A dictionary containing historical states and actions.
                            Expected keys: 'states', 'actions', 'opp_actions' (if applicable)
        Returns:
            float: The training loss for this step.
        """
        x = batch["states"][:, :-1].float()
        history = {k: v[:, :-1] for k, v in batch["history"].items()}

        self.inference_model.train()
        self.optimizer.zero_grad()

        # Encode with CVAE to get predicted latent distribution
        predicted_mu, predicted_logvar = self.inference_model.encode(x, history)

        # Encode with pre-trained VAE to get target latent distribution
        with torch.no_grad():
            target_mu, target_logvar = self.target_prior.encode(x)

        loss = self.inference_loss(batch, eval_policy)

        loss.backward()
        self.optimizer.step()

        return loss.item()