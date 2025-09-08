import tranformers as t
import torch
import torch.nn as nn

class SubGoalSelector:
    def __init__(self):
        pass
    
    def select_subgoal(self, vae, policy, s_t: torch.Tensor, future_states: torch.Tensor):
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
            values = policy.q_value(s_t, mu)
            best_idx = torch.argmin(values)
            best_future_state = future_states[best_idx].unsqueeze(0)
            subgoal, _ = vae.encode(best_future_state)
        return subgoal

class OpponentModel:
    def __init__(self, cvae: t.TransformerCVAE, vae: t.TransformerVAE, optimizer, device):
        self.inference_model = cvae
        self.vae = vae
        self.optimizer = optimizer
        self.device = device

    def train_step(self, x, history):
        """
        Performs a single training step for the opponent model.
        Args:
            x (Tensor): Input state of shape (B, H, W, F)
            history (dict): A dictionary containing historical states and actions.
                            Expected keys: 'states', 'actions', 'opp_actions' (if applicable)
        Returns:
            float: The training loss for this step.
        """
        self.cvae.train()
        self.optimizer.zero_grad()

        # Encode with CVAE to get predicted latent distribution
        predicted_mu, predicted_logvar = self.cvae.encode(x, history)

        # Encode with pre-trained VAE to get target latent distribution
        with torch.no_grad():
            target_mu, target_logvar = self.vae.encode(x)

        # Compute loss between predicted and target distributions
        loss = t.latent_matching_loss(predicted_mu, target_mu)

        # Backpropagation and optimization step
        loss.backward()
        self.optimizer.step()

        return loss.item()