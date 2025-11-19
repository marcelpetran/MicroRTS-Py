from dataclasses import dataclass
from typing import Tuple


@dataclass
class OMGArgs:
  device: str = "cpu"  # "cpu" or "cuda"
  gamma: float = 0.99
  lr: float = 5e-5
  batch_size: int = 16
  capacity: int = 250_000
  min_replay: int = 1_000
  train_every: int = 4
  target_update_every: int = 1_000
  visualise_every_n_step: int = 3
  eps_start: float = 1.0
  eps_end: float = 0.05
  eps_decay_steps: int = 800_000
  qnet_hidden: int = 128
  max_history_length: int = 100
  max_steps: int = 30
  seed: int = 0
  folder_id: int = 0

  # OMG-specific
  oracle: bool = False
  gmix_eps_start: float = 1.0      # Eq.(8): start using mostly g_bar
  gmix_eps_end: float = 0.0        # goes to 0 -> use g_hat only
  gmix_eps_decay_steps: int = 600_000
  horizon_H: int = 6
  # "conservative" => Eq.(7), "optimistic" => Eq.(6)
  selector_mode: str = "conservative"
  selector_tau_start: float = 40.0
  selector_tau_end: float = 0.01
  selector_tau_decay_steps: int = 600_000
  train_vae: bool = True
  vae_lr: float = 1e-4
  cvae_lr: float = 1e-4

  # Transformer architecture params
  beta_start: float = 0.1  # Weight for CVAE KL loss
  beta_end: float = 2.0
  beta_decay_steps: int = 800_000
  vae_beta: float = 0.1  # Weight for VAE KL loss
  state_shape: Tuple[int, int, int] = None  # (H, W, F)
  H: int = 7  # grid height
  W: int = 7  # grid width
  state_feature_splits: Tuple[int, ...] = ()
  # for DiscreteActionEmbedder - !at least one mustn't be None!
  action_dim: int = 4
  # for ActionEmbeddings - !at least one mustn't be None!
  action_feature_splits: Tuple[int, ...] = None
  latent_dim: int = 32
  d_model: int = 256
  nhead: int = 4
  num_encoder_layers: int = 1
  num_decoder_layers: int = 1
  dim_feedforward: int = 1024
  dropout: float = 0.1
