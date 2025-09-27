from dataclasses import dataclass
from typing import Tuple

@dataclass
class OMGArgs:
  gamma: float = 0.99
  lr: float = 2.5e-4
  batch_size: int = 16
  capacity: int = 50_000
  min_replay: int = 1_000
  train_every: int = 4
  target_update_every: int = 1_000
  eps_start: float = 1.0
  eps_end: float = 0.05
  eps_decay_steps: int = 50_000
  qnet_hidden: int = 256
  maximum_history_length = 10

  # OMG-specific
  gmix_eps_start: float = 1.0      # Eq.(8): start using mostly g_bar
  gmix_eps_end: float = 0.0        # goes to 0 -> use g_hat only
  gmix_eps_decay_steps: int = 50_000
  horizon_H: int = 3
  # "conservative" => Eq.(7), "optimistic" => Eq.(6)
  selector_mode: str = "conservative"
  train_vae: bool = True

  # Transformer architecture params
  alpha: float = 1.0
  beta: float = 1.002
  state_shape: Tuple[int, int, int] = None  # (H, W, F)
  H: int = 5  # grid height
  W: int = 5  # grid width
  state_feature_splits: Tuple[int, ...] = ()
  # for DiscreteActionEmbedder - !at least one mustn't be None!
  action_dim: int = None
  # for ActionEmbeddings - !at least one mustn't be None!
  action_feature_splits: Tuple[int, ...] = None
  latent_dim: int = 32
  d_model: int = 256
  nhead: int = 8
  num_encoder_layers: int = 6
  num_decoder_layers: int = 6
  dim_feedforward: int = 1024
  dropout: float = 0.1