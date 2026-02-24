from dataclasses import dataclass
from typing import Tuple


@dataclass
class OMGArgs:
  device: str = "cpu"  # cpu, cuda, mps
  gamma: float = 0.955
  lr: float = 1e-4
  batch_size: int = 128
  capacity: int = 150_000
  min_replay: int = 1_000
  train_every: int = 4
  target_update_every: int = 1_000
  visualise_every_n_step: int = 3
  qnet_hidden: int = 256
  max_history_length: int = 31
  max_steps: int = 30
  seed: int = 0
  folder_id: int = 0

  oracle: bool = False
  tau_soft: float = 0.005
  tau_start: float = 2.0
  tau_end: float = 0.01
  tau_decay_steps: int = 100_000
  beta_start: float = 1.0
  beta_end: float = 0.01
  beta_decay_steps: int = 100_000
  eps_start: float = 1.0
  eps_end: float = 0.05
  eps_decay_steps: int = 50_000


  # Transformer architecture params
  state_shape: Tuple[int, int, int] = None  # (H, W, F)
  H: int = 7  # grid height
  W: int = 7  # grid width
  action_dim: int = 4
  d_model: int = 256
  nhead: int = 4
  num_encoder_layers: int = 1
  dim_feedforward: int = 1024
  dropout: float = 0.1
