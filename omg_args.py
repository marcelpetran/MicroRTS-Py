from dataclasses import dataclass
from typing import Tuple


@dataclass
class OMGArgs:
  device: str = "cpu"  # cpu, cuda, mps
  gamma: float = 0.985
  lr: float = 3e-4
  batch_size: int = 128
  capacity: int = 250_000
  sl_capacity: int = 500_000
  min_replay: int = 256
  train_every: int = 4
  visualise_every_n_step: int = 3
  qnet_hidden: int = 256
  cnn_hidden: int = 64
  max_history_length: int = 50
  max_steps: int = 50
  seed: int = 0
  folder_id: int = 0

  oracle: bool = False
  tau_soft: float = 0.001
  tau_start: float = 2.0
  tau_end: float = 0.05
  tau_decay_steps: int = 600_000

  # Transformer architecture params
  state_shape: Tuple[int, int, int] = None  # (H, W, F)
  H: int = 7  # grid height
  W: int = 7  # grid width
  action_dim: int = 4
  d_model: int = 64
  nhead: int = 4
  num_encoder_layers: int = 1
  dim_feedforward: int = 256
  dropout: float = 0.1
