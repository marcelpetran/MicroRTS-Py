from dataclasses import dataclass


@dataclass
class OMGArgs:
    device: str = "cpu"  # cpu, cuda, mps
    gamma: float = 0.985
    lr: float = 3e-4
    lr_om: float = 1e-4
    batch_size: int = 128
    capacity: int = 250_000
    sl_capacity: int = 500_000
    min_replay: int = 10_000
    train_every: int = 4
    visualise_every_n_step: int = 3
    qnet_hidden: int = 256
    cnn_hidden: int = 64
    max_history_length: int = 50
    max_steps: int = 50
    seed: int = 0
    folder_id: int = 0
    true_intent: bool = False
    belief_channels: int = 3  # food_belief, opp_last_seen, opp_age
    belief_map_prior: bool = True

    oracle: bool = False
    tau_soft: float = 0.001
    tau_start: float = 1.5
    tau_end: float = 0.05
    tau_decay_steps: int = 600_000

    sigma: float = 2.0
    sigma_end: float = 0.5
    sigma_decay_steps: int = 150_000

    # Transformer architecture params
    state_shape: tuple[int, int, int] = (11, 11, 6)  # (H, W, F)
    H: int = 7  # grid height
    W: int = 7  # grid width
    action_dim: int = 5
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 1
    dim_feedforward: int = 256
    dropout: float = 0.1
