from simple_foraging_env import SimpleForagingEnv
from opponent_model import OpponentModel, SubGoalSelector
from q_agent import QLearningAgent, ReplayBuffer
from omg_args import OMGArgs
import experiments.om.transformers as t  # your module name
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

env = SimpleForagingEnv(grid_size=5, max_steps=50)

obs_sample = env.reset()
H, W, F = obs_sample[0].shape
NUM_ACTIONS = 4  # Up, Down, Left, Right

args = OMGArgs(
    batch_size=16,
    capacity=1_000,
    horizon_H=3,
    qnet_hidden=128,
    selector_mode="conservative",
    beta=2.0,  # Weight for the KL loss
    train_vae=True,
    state_shape=obs_sample[0].shape,
    H=H, W=W,
    state_feature_splits=(F,),
    action_dim=NUM_ACTIONS,
    latent_dim=8,
    d_model=128,
    nhead=2,
    num_encoder_layers=1,
    num_decoder_layers=1,
    dim_feedforward=512,
    dropout=0.1
)

# VAE (Teacher)
vae = t.TransformerVAE(args).to(device)

# CVAE (Student)
cvae = t.TransformerCVAE(args).to(device)

# --- Pre-train the VAE ---
if args.train_vae:
  print("Pre-training VAE...")
  vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-4)
  vae_replay = ReplayBuffer(10_000)

  t.train_vae(env, vae, vae_replay, vae_optimizer, num_epochs=50_000,
              save_every_n_epochs=50_000, batch_size=args.batch_size, max_steps=50, logg=1_000)
  print("VAE pre-training complete.")
else:
  vae.load_state_dict(torch.load(
    "./trained_VAE/vae_simple_foraging.pth", map_location=device))
  print("Loaded pre-trained VAE.")

selector = SubGoalSelector(args)
cvae_optimizer = torch.optim.Adam(cvae.parameters(), lr=3e-4)

op_model = OpponentModel(
  cvae, vae, selector, optimizer=cvae_optimizer, device=device, args=args)
agent = QLearningAgent(env, op_model, device=device, args=args)

for ep in range(50_000):
  stats = agent.run_episode(max_steps=15)
  if ep % 50 == 0:
    print(
      f"Episode {ep}: Return={stats['return']:.2f} ({True if stats['return'] > 0 else False}), Steps={stats['steps']}")

# Show map of Q-values for agent 0 at initial state
import numpy as np
import matplotlib.pyplot as plt

obs = env.reset()
q_values = np.zeros((env.grid_size, env.grid_size))
for i in range(env.grid_size):
  for j in range(env.grid_size):
    # Create a copy of the observation and place agent 0 at (i, j)
    obs_copy = {0: obs[0].copy(), 1: obs[1].copy()}
    obs_copy[0][:, :, 2] = 0  # Clear agent 0 position
    obs_copy[0][i, j, 2] = 1  # Set agent 0 position

    state_tensor = torch.tensor(
      obs_copy[0], dtype=torch.float32).unsqueeze(0).to(device)  # (1, H, W, F)
    # Use a dummy subgoal (zeros) since we don't have one at the start
    dummy_subgoal = torch.zeros((1, args.latent_dim), device=device)
    with torch.no_grad():
      q_val = agent.q.value(state_tensor, dummy_subgoal)  # (1, A)
      q_values[i, j] = q_val[0].max().item()  # Max Q-value over actions
plt.imshow(q_values, cmap='viridis')
plt.colorbar(label='Max Q-value')
plt.title('Q-value Heatmap for Agent 0 at Initial State')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.show()