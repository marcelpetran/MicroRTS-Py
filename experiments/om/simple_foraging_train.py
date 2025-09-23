from simple_foraging_env import SimpleForagingEnv
from opponent_model import OpponentModel, SubGoalSelector
from q_agent import QLearningAgent, OMGArgs, ReplayBuffer
import experiments.om.transformers as t  # your module name
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

env = SimpleForagingEnv(grid_size=5, max_steps=50)

obs_sample = env.reset()
H, W, F = obs_sample[0].shape
NUM_ACTIONS = 4 # Up, Down, Left, Right

args = OMGArgs(
    batch_size=16,
    capacity=100_000,
    horizon_H=3,
    selector_mode="conservative",
    alpha=0.1, # Weight for the KL loss
    state_shape=obs_sample[0].shape,
    H=H, W=W,
    state_feature_splits=(F,),
    action_dim=NUM_ACTIONS,
    latent_dim=8,
    d_model=128,
    nhead=2,
    num_encoder_layers=2,
    num_decoder_layers=2,
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
  vae_optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
  vae_replay = ReplayBuffer(10_000)

  t.train_vae(env, vae, vae_replay, vae_optimizer, num_epochs=5_000_000, save_every_n_epochs=100_000, batch_size=16, max_steps=50, logg=1_000)
  print("VAE pre-training complete.")
else:
  vae.load_state_dict(torch.load("./trained_VAE/vae_simple_foraging.pth", map_location=device))
  print("Loaded pre-trained VAE.")

selector = SubGoalSelector(args)
cvae_optimizer = torch.optim.Adam(cvae.parameters(), lr=3e-4)

op_model = OpponentModel(cvae, vae, selector, optimizer=cvae_optimizer, device=device, args=args)
agent = QLearningAgent(env, op_model, device=device, args=args)

for ep in range(10000):
    stats = agent.run_episode(max_steps=50)
    if ep % 50 == 0:
        print(f"Episode {ep}: Return={stats['return']:.2f}, Steps={stats['steps']}")