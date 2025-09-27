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

  t.train_vae(env, vae, vae_replay, vae_optimizer, num_epochs=100_000,
              save_every_n_epochs=150_000, batch_size=args.batch_size, max_steps=50, logg=1_000)
  print("VAE pre-training complete.")
  print("Simple test of VAE reconstruction:")
  vae.eval()
  with torch.no_grad():
    sample_state = torch.tensor(obs_sample[0], dtype=torch.float32).unsqueeze(0).to(device)  # (1, H, W, F)
    reconstructed_state, mu, logvar = vae(sample_state)
    print("Original State:\n", obs_sample[0])
    SimpleForagingEnv.render_from_obs(obs_sample[0])
    print("Reconstructed State:\n", reconstructed_state[0])
    SimpleForagingEnv.render_from_obs(t.reconstruct_state(reconstructed_state, args.state_feature_splits)[0])
    print("Latent Mu:\n", mu)
    print("Latent LogVar:\n", logvar)
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
  stats = agent.run_episode(max_steps=30)
  if (ep+1) % 50 == 0:
    print(
      f"Episode {ep+1}: Return={stats['return']:.2f} ({True if stats['return'] > 0 else False}), Steps={stats['steps']}")
  # run a test episode
  if (ep+1) % 100 == 0:
    agent.run_test_episode(max_steps=30)
