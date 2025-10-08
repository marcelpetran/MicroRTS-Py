from simple_foraging_env import SimpleForagingEnv
from opponent_model import OpponentModel, SubGoalSelector
from q_agent import QLearningAgent, ReplayBuffer
from q_agent_classic import QLearningAgentClassic
from omg_args import OMGArgs
import transformers as t
import torch
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--train_vae', action='store_true', default=False, help='Whether to pre-train the VAE')
parser.add_argument('--vae_path', type=str, default='./trained_vae/vae.pth', help='Path to pre-trained VAE weights')
parser.add_argument('--classic', action='store_true', default=False, help='Use classic Q-learning agent without opponent modeling')
parser.add_argument('--episodes', type=int, default=50_000, help='Number of training episodes')
parser.add_argument('--env_size', type=int, default=11, help='Grid size for SimpleForagingEnv')
parser.add_argument('--max_steps', type=int, default=50, help='Max steps per episode')
parser.add_argument('--qnet_dim', type=int, default=128, help='Hidden dimension for Q-network')
args_parsed = parser.parse_args()

# Necessary directories
os.makedirs('./trained_vae', exist_ok=True)
os.makedirs('./trained_cvae', exist_ok=True)
os.makedirs('./trained_qnet', exist_ok=True)
os.makedirs('./diagrams', exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

env = SimpleForagingEnv(grid_size=args_parsed.env_size, max_steps=args_parsed.max_steps)

obs_sample = env.reset()
H, W, F_dim = obs_sample[0].shape
NUM_ACTIONS = 4  # Up, Down, Left, Right

args = OMGArgs(
    batch_size=16,
    capacity=1_000,
    horizon_H=4,
    qnet_hidden=512,
    eps_decay_steps=150_000,
    visualise_every_n_step=3,
    max_steps=args_parsed.max_steps,
    selector_mode="conservative",
    beta=1.002,
    train_vae=args_parsed.train_vae,
    state_shape=obs_sample[0].shape,
    H=H, W=W,
    state_feature_splits=(F_dim,),
    action_dim=NUM_ACTIONS,
    latent_dim=8,
    d_model=256,
    nhead=4,
    num_encoder_layers=1,
    num_decoder_layers=1,
    dim_feedforward=1024,
    dropout=0.12
)

# VAE (Teacher)
vae = t.TransformerVAE(args).to(device)

# CVAE (Student)
cvae = t.TransformerCVAE(args).to(device)

# --- Pre-train the VAE ---
if args.train_vae:
  print("Pre-training VAE...")
  vae_optimizer = torch.optim.Adam(vae.parameters(), lr=args.vae_lr)
  vae_replay = ReplayBuffer(10_000)

  t.train_vae(env, vae, vae_replay, vae_optimizer, num_epochs=30_000,
              save_every_n_epochs=30_000, batch_size=args.batch_size, max_steps=args.max_steps, logg=1_000)
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
  assert os.path.exists(args_parsed.vae_path), "VAE path does not exist!"
  vae.load_state_dict(torch.load(
    args_parsed.vae_path, map_location=device))
  print("Loaded pre-trained VAE.")

selector = SubGoalSelector(args)
cvae_optimizer = torch.optim.Adam(cvae.parameters(), lr=args.cvae_lr)

op_model = OpponentModel(
  cvae, vae, selector, optimizer=cvae_optimizer, device=device, args=args)
if not args_parsed.classic:
  agent = QLearningAgent(env, op_model, device=device, args=args)
else:
  agent = QLearningAgentClassic(env, device=device, args=args)

return_list = []
steps_list = []
episode_list = []

for ep in range(args_parsed.episodes):
  stats = agent.run_episode(max_steps=args.max_steps)
  if (ep+1) % 50 == 0:
    print(f"Episode {ep+1}: Return={stats['return']:.2f}, Steps={stats['steps']}")
    
      
  # run a test episode
  if (ep+1) % 500 == 0:
    stats = agent.run_test_episode(max_steps=args.max_steps, render=True)
    return_list.append(stats['return'])
    steps_list.append(stats['steps'])
    episode_list.append(ep+1)

# Save the trained models
# torch.save(cvae.state_dict(), "./trained_cvae/cvae.pth")
# Save the Q-network
# torch.save(agent.q.state_dict(), "./trained_qnet/qnet.pth")
# torch.save(agent.q.state_dict(), "./trained_qnet/qnetclassic.pth")
# print("Training complete and models saved.")

# Two graphs: return over episodes on the left and steps over episodes on the right
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(episode_list, return_list, label='Return per Episode')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('Return over Episodes')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(episode_list, steps_list, label='Steps per Episode', color='orange')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps over Episodes')
plt.legend()
plt.tight_layout()
plt.savefig('./diagrams/training_progress.png')
plt.show()
