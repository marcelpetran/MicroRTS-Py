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

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--train_vae', action='store_true',
                    default=False, help='Whether to pre-train the VAE')
parser.add_argument('--visualize_vae', action='store_true',
                    default=False, help='Visualize VAE reconstructions logits')
parser.add_argument('--vae_path', type=str, default='./models/vae.pth',
                    help='Path to pre-trained VAE weights')
parser.add_argument('--classic', action='store_true', default=False,
                    help='Use classic Q-learning agent without opponent modeling')
parser.add_argument('--episodes', type=int, default=50_000,
                    help='Number of training episodes')
parser.add_argument('--vae_episodes', type=int, default=30_000,
                    help='Number of episodes for VAE pre-training')
parser.add_argument('--env_size', type=int, default=11,
                    help='Grid size for SimpleForagingEnv')
parser.add_argument('--max_steps', type=int, default=50,
                    help='Max steps per episode')
parser.add_argument('--batch_size', type=int, default=16,
                    help='Batch size for training')
parser.add_argument('--qnet_dim', type=int, default=128,
                    help='Hidden dimension for Q-network')
parser.add_argument('--latent_dim', type=int, default=8,
                    help='Latent dimension for VAE/CVAE')
parser.add_argument('--d_model', type=int, default=256,
                    help='Transformer model dimension')
parser.add_argument('--nhead', type=int, default=4,
                    help='Number of attention heads')
parser.add_argument('--num_encoder_layers', type=int,
                    default=1, help='Number of encoder layers')
parser.add_argument('--num_decoder_layers', type=int,
                    default=1, help='Number of decoder layers')
parser.add_argument('--dim_feedforward', type=int,
                    default=1024, help='Dimension of feedforward network')
parser.add_argument('--dropout', type=float, default=0.12, help='Dropout rate')
parser.add_argument('--vae_beta', type=float, default=0.1,
                    help='Beta for KL loss in VAE')
parser.add_argument('--beta_start', type=float, default=0.0,
                    help='Starting value of beta for KL loss in CVAE')
parser.add_argument('--beta_end', type=float, default=2.0,
                    help='Last value of beta for KL loss in CVAE')
parser.add_argument('--selector_tau_start', type=float, default=2.0,
                    help='Starting temperature value in selector module, using Boltzmann distribution')
parser.add_argument('--selector_tau_end', type=float, default=0.1,
                    help='Last temperature value in selector module, using Boltzmann distribution')
parser.add_argument('--horizon', type=int, default=3,
                    help='Future window H for opponent modeling (Selector module)')
parser.add_argument('--eps_decay_steps', type=int, default=150_000,
                    help='Epsilon decay steps for epsilon-greedy policy')
parser.add_argument('--train_every', type=int, default=4,
                    help='Train every N steps')
parser.add_argument('--target_update_every', type=int,
                    default=1_000, help='Target network update frequency')
parser.add_argument('--replay_capacity', type=int, default=1_000,
                    help='Replay buffer capacity')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed for reproducibility')
parser.add_argument('--folder_id', type=int, default=0,
                    help='Folder ID for saving models and diagrams')
args_parsed = parser.parse_args()

# Necessary directories
os.makedirs(f"./models_{args_parsed.folder_id}", exist_ok=True)
os.makedirs(f"./diagrams_{args_parsed.folder_id}", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

env = SimpleForagingEnv(grid_size=args_parsed.env_size,
                        max_steps=args_parsed.max_steps)

obs_sample = env.reset()
H, W, F_dim = obs_sample[0].shape
NUM_ACTIONS = 4  # Up, Down, Left, Right

args = OMGArgs(
    device=device,
    folder_id=args_parsed.folder_id,
    batch_size=args_parsed.batch_size,
    capacity=args_parsed.replay_capacity,
    horizon_H=args_parsed.horizon,
    qnet_hidden=args_parsed.qnet_dim,
    train_every=args_parsed.train_every,
    target_update_every=args_parsed.target_update_every,
    eps_decay_steps=args_parsed.eps_decay_steps,
    visualise_every_n_step=3,
    max_steps=args_parsed.max_steps,
    selector_mode="conservative",
    vae_beta=args_parsed.vae_beta,
    beta_start=args_parsed.beta_start,
    beta_end=args_parsed.beta_end,
    selector_tau_start=args_parsed.selector_tau_start,
    selector_tau_end=args_parsed.selector_tau_end,
    train_vae=args_parsed.train_vae,
    state_shape=obs_sample[0].shape,
    H=H, W=W,
    state_feature_splits=(F_dim,),
    action_dim=NUM_ACTIONS,
    latent_dim=args_parsed.latent_dim,
    d_model=args_parsed.d_model,
    nhead=args_parsed.nhead,
    num_encoder_layers=args_parsed.num_encoder_layers,
    num_decoder_layers=args_parsed.num_decoder_layers,
    dim_feedforward=args_parsed.dim_feedforward,
    dropout=args_parsed.dropout,
)
if not args_parsed.classic:
  # VAE (Teacher)
  vae = t.TransformerVAE(args).to(device)

  # CVAE (Student)
  cvae = t.TransformerCVAE(args).to(device)

  selector = SubGoalSelector(args)
  cvae_optimizer = torch.optim.Adam(cvae.parameters(), lr=args.cvae_lr)

  op_model = OpponentModel(
    cvae, vae, selector, optimizer=cvae_optimizer, device=device, args=args)

  # --- Pre-train the VAE ---
  if args.train_vae:
    print("Pre-training VAE...")
    vae_optimizer = torch.optim.Adam(vae.parameters(), lr=args.vae_lr)
    vae_replay = ReplayBuffer(10_000)

    op_model.train_vae(env, vae_replay, vae_optimizer, num_epochs=args_parsed.vae_episodes,
                       save_every_n_epochs=args_parsed.vae_episodes, max_steps=args.max_steps, logg=1_000)
    print("VAE pre-training complete.")
  else:
    assert os.path.exists(args_parsed.vae_path), "VAE path does not exist!"
    op_model.prior_model.load_state_dict(torch.load(
      args_parsed.vae_path, map_location=device))
    print("Loaded pre-trained VAE.")

  agent = QLearningAgent(env, op_model, device=device, args=args)
  if args_parsed.visualize_vae:
    agent.visualize_prior()
else:
  agent = QLearningAgentClassic(env, device=device, args=args)

return_list = []
steps_list = []
episode_list = []

for ep in range(args_parsed.episodes):
  stats = agent.run_episode(max_steps=args.max_steps)
  if (ep + 1) % 50 == 0:
    print(
      f"Episode {ep+1}: Return={stats['return']:.2f}, Steps={stats['steps']}")

  # run a test episode
  if (ep + 1) % 500 == 0:
    stats = agent.run_test_episode(max_steps=args.max_steps, render=True)
    return_list.append(stats['return'])
    steps_list.append(stats['steps'])
    episode_list.append(ep + 1)

# Save the trained models
torch.save(cvae.state_dict(), f"./models_{args.folder_id}/cvae.pth")
# Save the Q-network
torch.save(agent.q.state_dict(), f"./models_{args.folder_id}/qnet.pth")
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
plt.savefig(f"./diagrams_{args.folder_id}/training_progress.png")
plt.show()

# python -m cProfile -o run.prof simple_foraging_train.py @args.txt
# snakeviz run.prof