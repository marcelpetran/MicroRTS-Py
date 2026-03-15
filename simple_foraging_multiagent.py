import os
import argparse
import matplotlib.pyplot as plt
from simple_foraging_env import SimpleForagingEnv, SimpleAgent, GreedySwitchAgent
import maps
from maps import *
from opponent_model import OpponentModel
from opponent_model_oracle import OpponentModelOracle
from q_agent import QLearningAgent, ReplayBuffer
from q_agent_classic import QLearningAgentClassic
from omg_args import OMGArgs
from transformers import SpatialOpponentModel
from collect_data import collect_offline_data
import torch
import matplotlib
matplotlib.use('Agg')  # Prevents memory leak on headless clusters

torch.set_float32_matmul_precision('high')

# !KEEP IN MIND, dir(maps) will give all attributes in alphabetical order!
map_layouts = [getattr(maps, m) for m in dir(maps) if m.startswith("MAP_")]

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--oracle', action='store_true', default=False,
                    help='Use oracle opponent model (ground-truth future states)')
parser.add_argument('--classic', action='store_true', default=False,
                    help='Use classic Q-learning agent without opponent modeling')
parser.add_argument('--opponent', type=str, default='simple',
                    choices=['simple', 'greedy', 'classic', 'selfplay'],
                    help='Type of opponent agent to play against')
parser.add_argument('--map', type=int, default=1,
                    choices=[i for i in range(1, len(map_layouts)+1)], help='Map layout to use for the environment')
parser.add_argument('--episodes', type=int, default=12_000,
                    help='Number of training episodes')
parser.add_argument('--max_steps', type=int, default=50,
                    help='Max steps per episode')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size for training')
parser.add_argument('--qnet_dim', type=int, default=256,
                    help='Hidden dimension for Q-network')
parser.add_argument('--cnn_hidden', type=int, default=64,
                    help='Hidden dimension for CNN in Q-network')
parser.add_argument('--d_model', type=int, default=64,
                    help='Transformer model dimension')
parser.add_argument('--nhead', type=int, default=4,
                    help='Number of attention heads')
parser.add_argument('--num_encoder_layers', type=int,
                    default=1, help='Number of encoder layers')
parser.add_argument('--dim_feedforward', type=int,
                    default=256, help='Dimension of feedforward network')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--tau_start', type=float, default=2.1,
                    help='Starting temperature value, using Boltzmann distribution')
parser.add_argument('--tau_end', type=float, default=0.1,
                    help='Last temperature value, using Boltzmann distribution')
parser.add_argument('--train_every', type=int, default=2,
                    help='Train every N steps')
parser.add_argument('--target_update_every', type=int,
                    default=1_000, help='Target network update frequency')
parser.add_argument('--replay_capacity', type=int, default=150_000,
                    help='Replay buffer capacity')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed for reproducibility')
parser.add_argument('--folder_id', type=int, default=0,
                    help='Folder ID for saving models and diagrams')
parser.add_argument('--save_models_every', type=int, default=500,
                    help='Frequency of saving model checkpoints (in episodes)')
args_parsed = parser.parse_args()

# Necessary directories
os.makedirs(f"./models_{args_parsed.folder_id}", exist_ok=True)
os.makedirs(f"./diagrams_{args_parsed.folder_id}", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

map_name = f"map_{args_parsed.map}"

env = SimpleForagingEnv(max_steps=args_parsed.max_steps,
                        map_layout=map_layouts[args_parsed.map - 1])

obs_sample = env.reset()
H, W, F_dim = obs_sample[0].shape
NUM_ACTIONS = 4  # Up, Down, Left, Right

args = OMGArgs(
    device=device,
    oracle=args_parsed.oracle,
    folder_id=args_parsed.folder_id,
    batch_size=args_parsed.batch_size,
    capacity=args_parsed.replay_capacity,
    qnet_hidden=args_parsed.qnet_dim,
    train_every=args_parsed.train_every,
    target_update_every=args_parsed.target_update_every,
    visualise_every_n_step=3,
    max_steps=args_parsed.max_steps,
    tau_start=args_parsed.tau_start,
    tau_end=args_parsed.tau_end,
    state_shape=obs_sample[0].shape,
    H=H, W=W,
    action_dim=NUM_ACTIONS,
    d_model=args_parsed.d_model,
    nhead=args_parsed.nhead,
    num_encoder_layers=args_parsed.num_encoder_layers,
    dim_feedforward=args_parsed.dim_feedforward,
    dropout=args_parsed.dropout,
)
if not args_parsed.classic:
  if not args_parsed.oracle:
    inference_model = SpatialOpponentModel(args=args).to(device)
    op_model = OpponentModel(inference_model, args=args)
  else:
    op_model = OpponentModelOracle(
      args=args, opp_start=env._get_agent_positions()[1])

  agent = QLearningAgent(env, op_model, args=args)

else:
  agent = QLearningAgentClassic(env, args=args)

if args_parsed.opponent == 'simple':
  opponent_agent = SimpleAgent(agent_id=1)
elif args_parsed.opponent == 'greedy':
  opponent_agent = GreedySwitchAgent(agent_id=1)
elif args_parsed.opponent == 'classic':
  opponent_agent = QLearningAgentClassic(env, args=args)
elif args_parsed.opponent == 'selfplay':
  print("Self-play not fully implemented yet. Defaulting to SimpleAgent.")
  opponent_agent = SimpleAgent(agent_id=1)

if not args_parsed.classic and not args_parsed.oracle:
  os.makedirs(f"./dataset", exist_ok=True)
  dataset_path = f"./dataset/dataset_{map_name}.pt"

  if not os.path.exists(dataset_path):
    collect_offline_data(num_episodes=2000, save_path=dataset_path,
                         map_layout=map_layouts[args_parsed.map - 1])

  print("Loading offline dataset and pretraining opponent model...")
  dataset = torch.load(dataset_path, weights_only=False)
  agent.model.pretrain(dataset, epochs=15, batch_size=args_parsed.batch_size)
  print("Opponent Model pretraining complete! Starting RL episodes...")

return_list = []
steps_list = []
episode_list = []

for ep in range(args_parsed.episodes):
  stats = agent.run_episode(opponent_agent, max_steps=args.max_steps)

  if (ep + 1) % 50 == 0:
    print(
      f"Episode {ep+1}: Return={stats['return']:.2f}, Steps={stats['steps']}")

  # Run test episodes
  if (ep + 1) % args_parsed.save_models_every == 0:
    torch.save(agent.q.state_dict(),
               f"./models_{args.folder_id}/qnet_ep{ep+1}.pth")
    if not args_parsed.classic and not args_parsed.oracle:
      torch.save(agent.model.inference_model.state_dict(),
                 f"./models_{args.folder_id}/opponent_model_ep{ep+1}.pth")
    avg_ret, avg_steps = [], []
    stats = agent.run_test_episode(
      opponent_agent, max_steps=args.max_steps, render=True)
    avg_ret.append(stats['return'])
    avg_steps.append(stats['steps'])
    for i in range(99):
      st = agent.run_test_episode(
        opponent_agent, max_steps=args.max_steps, render=False)
      avg_ret.append(st['return'])
      avg_steps.append(st['steps'])

    return_list.append(sum(avg_ret) / len(avg_ret))
    steps_list.append(sum(avg_steps) / len(avg_steps))
    episode_list.append(ep + 1)
    print(
      f"Test Episode {ep+1}: Return={stats['return']:.2f} | Avg={return_list[-1]:.2f}, Steps={stats['steps']} | Avg={steps_list[-1]:.1f}")

# Save final models
torch.save(agent.q.state_dict(), f"./models_{args.folder_id}/qnet.pth")
if not args_parsed.classic and not args_parsed.oracle:
  torch.save(agent.model.inference_model.state_dict(),
             f"./models_{args.folder_id}/opponent_model.pth")
print("Training complete and models saved.")

# Plotting
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
plt.close('all')

# python -m cProfile -o run.prof simple_foraging_train.py @args.txt
# snakeviz run.prof
