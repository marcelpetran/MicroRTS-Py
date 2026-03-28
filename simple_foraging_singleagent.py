import os
import argparse
import matplotlib.pyplot as plt
from simple_foraging_env import SimpleForagingEnv, SimpleAgent, GreedySwitchAgent, StalkerAgent, ChameleonAgent
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
import wandb
from tqdm import tqdm
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
                    choices=['simple', 'greedy', 'stalker', 'chameleon', 'classic', 'selfplay'],
                    help='Type of opponent agent to play against')
parser.add_argument('--map', type=int, default=1,
                    choices=[i for i in range(1, len(map_layouts)+1)], help='Map layout to use for the environment')
parser.add_argument('--episodes', type=int, default=12_000,
                    help='Number of training episodes')
parser.add_argument('--episodes_per_epoch', type=int, default=500,
                    help='Number of episodes per epoch for logging and evaluation')
parser.add_argument('--pretrain_epochs', type=int, default=15,
                    help='Number of epochs for pretraining the opponent model')
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
parser.add_argument('--replay_capacity', type=int, default=150_000,
                    help='Replay buffer capacity')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed for reproducibility')
parser.add_argument('--folder_id', type=int, default=0,
                    help='Folder ID for saving models and diagrams')
parser.add_argument('--save_models_every', type=int, default=500,
                    help='Frequency of saving model checkpoints (in episodes)')
args_parsed = parser.parse_args()

# Necessary directoriess
os.makedirs(f"./models/{args_parsed.folder_id}", exist_ok=True)
os.makedirs(f"./diagrams/{args_parsed.folder_id}", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

map_name = f"map_{args_parsed.map}"

env = SimpleForagingEnv(max_steps=args_parsed.max_steps,
                        map_layout=map_layouts[args_parsed.map - 1])

obs_sample = env.reset()
H, W, F_dim = obs_sample[0].shape
NUM_ACTIONS = 4  # Up, Down, Left, Right

wandb.init(
    project="om-simple-foraging",
    config=args_parsed,
    name=f"map{args_parsed.map}_{args_parsed.opponent}_id{args_parsed.folder_id}"
)

args = OMGArgs(
    device=device,
    oracle=args_parsed.oracle,
    folder_id=args_parsed.folder_id,
    batch_size=args_parsed.batch_size,
    capacity=args_parsed.replay_capacity,
    qnet_hidden=args_parsed.qnet_dim,
    train_every=args_parsed.train_every,
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
elif args_parsed.opponent == 'stalker':
  opponent_agent = StalkerAgent(agent_id=1)
elif args_parsed.opponent == 'chameleon':
  opponent_agent = ChameleonAgent(agent_id=1)
elif args_parsed.opponent == 'classic':
  opponent_agent = QLearningAgentClassic(env, args=args)
elif args_parsed.opponent == 'selfplay':
  print("Self-play not fully implemented yet. Defaulting to SimpleAgent.")
  opponent_agent = SimpleAgent(agent_id=1)

# Pretraining opponent model with offline dataset
if not args_parsed.classic and not args_parsed.oracle:
  os.makedirs(f"./dataset", exist_ok=True)
  dataset_path = f"./dataset/dataset_{map_name}.pt"

  if not os.path.exists(dataset_path):
    collect_offline_data(num_episodes=500, save_path=dataset_path,
                         map_layout=map_layouts[args_parsed.map - 1])

  print("Loading offline dataset and pretraining opponent model...")
  dataset = torch.load(dataset_path, weights_only=False)
  print(f"Dataset loaded with {len(dataset)} samples. Starting pretraining...")
  agent.model.pretrain(dataset, epochs=args_parsed.pretrain_epochs, batch_size=args_parsed.batch_size)
  print("Opponent Model pretraining complete! Starting RL episodes...")
  # once pretraining is done, we can free up memory by deleting the dataset variable
  del dataset

return_list = []
opp_return_list = []
steps_list = []
episode_list = []

num_epochs = args_parsed.episodes // args_parsed.episodes_per_epoch

for epoch in range(num_epochs):
  epoch_returns, epoch_opp_returns, epoch_steps = [], [], []

  # Training Phase
  pbar = tqdm(range(args_parsed.episodes_per_epoch), desc=f"Epoch {epoch+1:02d}/{num_epochs} [Train]", leave=False)
  
  for ep in pbar:
    global_ep = (epoch * args_parsed.episodes_per_epoch) + ep
    stats = agent.run_episode(opponent_agent, max_steps=args.max_steps)

    epoch_returns.append(stats['return'])
    epoch_opp_returns.append(stats['opp_return'])
    epoch_steps.append(stats['steps'])

    wandb.log({
      "train/return": stats['return'],
      "train/opp_return": stats['opp_return'],
      "train/steps": stats['steps'],
      "episode": global_ep + 1
    })

    pbar.set_postfix({"Ret": f"{stats['return']:.1f}", "Opp": f"{stats['opp_return']:.1f}"})

  # Calculate average training metrics for the epoch
  avg_train_ret = sum(epoch_returns) / len(epoch_returns)
  avg_train_opp = sum(epoch_opp_returns) / len(epoch_opp_returns)

  # Evaluation Phase
  # Save checkpoints at the end of every epoch
  torch.save(agent.q.state_dict(), f"./models/{args.folder_id}/qnet_ep{epoch+1}.pth")
  if not args_parsed.classic and not args_parsed.oracle:
    torch.save(agent.model.inference_model.state_dict(), f"./models/{args.folder_id}/opponent_model_ep{epoch+1}.pth")
  if args_parsed.opponent == 'classic':
    torch.save(opponent_agent.q.state_dict(), f"./models/{args.folder_id}/opponent_classic_qnet_ep{epoch+1}.pth")

  # Run Eval Episodes
  avg_ret, avg_opp_ret, avg_steps = [], [], []
  
  # Rendered Test
  test_stats = agent.run_test_episode(opponent_agent, max_steps=args.max_steps, render=True)
  avg_ret.append(test_stats['return'])
  avg_opp_ret.append(test_stats['opp_return'])
  avg_steps.append(test_stats['steps'])
  
  # Headless Tests
  for _ in range(99):
    st = agent.run_test_episode(opponent_agent, max_steps=args.max_steps, render=False)
    avg_ret.append(st['return'])
    avg_opp_ret.append(st['opp_return'])
    avg_steps.append(st['steps'])

  final_avg_ret = sum(avg_ret) / len(avg_ret)
  final_avg_opp_ret = sum(avg_opp_ret) / len(avg_opp_ret)
  final_avg_steps = sum(avg_steps) / len(avg_steps)

  wandb.log({
    "eval/avg_return": final_avg_ret,
    "eval/avg_opp_return": final_avg_opp_ret,
    "eval/avg_steps": final_avg_steps,
    "epoch": epoch + 1
  })

  return_list.append(final_avg_ret)
  opp_return_list.append(final_avg_opp_ret)
  steps_list.append(final_avg_steps)
  episode_list.append((epoch + 1) * args_parsed.episodes_per_epoch)

  print(f"Epoch {epoch+1:02d} Complete | Train Ret: {avg_train_ret:>4.1f} | Eval Ret: {final_avg_ret:>4.2f} | Eval Opp: {final_avg_opp_ret:>4.2f}")

# Save final models
torch.save(agent.q.state_dict(), f"./models/{args.folder_id}/qnet.pth")
if not args_parsed.classic and not args_parsed.oracle:
  torch.save(agent.model.inference_model.state_dict(),
             f"./models/{args.folder_id}/opponent_model.pth")
if args_parsed.opponent == 'classic':
  torch.save(opponent_agent.q.state_dict(), f"./models/{args.folder_id}/opponent_classic_qnet.pth")
print("Training complete and models saved.")

# Plotting
plt.figure(figsize=(18, 5))

# Subplot 1: Agent 0 Return
plt.subplot(1, 3, 1)
plt.plot(episode_list, return_list, label='Agent 0 Return', color='blue')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('Agent 0 Return over Episodes')
plt.legend()

# Subplot 2: Agent 1 (Opponent) Return
plt.subplot(1, 3, 2)
plt.plot(episode_list, opp_return_list, label='Agent 1 (Opp) Return', color='red')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.title('Opponent Return over Episodes')
plt.legend()

# Subplot 3: Steps
plt.subplot(1, 3, 3)
plt.plot(episode_list, steps_list, label='Steps per Episode', color='orange')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps over Episodes')
plt.legend()

plt.tight_layout()
plt.savefig(f"./diagrams/{args.folder_id}/training_progress.png")
plt.show()

plt.close('all')
wandb.finish()

# python -m cProfile -o run.prof simple_foraging_train.py @args.txt
# snakeviz run.prof