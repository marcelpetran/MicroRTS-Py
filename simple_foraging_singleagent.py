import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib
import wandb
from tqdm import tqdm

from simple_foraging_env import SimpleForagingEnv, SimpleAgent, GreedySwitchAgent, StalkerAgent, ChameleonAgent
import maps
from maps import *
from opponent_model import OpponentModel
from transformers import SpatialOpponentModel
from collect_data import collect_offline_data
from omg_args import OMGArgs

# Import the standard RL agents
from q_agent import QLearningAgent
from q_agent_classic import QLearningAgentClassic

matplotlib.use('Agg')  # Prevents memory leak on headless clusters
torch.set_float32_matmul_precision('high')

map_layouts = [getattr(maps, m) for m in dir(maps) if m.startswith("MAP_")]

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--opponent', type=str, default='simple',
                    choices=['simple', 'greedy', 'stalker', 'chameleon'],
                    help='Type of heuristic opponent agent to play against')
parser.add_argument('--map', type=int, default=1,
                    choices=[i for i in range(1, len(map_layouts) + 1)], help='Map layout to use')
parser.add_argument('--episodes', type=int, default=12_000,
                    help='Total training episodes per phase')
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
                    help='Starting temperature value for Boltzmann')
parser.add_argument('--tau_end', type=float, default=0.1,
                    help='Ending temperature value for Boltzmann')
parser.add_argument('--tau_decay_steps', type=int, default=600_000,
                    help='Number of steps to decay Boltzmann temperature')
parser.add_argument('--train_every', type=int, default=2,
                    help='Train every N steps')
parser.add_argument('--replay_capacity', type=int, default=150_000,
                    help='Replay buffer capacity')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed for reproducibility')
parser.add_argument('--folder_id', type=int, default=0,
                    help='Folder ID for saving models and diagrams')
args_parsed = parser.parse_args()

# Setup directories
os.makedirs(f"./models/{args_parsed.folder_id}", exist_ok=True)
os.makedirs(f"./diagrams/{args_parsed.folder_id}", exist_ok=True)
os.makedirs("./dataset", exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

env = SimpleForagingEnv(max_steps=args_parsed.max_steps,
                        map_layout=map_layouts[args_parsed.map - 1])

obs_sample = env.reset()
H, W, F_dim = obs_sample[0].shape
NUM_ACTIONS = 4

wandb.init(
    project="om-simple-foraging",
    config=args_parsed,
    name=f"map{args_parsed.map}_vs_{args_parsed.opponent}_id{args_parsed.folder_id}"
)

args = OMGArgs(
    device=device,
    folder_id=args_parsed.folder_id,
    batch_size=args_parsed.batch_size,
    capacity=args_parsed.replay_capacity,
    qnet_hidden=args_parsed.qnet_dim,
    cnn_hidden=args_parsed.cnn_hidden,
    train_every=args_parsed.train_every,
    max_steps=args_parsed.max_steps,
    tau_start=args_parsed.tau_start,
    tau_end=args_parsed.tau_end,
    tau_decay_steps=args_parsed.tau_decay_steps,
    state_shape=obs_sample[0].shape,
    H=H, W=W, action_dim=NUM_ACTIONS,
    d_model=args_parsed.d_model, nhead=args_parsed.nhead,
    num_encoder_layers=args_parsed.num_encoder_layers,
    dim_feedforward=args_parsed.dim_feedforward, dropout=args_parsed.dropout,
)

# Initialize Heuristic Opponent
if args_parsed.opponent == 'simple':
  opponent_agent = SimpleAgent(agent_id=1)
elif args_parsed.opponent == 'greedy':
  opponent_agent = GreedySwitchAgent(agent_id=1)
elif args_parsed.opponent == 'stalker':
  opponent_agent = StalkerAgent(agent_id=1)
elif args_parsed.opponent == 'chameleon':
  opponent_agent = ChameleonAgent(agent_id=1)

num_epochs = args_parsed.episodes // args_parsed.episodes_per_epoch
eval_episodes = 100

# ==========================================
# PHASE 1: TRAIN CLASSIC AGENT
# ==========================================
print(
  f"\n--- PHASE 1: Training Classic Agent vs {args_parsed.opponent.capitalize()} ---")
agent_classic = QLearningAgentClassic(env, args=args)

phase1_train_returns, phase1_train_steps, phase1_entropies, phase1_q_losses = [], [], [], []
phase1_eval_opp_returns, phase1_eval_steps, phase1_eval_returns = [], [], []

for epoch in range(num_epochs):
  epoch_returns, epoch_entropies, epoch_steps, epoch_q_losses = [], [], [], []

  # Training
  pbar = tqdm(range(args_parsed.episodes_per_epoch),
              desc=f"Phase 1 Epoch {epoch + 1:02d}/{num_epochs} [Train]", leave=False)
  for ep in pbar:
    stats = agent_classic.run_episode(opponent_agent, max_steps=args.max_steps)
    epoch_returns.append(stats['return'])
    epoch_entropies.append(stats['avg_entropy'])
    epoch_steps.append(stats['steps'])
    epoch_q_losses.append(stats['avg_q_loss'])

  # Evaluation
  eval_rets, eval_opp_rets, eval_steps = [], [], []
  for _ in range(eval_episodes):
    test_stats = agent_classic.run_test_episode(
      opponent_agent, max_steps=args.max_steps, render=False)
    eval_rets.append(test_stats['return'])
    eval_opp_rets.append(test_stats['opp_return'])
    eval_steps.append(test_stats['steps'])

  avg_eval_ret = sum(eval_rets) / eval_episodes
  avg_eval_opp = sum(eval_opp_rets) / eval_episodes
  avg_eval_steps = sum(eval_steps) / eval_episodes

  avg_train_ret = sum(epoch_returns) / len(epoch_returns)
  avg_train_steps = sum(epoch_steps) / len(epoch_steps)
  avg_entropy = sum(epoch_entropies) / len(epoch_entropies)
  avg_q_loss = sum(epoch_q_losses) / len(epoch_q_losses)

  phase1_train_returns.append(avg_train_ret)
  phase1_train_steps.append(avg_train_steps)
  phase1_entropies.append(avg_entropy)
  phase1_q_losses.append(avg_q_loss)

  phase1_eval_returns.append(avg_eval_ret)
  phase1_eval_opp_returns.append(avg_eval_opp)
  phase1_eval_steps.append(avg_eval_steps)

  wandb.log({
      "classic/train_return": avg_train_ret,
      "classic/train_steps": avg_train_steps,
      "classic/train_entropy": avg_entropy,
      "classic/avg_q_loss": avg_q_loss,
      "classic/eval_return": avg_eval_ret,
      "classic/eval_opp_return": avg_eval_opp,
      "classic/eval_steps": avg_eval_steps,
      "epoch": epoch + 1
  })

  torch.save(agent_classic.q.state_dict(),
             f"./models/{args.folder_id}/classic_qnet_ep{epoch + 1}.pth")
  print(f"Phase 1 | Epoch {epoch + 1:02d} | Train Ret: {avg_train_ret:>4.1f} | Eval Ret: {avg_eval_ret:>4.2f} | Eval Opp: {avg_eval_opp:>4.2f} | Entropy: {avg_entropy:.4f}")


# ==========================================
# PHASE 2: TRAIN OM AGENT
# ==========================================
print(
  f"\n--- PHASE 2: Training OM Agent vs {args_parsed.opponent.capitalize()} ---")
inference_model = SpatialOpponentModel(args=args).to(device)
op_model = OpponentModel(inference_model, args=args)
agent_om = QLearningAgent(env, op_model, args=args)

# OM Pretraining
dataset_path = f"./dataset/dataset_map_{args_parsed.map}.pt"
if not os.path.exists(dataset_path):
  print("Collecting Offline Data for OM Pretraining...")
  collect_offline_data(num_episodes=500, save_path=dataset_path,
                       map_layout=map_layouts[args_parsed.map - 1])

print("Loading dataset and pretraining OM...")
dataset = torch.load(dataset_path, weights_only=False)
agent_om.model.pretrain(
  dataset, epochs=args_parsed.pretrain_epochs, batch_size=args_parsed.batch_size)
del dataset

phase2_train_returns, phase2_train_steps, phase2_entropies, phase2_q_losses, phase2_model_losses = [], [], [], [], []
phase2_eval_opp_returns, phase2_eval_steps, phase2_eval_returns = [], [], []

for epoch in range(num_epochs):
  epoch_returns, epoch_entropies, epoch_steps, epoch_q_losses, epoch_model_losses = [], [], [], [], []
  epoch_eval_kl_errors, epoch_eval_spatial_errors = [], []

  # Training
  pbar = tqdm(range(args_parsed.episodes_per_epoch),
              desc=f"Phase 2 Epoch {epoch + 1:02d}/{num_epochs} [Train]", leave=False)
  for ep in pbar:
    stats = agent_om.run_episode(opponent_agent, max_steps=args.max_steps)
    epoch_returns.append(stats['return'])
    epoch_entropies.append(stats['avg_entropy'])
    epoch_steps.append(stats['steps'])
    epoch_q_losses.append(stats['avg_q_loss'])
    epoch_model_losses.append(stats['avg_model_loss'])

  # Evaluation
  eval_rets, eval_opp_rets, eval_steps = [], [], []
  for _ in range(eval_episodes):
    test_stats = agent_om.run_test_episode(
      opponent_agent, max_steps=args.max_steps, render=False)
    eval_rets.append(test_stats['return'])
    eval_opp_rets.append(test_stats['opp_return'])
    eval_steps.append(test_stats['steps'])
    epoch_eval_kl_errors.append(test_stats['avg_kl_error'])
    epoch_eval_spatial_errors.append(test_stats['avg_spatial_error'])

  avg_eval_ret = sum(eval_rets) / eval_episodes
  avg_eval_opp = sum(eval_opp_rets) / eval_episodes
  avg_eval_steps = sum(eval_steps) / eval_episodes
  avg_eval_kl_error = sum(epoch_eval_kl_errors) / len(epoch_eval_kl_errors)
  avg_eval_spatial_error = sum(
    epoch_eval_spatial_errors) / len(epoch_eval_spatial_errors)

  avg_train_ret = sum(epoch_returns) / len(epoch_returns)
  avg_train_steps = sum(epoch_steps) / len(epoch_steps)
  avg_entropy = sum(epoch_entropies) / len(epoch_entropies)
  avg_q_loss = sum(epoch_q_losses) / len(epoch_q_losses)
  avg_model_loss = sum(epoch_model_losses) / len(epoch_model_losses)

  phase2_eval_returns.append(avg_eval_ret)
  phase2_eval_opp_returns.append(avg_eval_opp)
  phase2_eval_steps.append(avg_eval_steps)

  phase2_train_returns.append(avg_train_ret)
  phase2_train_steps.append(avg_train_steps)
  phase2_entropies.append(avg_entropy)
  phase2_q_losses.append(avg_q_loss)
  phase2_model_losses.append(avg_model_loss)

  wandb.log({
      "om/train_return": avg_train_ret,
      "om/train_steps": avg_train_steps,
      "om/train_entropy": avg_entropy,
      "om/avg_q_loss": avg_q_loss,
      "om/avg_model_loss": avg_model_loss,
      "om/eval_return": avg_eval_ret,
      "om/eval_opp_return": avg_eval_opp,
      "om/eval_steps": avg_eval_steps,
      "om/eval_kl_error": avg_eval_kl_error,
      "om/eval_spatial_error": avg_eval_spatial_error,
      "epoch": epoch + 1
  })

  torch.save(agent_om.q.state_dict(),
             f"./models/{args.folder_id}/om_qnet_ep{epoch + 1}.pth")
  torch.save(agent_om.model.inference_model.state_dict(),
             f"./models/{args.folder_id}/om_inference_ep{epoch + 1}.pth")
  print(f"Phase 2 | Epoch {epoch + 1:02d} | Train Ret: {avg_train_ret:>4.1f} | Eval Ret: {avg_eval_ret:>4.2f} | Eval Opp: {avg_eval_opp:>4.2f} | Entropy: {avg_entropy:.4f} | Q Loss: {avg_q_loss:.4f} | Model Loss: {avg_model_loss:.4f}")


# ==========================================
# PLOTTING
# ==========================================
print("\n--- Generating Evaluation Charts ---")
episode_list = [
  (i + 1) * args_parsed.episodes_per_epoch for i in range(num_epochs)]

plt.figure(figsize=(18, 5))

# Subplot 1: Evaluation Returns
plt.subplot(1, 3, 1)
plt.plot(episode_list, phase1_eval_returns,
         label='Classic Agent', color='blue', linestyle='--')
plt.plot(episode_list, phase2_eval_returns, label='OM Agent', color='green')
plt.xlabel('Training Episodes')
plt.ylabel('Average Eval Return')
plt.title(f'Agent Performance vs {args_parsed.opponent.capitalize()}')
plt.legend()

# Subplot 2: Opponent Evaluation Returns
plt.subplot(1, 3, 2)
plt.plot(episode_list, phase1_eval_opp_returns,
         label='Opponent (vs Classic)', color='red', linestyle='--')
plt.plot(episode_list, phase2_eval_opp_returns,
         label='Opponent (vs OM)', color='orange')
plt.xlabel('Training Episodes')
plt.ylabel('Average Eval Return')
plt.title(f'Heuristic Opponent Performance')
plt.legend()

# Subplot 3: Episode Steps
plt.subplot(1, 3, 3)
plt.plot(episode_list, phase1_eval_steps,
         label='Classic Agent', color='blue', linestyle='--')
plt.plot(episode_list, phase2_eval_steps, label='OM Agent', color='green')
plt.xlabel('Training Episodes')
plt.ylabel('Average Steps')
plt.title('Steps per Episode')
plt.legend()

plt.tight_layout()
plt.savefig(
  f"./diagrams/{args.folder_id}/comparative_training_{args_parsed.opponent}.png")
plt.close('all')

wandb.finish()
print("Pipeline complete. All models and charts saved.")
