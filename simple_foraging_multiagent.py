import os
import argparse
import copy
import random
from collections import deque, defaultdict
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import wandb
from tqdm import tqdm

from simple_foraging_env import SimpleForagingEnv, SimpleAgent, GreedySwitchAgent
import maps
from maps import *
from opponent_model import OpponentModel
from transformers import SpatialOpponentModel
from collect_data import collect_offline_data
from omg_args import OMGArgs

# Import the modified MA versions
from q_agent import QLearningAgent
from q_agent_classic import QLearningAgentClassic
from sl_agent import SLAgent

matplotlib.use('Agg')  # Prevents memory leak on headless clusters
torch.set_float32_matmul_precision('high')

map_layouts = [getattr(maps, m) for m in dir(maps) if m.startswith("MAP_")]

parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
parser.add_argument('--map', type=int, default=1, help='Map layout to use')
parser.add_argument('--episodes', type=int, default=12_000,
                    help='Total training episodes per phase')
parser.add_argument('--episodes_per_epoch', type=int,
                    default=500, help='Episodes per epoch')
parser.add_argument('--pretrain_epochs', type=int,
                    default=15, help='OM Pretraining epochs')
parser.add_argument('--max_steps', type=int, default=50,
                    help='Max steps per episode')
parser.add_argument('--batch_size', type=int, default=128,
                    help='Batch size for training')
parser.add_argument('--qnet_dim', type=int, default=256,
                    help='Q-network hidden dim')
parser.add_argument('--cnn_hidden', type=int,
                    default=64, help='CNN hidden dim')
parser.add_argument('--d_model', type=int, default=64,
                    help='Transformer d_model')
parser.add_argument('--nhead', type=int, default=4, help='Transformer heads')
parser.add_argument('--num_encoder_layers', type=int,
                    default=1, help='Transformer layers')
parser.add_argument('--dim_feedforward', type=int,
                    default=256, help='Transformer feedforward dim')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--tau_start', type=float, default=2.1,
                    help='Boltzmann start temp')
parser.add_argument('--tau_end', type=float, default=0.1,
                    help='Boltzmann end temp')
parser.add_argument('--train_every', type=int,
                    default=2, help='Train frequency')
parser.add_argument('--replay_capacity', type=int,
                    default=150_000, help='Replay buffer capacity')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--folder_id', type=int,
                    default=0, help='Output folder ID')
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

wandb.init(
    project="om-simple-foraging",
    config=args_parsed,
    name=f"map{args_parsed.map}_FSP_pipeline_id{args_parsed.folder_id}"
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
    state_shape=obs_sample[0].shape,
    H=H, W=W, action_dim=4,
    d_model=args_parsed.d_model, nhead=args_parsed.nhead,
    num_encoder_layers=args_parsed.num_encoder_layers,
    dim_feedforward=args_parsed.dim_feedforward, dropout=args_parsed.dropout,
)

num_epochs = args_parsed.episodes // args_parsed.episodes_per_epoch

# ==========================================
# PHASE 1: GENERATE CLASSIC FSP CURRICULUM
# ==========================================
print("\n--- PHASE 1: Training Classic Agent ---")
agent_classic = QLearningAgentClassic(env, args=args)
opp_classic = SLAgent(env, args=args)

phase1_returns, phase1_opp_returns, phase1_entropies = [], [], []


for epoch in range(num_epochs):
  epoch_returns, epoch_opp_returns, epoch_entropies = [], [], []
  pbar = tqdm(range(args_parsed.episodes_per_epoch),
              desc=f"Phase 1 Epoch {epoch + 1}/{num_epochs}", leave=False)

  for ep in pbar:
    stats = agent_classic.run_episode(opp_classic, max_steps=args.max_steps)
    epoch_returns.append(stats['return'])
    epoch_opp_returns.append(stats['opp_return'])
    epoch_entropies.append(stats['avg_entropy'])

    wandb.log(
      {"phase1/train_return": stats['return'],
       "phase1/opp_return": stats['opp_return'],
       "phase1/entropy": stats['avg_entropy']
       })

  avg_ret = sum(epoch_returns) / len(epoch_returns)
  avg_opp = sum(epoch_opp_returns) / len(epoch_opp_returns)
  avg_ent = sum(epoch_entropies) / len(epoch_entropies)
  phase1_returns.append(avg_ret)
  phase1_opp_returns.append(avg_opp)
  phase1_entropies.append(avg_ent)

  torch.save(agent_classic.q.state_dict(),
             f"./models/{args.folder_id}/classic_qnet_ep{epoch + 1}.pth")
  print(
    f"Phase 1 | Epoch {epoch + 1:02d} | Classic Ret: {avg_ret:>4.1f} | Opp Ret: {avg_opp:>4.1f} | Classic Entropy: {avg_ent:.4f} | Replay Size: {len(opp_classic.replay)}")

# ==========================================
# PHASE 2: TRAIN OM AGENT ON FROZEN CURRICULUM
# ==========================================
print("\n--- PHASE 2: Training OM Agent ---")
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

# FSP Opponent
opp_om = SLAgent(env, args=args)

phase2_returns, phase2_opp_returns, phase2_entropies = [], [], []

for epoch in range(num_epochs):
  epoch_returns, epoch_opp_returns, epoch_entropies = [], [], []
  pbar = tqdm(range(args_parsed.episodes_per_epoch),
              desc=f"Phase 2 Epoch {epoch + 1}/{num_epochs}", leave=False)

  for ep in pbar:
    stats = agent_om.run_episode(opp_om, max_steps=args.max_steps)
    epoch_returns.append(stats['return'])
    epoch_opp_returns.append(stats['opp_return'])
    epoch_entropies.append(stats['avg_entropy'])

    wandb.log(
      {"phase2/train_return": stats['return'],
       "phase2/opp_return": stats['opp_return'],
       "phase2/entropy": stats['avg_entropy']
       })

  avg_ret = sum(epoch_returns) / len(epoch_returns)
  avg_opp = sum(epoch_opp_returns) / len(epoch_opp_returns)
  avg_ent = sum(epoch_entropies) / len(epoch_entropies)
  phase2_returns.append(avg_ret)
  phase2_opp_returns.append(avg_opp)
  phase2_entropies.append(avg_ent)

  torch.save(agent_om.q.state_dict(),
             f"./models/{args.folder_id}/om_qnet_ep{epoch + 1}.pth")
  torch.save(agent_om.model.inference_model.state_dict(),
             f"./models/{args.folder_id}/om_inference_ep{epoch + 1}.pth")
  print(
    f"Phase 2 | Epoch {epoch + 1:02d} | OM Agent Ret: {avg_ret:>4.1f} | Opp Ret: {avg_opp:>4.1f} | OM Entropy: {avg_ent:.4f}")

# ==========================================
# PHASE 3: EVALUATION AGAINST HEURISTICS
# ==========================================
print("\n--- PHASE 3a: Headless Evaluation against Heuristics ---")
heuristics = {
    "Simple": SimpleAgent(agent_id=1),
    "GreedySwitch": GreedySwitchAgent(agent_id=1)
}
agent_classic.q.eval()
agent_om.q.eval()
agent_om.model.inference_model.eval()

eval_results = {}
total_eval_episodes = 1_000
for opp_name, heuristic_opp in heuristics.items():
  classic_eval_returns = []
  om_eval_returns = []

  for _ in range(total_eval_episodes):
    # Eval Classic
    c_stats = agent_classic.run_test_episode(
      heuristic_opp, max_steps=args.max_steps, render=False)
    classic_eval_returns.append(c_stats['return'])

    # Eval OM
    o_stats = agent_om.run_test_episode(
      heuristic_opp, max_steps=args.max_steps, render=False)
    om_eval_returns.append(o_stats['return'])

  c_avg = sum(classic_eval_returns) / total_eval_episodes
  o_avg = sum(om_eval_returns) / total_eval_episodes
  eval_results[opp_name] = {"Classic": c_avg, "OM": o_avg}

  print(
    f"Vs {opp_name:>12} | Classic Avg Ret: {c_avg:>5.2f} | OM Avg Ret: {o_avg:>5.2f}")
  wandb.log(
    {f"eval/classic_vs_{opp_name}": c_avg,
     f"eval/om_vs_{opp_name}": o_avg
     })

print("\n--- PHASE 3b: Head-to-Head Cross-Play ---")
classic_cross_returns = []
om_cross_returns = []
# ==========================================
# Matchup A: Classic (Player 0) vs OM (Player 1)
# ==========================================
for _ in range(total_eval_episodes):
  obs = env.reset()
  c_ret, o_ret = 0, 0
  history_len = args.max_history_length
  history = {
      "states": deque(maxlen=history_len),
      "actions": deque(maxlen=history_len)
  }
  for step in range(args.max_steps):
    current_history = {k: list(v) for k, v in history.items()}
    a_c, _ = agent_classic.select_action(obs[0], eval=True)
    a_o, _, _ = agent_om.select_action(obs[1], current_history, eval=True)
    obs, rewards, done, _ = env.step({0: a_c, 1: a_o})

    history["states"].append(obs[1])
    history["actions"].append(a_c)

    c_ret += rewards[0]
    o_ret += rewards[1]
    if done:
      break

  classic_cross_returns.append(c_ret)
  om_cross_returns.append(o_ret)

# ==========================================
# Matchup B: OM (Player 0) vs Classic (Player 1)
# ==========================================
for _ in range(total_eval_episodes):
  obs = env.reset()
  c_ret, o_ret = 0, 0
  history_len = args.max_history_length
  history = {
      "states": deque(maxlen=history_len),
      "actions": deque(maxlen=history_len)
  }
  for step in range(args.max_steps):
    current_history = {k: list(v) for k, v in history.items()}
    a_o, _, _ = agent_om.select_action(obs[0], current_history, eval=True)
    a_c, _ = agent_classic.select_action(obs[1], eval=True)
    obs, rewards, done, _ = env.step({1: a_c, 0: a_o})

    history["states"].append(obs[0])
    history["actions"].append(a_c)

    c_ret += rewards[1]
    o_ret += rewards[0]
    if done:
      break

  classic_cross_returns.append(c_ret)
  om_cross_returns.append(o_ret)

c_cross_avg = sum(classic_cross_returns) / (total_eval_episodes * 2)
o_cross_avg = sum(om_cross_returns) / (total_eval_episodes * 2)
eval_results["CrossPlay"] = {"Classic": c_cross_avg, "OM": o_cross_avg}

print(
  f"Head-to-Head | Classic Avg: {c_cross_avg:>5.2f} | OM Avg: {o_cross_avg:>5.2f}")
wandb.log({"eval/crossplay_classic": c_cross_avg,
          "eval/crossplay_om": o_cross_avg})
# ==========================================
# PLOTTING
# ==========================================
episode_list = [
  (i + 1) * args_parsed.episodes_per_epoch for i in range(num_epochs)]

# Use a rectangular size so the line charts aren't vertically squished
plt.figure(figsize=(16, 10))

# Subplot 1: Agent Return
plt.subplot(2, 2, 1)
plt.plot(episode_list, phase1_returns,
         label='Classic Agent', color='blue', linestyle='--')
plt.plot(episode_list, phase2_returns, label='OM Agent', color='green')
plt.xlabel('Episodes')
plt.ylabel('Average Return')
plt.title('Training Returns')
plt.legend()

# Subplot 2: Opponent Return
plt.subplot(2, 2, 2)
plt.plot(episode_list, phase1_opp_returns,
         label='Opponent (Phase 1)', color='red', linestyle='--')
plt.plot(episode_list, phase2_opp_returns,
         label='Opponent (Phase 2)', color='orange')
plt.xlabel('Episodes')
plt.ylabel('Average Return')
plt.title('Opponent Returns')
plt.legend()

# Subplot 3: Policy Entropy
plt.subplot(2, 2, 3)
plt.plot(episode_list, phase1_entropies,
         label='Classic Agent Entropy', color='blue', linestyle='--')
plt.plot(episode_list, phase2_entropies,
         label='OM Agent Entropy', color='green')
plt.xlabel('Episodes')
plt.ylabel('Shannon Entropy')
plt.title('Policy Entropy')
plt.legend()

# Subplot 4: Evaluation Returns vs All Heuristics
plt.subplot(2, 2, 4)
labels = list(eval_results.keys())  # This is ['Simple', 'GreedySwitch']
classic_wins = [eval_results[opp]['Classic'] for opp in labels]
om_wins = [eval_results[opp]['OM'] for opp in labels]

x = np.arange(len(labels))
width = 0.35
plt.bar(x - width / 2, classic_wins, width,
        label='Classic Agent', color='blue')
plt.bar(x + width / 2, om_wins, width, label='OM Agent', color='green')

plt.xlabel('Opponent Heuristic')
plt.ylabel('Average Return')
plt.title('Headless Evaluation: Classic vs OM')
plt.xticks(x, labels)
plt.legend()

plt.tight_layout()
plt.savefig(f"./diagrams/{args.folder_id}/fsp_comparative_training.png")
plt.close()

wandb.finish()
print("Pipeline complete. All models and charts saved.")
