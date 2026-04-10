import os
import argparse
import random
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

# Import the new unified FSP versions
from slq_agent import FSPAgentOM
from slq_agent_classic import FSPAgentClassic

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
parser.add_argument('--tau_decay_steps', type=int, default=600_000,
                    help='Boltzmann decay steps')
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
    tau_decay_steps=args_parsed.tau_decay_steps,
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
print("\n--- PHASE 1: Training Classic FSP Agent ---")
# FSP agent contains both RL and SL, and plays against itself
agent_classic = FSPAgentClassic(env, args=args)

phase1_returns, phase1_opp_returns, phase1_rl_entropies, phase1_sl_entropies, phase1_steps = [], [], [], [], []

for epoch in range(num_epochs):
  # Calculate mixing parameter
  eta = 0.1
  epoch_returns, epoch_opp_returns, epoch_rl_entropies, epoch_sl_entropies, epoch_steps = [], [], [], [], []

  # 1. DATA GENERATION PHASE
  pbar = tqdm(range(args_parsed.episodes_per_epoch),
              desc=f"Phase 1 Epoch {epoch + 1}/{num_epochs}", leave=False)

  for ep in pbar:
    # Agent self-plays against itself to build the curriculum
    stats = agent_classic.run_fsp_episode(
      opponent_agent=agent_classic, eta=eta, max_steps=args.max_steps)
    epoch_returns.append(stats['return'])
    epoch_opp_returns.append(stats['opp_return'])
    epoch_rl_entropies.append(stats['avg_rl_entropy'])
    epoch_sl_entropies.append(stats['avg_sl_entropy'])
    epoch_steps.append(stats['steps'])
    steps_taken = stats['steps']
    updates = steps_taken // args.train_every
    for _ in range(updates):
      agent_classic.update_rl()
      agent_classic.update_sl()

  avg_ret = sum(epoch_returns) / len(epoch_returns)
  avg_opp = sum(epoch_opp_returns) / len(epoch_opp_returns)
  avg_rl_ent = sum(epoch_rl_entropies) / len(epoch_rl_entropies)
  avg_sl_ent = sum(epoch_sl_entropies) / len(epoch_sl_entropies)
  avg_steps = sum(epoch_steps) / len(epoch_steps)

  phase1_returns.append(avg_ret)
  phase1_opp_returns.append(avg_opp)
  phase1_rl_entropies.append(avg_rl_ent)
  phase1_sl_entropies.append(avg_sl_ent)
  phase1_steps.append(avg_steps)

  wandb.log({
      "phase1/train_return": avg_ret,
      "phase1/opp_return": avg_opp,
      "phase1/avg_steps": avg_steps,
      "phase1/rl_entropy": avg_rl_ent,
      "phase1/sl_entropy": avg_sl_ent,
      "phase1/eta": eta
  })

  torch.save(agent_classic.q.state_dict(),
             f"./models/{args.folder_id}/classic_qnet_ep{epoch + 1}.pth")
  torch.save(agent_classic.sl.state_dict(),
             f"./models/{args.folder_id}/classic_slnet_ep{epoch + 1}.pth")

  print(f"Phase 1 | Epoch {epoch + 1:02d} | Classic Ret: {avg_ret:>4.1f} | Opp Ret: {avg_opp:>4.1f} | RL Entropy: {avg_rl_ent:.4f} | SL Entropy: {avg_sl_ent:.4f}| SL Buffer: {len(agent_classic.sl_replay)}")


# ==========================================
# PHASE 2: TRAIN OM AGENT
# ==========================================
print("\n--- PHASE 2: Training OM Agent ---")
inference_model = SpatialOpponentModel(args=args).to(device)
op_model = OpponentModel(inference_model, args=args)
agent_om = FSPAgentOM(env, op_model, args=args)

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

phase2_returns, phase2_opp_returns, phase2_rl_entropies, phase2_sl_entropies, phase2_steps = [], [], [], [], []

for epoch in range(num_epochs):
  eta = 0.1
  epoch_returns, epoch_opp_returns, epoch_rl_entropies, epoch_sl_entropies, epoch_steps = [], [], [], [], []

  # 1. DATA GENERATION PHASE
  pbar = tqdm(range(args_parsed.episodes_per_epoch),
              desc=f"Phase 2 Epoch {epoch + 1}/{num_epochs}", leave=False)

  for ep in pbar:
    # FSP OM Agent mixes between RL and SL.
    stats = agent_om.run_fsp_episode(
      opponent_agent=agent_om, eta=eta, max_steps=args.max_steps)
    epoch_returns.append(stats['return'])
    epoch_opp_returns.append(stats['opp_return'])
    epoch_rl_entropies.append(stats['avg_rl_entropy'])
    epoch_sl_entropies.append(stats['avg_sl_entropy'])
    epoch_steps.append(stats['steps'])
    steps_taken = stats['steps']
    updates = steps_taken // args.train_every
    for _ in range(updates):
      agent_om.update_rl()
      agent_om.update_sl()

  avg_ret = sum(epoch_returns) / len(epoch_returns)
  avg_opp = sum(epoch_opp_returns) / len(epoch_opp_returns)
  avg_rl_ent = sum(epoch_rl_entropies) / len(epoch_rl_entropies)
  avg_sl_ent = sum(epoch_sl_entropies) / len(epoch_sl_entropies)
  avg_steps = sum(epoch_steps) / len(epoch_steps)

  phase2_returns.append(avg_ret)
  phase2_opp_returns.append(avg_opp)
  phase2_rl_entropies.append(avg_rl_ent)
  phase2_sl_entropies.append(avg_sl_ent)
  phase2_steps.append(avg_steps)

  wandb.log({
      "phase2/train_return": avg_ret,
      "phase2/opp_return": avg_opp,
      "phase2/avg_steps": avg_steps,
      "phase2/rl_entropy": avg_rl_ent,
      "phase2/sl_entropy": avg_sl_ent,
      "phase2/eta": eta
  })

  torch.save(agent_om.q.state_dict(),
             f"./models/{args.folder_id}/om_qnet_ep{epoch + 1}.pth")
  torch.save(agent_om.sl.state_dict(),
             f"./models/{args.folder_id}/om_slnet_ep{epoch + 1}.pth")
  torch.save(agent_om.model.inference_model.state_dict(),
             f"./models/{args.folder_id}/om_inference_ep{epoch + 1}.pth")

  print(
    f"Phase 2 | Epoch {epoch + 1:02d} | OM Agent Ret: {avg_ret:>4.1f} | Opp Ret: {avg_opp:>4.1f} | RL Entropy: {avg_rl_ent:.4f} | SL Entropy: {avg_sl_ent:.4f} | SL Buffer: {len(agent_om.sl_replay)}")


# ==========================================
# PHASE 3: EVALUATION
# ==========================================
print("\n--- PHASE 3: Evaluation ---")
agent_classic.q.eval()
agent_classic.sl.eval()
agent_om.q.eval()
agent_om.sl.eval()
agent_om.model.inference_model.eval()


def evaluate_matchup(agent0, agent1, env, args, use_sl=False):
  """
  Safely executes a matchup, handling the rolling GPU history correctly 
  for whichever agent contains the Opponent Modeling architecture.
  """
  agent0_is_om = isinstance(agent0, FSPAgentOM)
  agent1_is_om = isinstance(agent1, FSPAgentOM)

  if agent0_is_om:
    hist_len0 = agent0.args.max_history_length
    rf0 = torch.zeros((1, hist_len0, agent0.args.d_model),
                      device=agent0.device)
    ra0 = torch.zeros((1, hist_len0), dtype=torch.long, device=agent0.device)
    rm0 = torch.zeros((1, hist_len0), dtype=torch.bool, device=agent0.device)
    c_seq0 = 0

  if agent1_is_om:
    hist_len1 = agent1.args.max_history_length
    rf1 = torch.zeros((1, hist_len1, agent1.args.d_model),
                      device=agent1.device)
    ra1 = torch.zeros((1, hist_len1), dtype=torch.long, device=agent1.device)
    rm1 = torch.zeros((1, hist_len1), dtype=torch.bool, device=agent1.device)
    c_seq1 = 0

  obs = env.reset()
  agent0.reset()
  agent1.reset()
  ret0, ret1 = 0, 0

  for step in range(args.max_steps):
    # Agent 0 action
    if agent0_is_om:
      h0 = {"state_features": rf0, "actions": ra0, "mask": rm0}
      if use_sl:
        a0, _ = agent0.select_sl_action(obs[0], eval=True)
      else:
        a0, _, _ = agent0.select_rl_action(obs[0], h0, eval=True)
    else:
      if use_sl and hasattr(agent0, 'select_sl_action'):
        a0, _ = agent0.select_sl_action(obs[0], eval=True)
      else:
        a0, _ = agent0.select_rl_action(obs[0], eval=True) if hasattr(
          agent0, 'select_rl_action') else agent0.select_action(obs[0])

    # Agent 1 action
    if agent1_is_om:
      h1 = {"state_features": rf1, "actions": ra1, "mask": rm1}
      if use_sl:
        a1, _ = agent1.select_sl_action(obs[1], eval=True)
      else:
        a1, _, _ = agent1.select_rl_action(obs[1], h1, eval=True)
    else:
      if use_sl and hasattr(agent1, 'select_sl_action'):
        a1, _ = agent1.select_sl_action(obs[1], eval=True)
      else:
        a1, _ = agent1.select_rl_action(obs[1], eval=True) if hasattr(
          agent1, 'select_rl_action') else agent1.select_action(obs[1])

    next_obs, rewards, done, _ = env.step({0: a0, 1: a1})

    # Update OM histories
    if agent0_is_om:
      s_tens = torch.from_numpy(obs[0]).float().unsqueeze(0).to(agent0.device)
      with torch.no_grad():
        nf = agent0.model.inference_model.get_features(s_tens)
      rf0 = torch.roll(rf0, shifts=-1, dims=1)
      ra0 = torch.roll(ra0, shifts=-1, dims=1)
      rm0 = torch.roll(rm0, shifts=-1, dims=1)
      rf0[:, -1, :] = nf
      ra0[:, -1] = a1  # opponent action
      if c_seq0 < hist_len0:
        c_seq0 += 1
        rm0[:, -c_seq0:] = True

    if agent1_is_om:
      s_tens = torch.from_numpy(obs[1]).float().unsqueeze(0).to(agent1.device)
      with torch.no_grad():
        nf = agent1.model.inference_model.get_features(s_tens)
      rf1 = torch.roll(rf1, shifts=-1, dims=1)
      ra1 = torch.roll(ra1, shifts=-1, dims=1)
      rm1 = torch.roll(rm1, shifts=-1, dims=1)
      rf1[:, -1, :] = nf
      ra1[:, -1] = a0  # opponent action
      if c_seq1 < hist_len1:
        c_seq1 += 1
        rm1[:, -c_seq1:] = True

    obs = next_obs
    ret0 += rewards[0]
    ret1 += rewards[1]
    if done:
      break

  return ret0, ret1


print("\n--- PHASE 3a: Headless Evaluation against Heuristics ---")
heuristics = {
    "Simple": SimpleAgent(agent_id=1),
    "GreedySwitch": GreedySwitchAgent(agent_id=1)
}

eval_results = {}
total_eval_episodes = 1_000

for opp_name, heuristic_opp in heuristics.items():
  classic_eval_returns, om_eval_returns = [], []

  for _ in range(total_eval_episodes):
    # Evaluate Average Strategy/Nash Equilibrium (use_sl=True) against heuristics
    c_ret, _ = evaluate_matchup(
      agent_classic, heuristic_opp, env, args, use_sl=True)
    o_ret, _ = evaluate_matchup(
      agent_om, heuristic_opp, env, args, use_sl=True)
    classic_eval_returns.append(c_ret)
    om_eval_returns.append(o_ret)

  c_avg = sum(classic_eval_returns) / total_eval_episodes
  o_avg = sum(om_eval_returns) / total_eval_episodes
  eval_results[opp_name] = {"Classic": c_avg, "OM": o_avg}

  print(
    f"Vs {opp_name:>12} | Classic Avg Ret: {c_avg:>5.2f} | OM Avg Ret: {o_avg:>5.2f}")
  wandb.log({
      f"eval/classic_vs_{opp_name}": c_avg,
      f"eval/om_vs_{opp_name}": o_avg
  })

print("\n--- PHASE 3b: Head-to-Head Cross-Play ---")
classic_cross_returns = []
om_cross_returns = []

for _ in range(total_eval_episodes):
  # Matchup A: Classic (0) vs OM (1) - Evaluating Average Strategies
  c_ret, o_ret = evaluate_matchup(
    agent_classic, agent_om, env, args, use_sl=True)
  classic_cross_returns.append(c_ret)
  om_cross_returns.append(o_ret)

  # Matchup B: OM (0) vs Classic (1)
  o_ret, c_ret = evaluate_matchup(
    agent_om, agent_classic, env, args, use_sl=True)
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
plt.plot(episode_list, phase1_rl_entropies,
         label='Classic Agent RL Entropy', color='blue', linestyle='--')
plt.plot(episode_list, phase1_sl_entropies,
         label='Classic Agent SL Entropy', color='red', linestyle='--')
plt.plot(episode_list, phase2_rl_entropies,
         label='OM Agent RL Entropy', color='green')
plt.plot(episode_list, phase2_sl_entropies,
         label='OM Agent SL Entropy', color='orange')
plt.xlabel('Episodes')
plt.ylabel('Shannon Entropy')
plt.title('Policy Entropy')
plt.legend()

# Subplot 4: Evaluation Returns vs All Heuristics
plt.subplot(2, 2, 4)
labels = list(eval_results.keys())
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
