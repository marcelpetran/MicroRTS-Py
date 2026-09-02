"""Team-based competitive exploration: training script.

Learning team (one shared Q-net + hostile/friendly OMs, QLearningAgent)
vs. a scripted greedy TeamAgent on a MovingAI benchmark map.

Run (from project root):
  /opt/homebrew/anaconda3/envs/om/bin/python scripts/train_team_exploration.py \
      --map den312d --episodes 3000

Defaults are sized for the 81x65 den312d map (see roadmap notes):
  max_history_length=8   (one collation = B x L x H x W x F floats)
  capacity=20000         (~200KB/transition in RAM)
  train_every=8, gamma=0.995, max_steps=400
"""

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import wandb
from omexplore.agents.q_agent import QLearningAgent
from omexplore.agents.team_agents import TeamAgent
from omexplore.envs.roadmap_foraging_env import TeamRoadmapEnv
from omexplore.models.opponent_model import OpponentModel
from omexplore.models.transformers import SpatialOpponentModel
from omexplore.utils.omg_args import OMGArgs

torch.set_float32_matmul_precision("high")

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
parser.add_argument("--map", type=str, default="den312d", help="MovingAI map name")
parser.add_argument(
    "--team_sizes",
    type=str,
    default="2,2",
    help="Comma-separated team sizes; team 0 learns, the rest are scripted",
)
parser.add_argument(
    "--num_goals", type=int, default=16, help="Goals in the shared pool"
)
parser.add_argument("--vision_radius", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument(
    "--episodes", type=int, default=3000, help="Total training episodes"
)
parser.add_argument(
    "--episodes_per_epoch", type=int, default=100, help="Episodes per logged epoch"
)
parser.add_argument("--eval_episodes", type=int, default=20)
# --- hyperparameters (scaled for the big map) ---
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--max_history_length", type=int, default=8)
parser.add_argument("--replay_capacity", type=int, default=20_000)
parser.add_argument("--min_replay", type=int, default=2_000)
parser.add_argument("--train_every", type=int, default=8)
parser.add_argument("--gamma", type=float, default=0.995)
parser.add_argument("--qnet_dim", type=int, default=256)
parser.add_argument("--cnn_hidden", type=int, default=64)
parser.add_argument("--d_model", type=int, default=64)
parser.add_argument("--nhead", type=int, default=4)
parser.add_argument("--num_encoder_layers", type=int, default=1)
parser.add_argument("--dim_feedforward", type=int, default=256)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--tau_start", type=float, default=2.1)
parser.add_argument("--tau_end", type=float, default=0.1)
parser.add_argument(
    "--tau_decay_steps", type=int, default=2_000_000, help="Global steps for tau decay"
)
parser.add_argument(
    "--device",
    type=str,
    default="auto",
    help="auto|cpu|cuda|mps (auto = cuda if available, else cpu)",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--folder_id", type=int, default=0)
parser.add_argument("--wandb_project", type=str, default="om-team-exploration")
parser.add_argument(
    "--no_wandb", action="store_true", help="Disable wandb (local logging only)"
)
args_parsed = parser.parse_args()

team_sizes = tuple(int(s) for s in args_parsed.team_sizes.split(","))

# Setup
os.makedirs(f"./models/{args_parsed.folder_id}", exist_ok=True)
os.makedirs(f"./diagrams/{args_parsed.folder_id}", exist_ok=True)

random.seed(args_parsed.seed)
np.random.seed(args_parsed.seed)
torch.manual_seed(args_parsed.seed)

if args_parsed.device == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = args_parsed.device
print(f"Using device: {device}")

env = TeamRoadmapEnv(
    map_name=args_parsed.map,
    max_steps=args_parsed.max_steps,
    vision_radius=args_parsed.vision_radius,
    num_goals=args_parsed.num_goals,
    team_sizes=team_sizes,
)
obs_sample = env.reset()

wandb.init(
    project=args_parsed.wandb_project,
    config={**vars(args_parsed), "team_sizes": team_sizes},
    name=f"{args_parsed.map}_{'v'.join(str(s) for s in team_sizes)}_id{args_parsed.folder_id}",
    mode="disabled" if args_parsed.no_wandb else None,
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
    gamma=args_parsed.gamma,
    tau_start=args_parsed.tau_start,
    tau_end=args_parsed.tau_end,
    tau_decay_steps=args_parsed.tau_decay_steps,
    state_shape=obs_sample[0].shape,
    H=env.height,
    W=env.width,
    action_dim=8,
    max_history_length=args_parsed.max_history_length,
    min_replay=args_parsed.min_replay,
    d_model=args_parsed.d_model,
    nhead=args_parsed.nhead,
    num_encoder_layers=args_parsed.num_encoder_layers,
    dim_feedforward=args_parsed.dim_feedforward,
    dropout=args_parsed.dropout,
)

hostile_om = OpponentModel(SpatialOpponentModel(args), args)
friendly_om = OpponentModel(SpatialOpponentModel(args), args)
agent = QLearningAgent(env, hostile_om, friendly_om, args=args)
opponent = TeamAgent(env, team_id=1)

print(f"learn_ids={agent.learn_ids} hostile_ids={agent.hostile_ids}")
print(
    f"map {env.height}x{env.width}, teams {team_sizes}, goals {args_parsed.num_goals}"
)

num_epochs = max(1, args_parsed.episodes // args_parsed.episodes_per_epoch)

# ==========================================
# TRAINING
# ==========================================
train_hist = {
    "returns": [],
    "opp_returns": [],
    "steps": [],
    "entropy": [],
    "q_loss": [],
    "model_loss": [],
    "team_model_loss": [],
    "coverage": [],
    "eval_returns": [],
    "eval_opp_returns": [],
    "eval_steps": [],
    "eval_kl": [],
    "eval_spatial": [],
    "eval_coverage": [],
    "eval_opp_coverage": [],
}

for epoch in range(num_epochs):
    ep_returns, ep_opp, ep_steps, ep_ent = [], [], [], []
    ep_q, ep_m, ep_tm = [], [], []

    pbar = tqdm(
        range(args_parsed.episodes_per_epoch),
        desc=f"Epoch {epoch + 1:02d}/{num_epochs} [Train]",
        leave=False,
    )
    for _ in pbar:
        stats = agent.run_episode(opponent, max_steps=args_parsed.max_steps)
        ep_returns.append(stats["return"])
        ep_opp.append(stats["opp_return"])
        ep_steps.append(stats["steps"])
        ep_ent.append(stats["avg_entropy"])
        ep_q.append(stats["avg_q_loss"])
        ep_m.append(stats["avg_model_loss"])
        ep_tm.append(stats["avg_team_model_loss"])
        pbar.set_postfix(
            ret=f"{stats['return']:.1f}",
            opp=f"{stats['opp_return']:.1f}",
            ql=f"{stats['avg_q_loss']:.3f}",
        )

    # Evaluation
    ev_rets, ev_opp, ev_steps, ev_kl, ev_sp = [], [], [], [], []
    ev_cov, ev_opp_cov = [], []
    for _ in range(args_parsed.eval_episodes):
        t = agent.run_test_episode(opponent, max_steps=args_parsed.max_steps)
        ev_rets.append(t["return"])
        ev_opp.append(t["opp_return"])
        ev_steps.append(t["steps"])
        if t["avg_kl_error"] is not None:
            ev_kl.append(t["avg_kl_error"])
        if t["avg_spatial_error"] is not None:
            ev_sp.append(t["avg_spatial_error"])
        # coverage is read after the episode ended (env state is post-terminal)
        ev_cov.append(env.get_coverage(0))
        ev_opp_cov.append(env.get_coverage(1))

    def _avg(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else 0.0

    avg = {
        "train_return": _avg(ep_returns),
        "train_opp_return": _avg(ep_opp),
        "train_steps": _avg(ep_steps),
        "train_entropy": _avg(ep_ent),
        "train_q_loss": _avg(ep_q),
        "train_model_loss": _avg(ep_m),
        "train_team_model_loss": _avg(ep_tm),
        "eval_return": _avg(ev_rets),
        "eval_opp_return": _avg(ev_opp),
        "eval_steps": _avg(ev_steps),
        "eval_kl_error": _avg(ev_kl),
        "eval_spatial_error": _avg(ev_sp),
        "eval_coverage": _avg(ev_cov),
        "eval_opp_coverage": _avg(ev_opp_cov),
    }

    train_hist["returns"].append(avg["train_return"])
    train_hist["opp_returns"].append(avg["train_opp_return"])
    train_hist["steps"].append(avg["train_steps"])
    train_hist["entropy"].append(avg["train_entropy"])
    train_hist["q_loss"].append(avg["train_q_loss"])
    train_hist["model_loss"].append(avg["train_model_loss"])
    train_hist["team_model_loss"].append(avg["train_team_model_loss"])
    train_hist["eval_returns"].append(avg["eval_return"])
    train_hist["eval_opp_returns"].append(avg["eval_opp_return"])
    train_hist["eval_steps"].append(avg["eval_steps"])
    train_hist["eval_kl"].append(avg["eval_kl_error"])
    train_hist["eval_spatial"].append(avg["eval_spatial_error"])
    train_hist["eval_coverage"].append(avg["eval_coverage"])
    train_hist["eval_opp_coverage"].append(avg["eval_opp_coverage"])

    wandb.log({f"team/{k}": v for k, v in avg.items()} | {"epoch": epoch + 1})

    torch.save(agent.q.state_dict(), f"./models/{args.folder_id}/team_qnet.pth")
    torch.save(
        agent.model.inference_model.state_dict(),
        f"./models/{args.folder_id}/team_hostile_om.pth",
    )
    torch.save(
        agent.team_model.inference_model.state_dict(),
        f"./models/{args.folder_id}/team_friendly_om.pth",
    )

    print(
        f"Epoch {epoch + 1:02d} | Train Ret {avg['train_return']:>5.2f} "
        f"(opp {avg['train_opp_return']:.2f}) | Eval Ret {avg['eval_return']:>5.2f} "
        f"(opp {avg['eval_opp_return']:.2f}) | Cov {avg['eval_coverage']:.3f} "
        f"(opp {avg['eval_opp_coverage']:.3f}) | Q {avg['train_q_loss']:.3f} "
        f"| OM {avg['train_model_loss']:.3f} | tOM {avg['train_team_model_loss']:.3f}"
    )

# ==========================================
# PLOTTING
# ==========================================
print("\n--- Generating Evaluation Charts ---")
epochs = [(i + 1) * args_parsed.episodes_per_epoch for i in range(num_epochs)]

plt.figure(figsize=(18, 10))

plt.subplot(2, 3, 1)
plt.plot(epochs, train_hist["eval_returns"], label="Learning team", color="green")
plt.plot(epochs, train_hist["eval_opp_returns"], label="Greedy team", color="red")
plt.xlabel("Training Episodes")
plt.ylabel("Average Eval Return")
plt.title(f"Team Scores ({args_parsed.map})")
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(epochs, train_hist["eval_coverage"], label="Learning team", color="green")
plt.plot(epochs, train_hist["eval_opp_coverage"], label="Greedy team", color="red")
plt.xlabel("Training Episodes")
plt.ylabel("Fraction of map seen")
plt.title("Exploration Coverage")
plt.legend()

plt.subplot(2, 3, 3)
plt.plot(epochs, train_hist["entropy"], color="purple")
plt.xlabel("Training Episodes")
plt.ylabel("Avg Entropy")
plt.title("Policy Entropy")

plt.subplot(2, 3, 4)
plt.plot(epochs, train_hist["q_loss"], label="Q loss", color="blue")
plt.xlabel("Training Episodes")
plt.ylabel("Loss")
plt.title("Q-Learning Loss")
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(epochs, train_hist["model_loss"], label="Hostile OM", color="orange")
plt.plot(epochs, train_hist["team_model_loss"], label="Friendly OM", color="brown")
plt.xlabel("Training Episodes")
plt.ylabel("Loss")
plt.title("Opponent Model Losses")
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(epochs, train_hist["eval_kl"], label="KL error", color="teal")
plt.plot(epochs, train_hist["eval_spatial"], label="Spatial error", color="gray")
plt.xlabel("Training Episodes")
plt.ylabel("Error")
plt.title("Hostile OM Prediction Quality")
plt.legend()

plt.tight_layout()
plt.savefig(f"./diagrams/{args.folder_id}/team_training_{args_parsed.map}.png")
plt.close("all")

wandb.finish()
print("Training complete. Models and charts saved.")
