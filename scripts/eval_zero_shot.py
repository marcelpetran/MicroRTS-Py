"""Zero-shot evaluation of trained 1v1 checkpoints against a held-out opponent.

Loads the classic and OM checkpoints produced by scripts/train_1v1.py from one
or more training folders (seeds), evaluates both against the chosen heuristic
opponent with per-episode seat swapping, and reports per-seed and aggregated
mean+-std with a per-seat breakdown.

Usage (MAP_4, vision 2, belief prior on, 3 seeds trained vs greedy):
    python scripts/eval_zero_shot.py --map 4 --opponent stalker \
        --folder_ids 11695749,11698xxx,1170xxxx \
        --vision_radius 2 --belief_map_prior --episodes 1000 --swap_prob 0.5

IMPORTANT: --vision_radius / --belief_map_prior / net-dim flags must match the
training run (they define the checkpoint architectures and input shapes). The
run folders are named map{m}_{opp}_v{vision}_p{prior}_n{nstep}_s{seed}_id{id}.
"""

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from omexplore.agents.q_agent import QLearningAgent
from omexplore.agents.q_agent_classic import QLearningAgentClassic
from omexplore.envs.simple_foraging_env import (
    ChameleonAgent,
    GreedySwitchAgent,
    OracleStalkerAgent,
    SimpleAgent,
    SimpleForagingEnv,
    StalkerAgent,
)
from omexplore.models.opponent_model import OpponentModel
from omexplore.models.transformers import SpatialOpponentModel
from omexplore.utils import maps as maps
from omexplore.utils.omg_args import OMGArgs

map_layouts = [getattr(maps, m) for m in dir(maps) if m.startswith("MAP_")]

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
parser.add_argument(
    "--opponent",
    type=str,
    default="stalker",
    choices=["simple", "greedy", "stalker", "oracle_stalker", "chameleon"],
    help="Held-out heuristic opponent to evaluate against (oracle_stalker = "
    "thesis-faithful deceptive adversary with full observability)",
)
parser.add_argument(
    "--map", type=int, default=4, choices=[i for i in range(1, len(map_layouts) + 1)]
)
parser.add_argument(
    "--folder_ids",
    type=str,
    required=True,
    help="Comma-separated training folder ids (one per seed), e.g. 11695749,11696000",
)
parser.add_argument(
    "--epoch",
    type=int,
    default=60,
    help="Checkpoint epoch to load (default: latest in the folder)",
)
parser.add_argument(
    "--episodes", type=int, default=1000, help="Eval episodes per agent per seed"
)
parser.add_argument("--max_steps", type=int, default=50)
parser.add_argument(
    "--swap_prob", type=float, default=0.5, help="Per-episode seat-swap probability"
)
parser.add_argument("--seed", type=int, default=0)

# --- Architecture / input flags: must mirror train_1v1.py defaults ---
parser.add_argument("--qnet_dim", type=int, default=256)
parser.add_argument("--cnn_hidden", type=int, default=64)
parser.add_argument("--d_model", type=int, default=64)
parser.add_argument("--nhead", type=int, default=4)
parser.add_argument("--num_encoder_layers", type=int, default=1)
parser.add_argument("--dim_feedforward", type=int, default=256)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--tau_start", type=float, default=2.1)
parser.add_argument("--tau_end", type=float, default=0.1)
parser.add_argument("--tau_decay_steps", type=int, default=600_000)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--replay_capacity", type=int, default=150_000)
parser.add_argument("--train_every", type=int, default=2)
parser.add_argument(
    "--n_step", type=int, default=1, help="Must match training; unused at eval time"
)
parser.add_argument("--vision_radius", type=int, default=2)
parser.add_argument(
    "--belief_map_prior",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Must match training; seeds the learner's belief channels",
)
args_parsed = parser.parse_args()

random.seed(args_parsed.seed)
np.random.seed(args_parsed.seed)
torch.manual_seed(args_parsed.seed)


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

env = SimpleForagingEnv(
    max_steps=args_parsed.max_steps,
    map_layout=map_layouts[args_parsed.map - 1],
    vision_radius=args_parsed.vision_radius,
)

obs_sample = env.reset()
H, W, F_dim = obs_sample[0].shape
NUM_ACTIONS = 4

args = OMGArgs(
    device=device,
    folder_id=0,
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
    H=H,
    W=W,
    action_dim=NUM_ACTIONS,
    d_model=args_parsed.d_model,
    nhead=args_parsed.nhead,
    num_encoder_layers=args_parsed.num_encoder_layers,
    dim_feedforward=args_parsed.dim_feedforward,
    dropout=args_parsed.dropout,
    true_intent=False,
    n_step=args_parsed.n_step,
    friendly_om=False,
    belief_map_prior=args_parsed.belief_map_prior,
)


def make_opponent():
    if args_parsed.opponent == "simple":
        return SimpleAgent(agent_id=1, map_layout=map_layouts[args_parsed.map - 1])
    if args_parsed.opponent == "greedy":
        return GreedySwitchAgent(
            agent_id=1, map_layout=map_layouts[args_parsed.map - 1]
        )
    if args_parsed.opponent == "stalker":
        return StalkerAgent(agent_id=1, map_layout=map_layouts[args_parsed.map - 1])
    if args_parsed.opponent == "oracle_stalker":
        return OracleStalkerAgent(
            agent_id=1, env=env, map_layout=map_layouts[args_parsed.map - 1]
        )
    return ChameleonAgent(agent_id=1, map_layout=map_layouts[args_parsed.map - 1])


def latest_epoch(folder: str, prefix: str) -> int:
    ckpts = glob.glob(os.path.join(folder, f"{prefix}_ep*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matching {prefix}_ep*.pth in {folder}")
    return max(int(c.rsplit("ep", 1)[1].split(".")[0]) for c in ckpts)


def summarize(rets: list) -> dict:
    a = np.asarray(rets, dtype=float)
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
        "se": float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0,
        "n": len(a),
    }


folder_ids = [f.strip() for f in args_parsed.folder_ids.split(",") if f.strip()]
results = {"config": vars(args_parsed), "seeds": [], "agents": {}, "paired": {}}

for fid in folder_ids:
    folder = f"./models/{fid}"
    epoch = args_parsed.epoch or latest_epoch(folder, "classic_qnet")
    print(f"\n=== Seed folder {fid} (epoch {epoch}) ===")

    # ---- Classic ----
    agent_classic = QLearningAgentClassic(env, args=args)
    agent_classic.q.load_state_dict(
        torch.load(
            f"{folder}/classic_qnet_ep{epoch}.pth",
            map_location=device,
            weights_only=True,
        )
    )
    agent_classic.q.eval()

    opp = make_opponent()
    classic_stats = []
    for _ in range(args_parsed.episodes):
        classic_stats.append(
            agent_classic.run_test_episode(
                opp, max_steps=args_parsed.max_steps, swap_prob=args_parsed.swap_prob
            )
        )
    classic_rets = [s["return"] for s in classic_stats]
    classic_opp_rets = [s["opp_return"] for s in classic_stats]
    classic_seat = {
        "seat_A": summarize([s["return"] for s in classic_stats if not s["swapped"]]),
        "seat_B": summarize([s["return"] for s in classic_stats if s["swapped"]]),
    }

    # ---- OM ----
    inference_model = SpatialOpponentModel(args=args).to(device)
    inference_model.load_state_dict(
        torch.load(
            f"{folder}/om_inference_ep{epoch}.pth",
            map_location=device,
            weights_only=True,
        )
    )
    op_model = OpponentModel(inference_model, args=args)
    agent_om = QLearningAgent(env, op_model, args=args)
    agent_om.q.load_state_dict(
        torch.load(
            f"{folder}/om_qnet_ep{epoch}.pth", map_location=device, weights_only=True
        )
    )
    agent_om.q.eval()

    opp = make_opponent()
    om_stats = []
    for _ in range(args_parsed.episodes):
        om_stats.append(
            agent_om.run_test_episode(
                opp, max_steps=args_parsed.max_steps, swap_prob=args_parsed.swap_prob
            )
        )
    om_rets = [s["return"] for s in om_stats]
    om_opp_rets = [s["opp_return"] for s in om_stats]
    om_seat = {
        "seat_A": summarize([s["return"] for s in om_stats if not s["swapped"]]),
        "seat_B": summarize([s["return"] for s in om_stats if s["swapped"]]),
    }
    om_kl = [s["avg_kl_error"] for s in om_stats if s["avg_kl_error"] is not None]

    results["seeds"].append(
        {
            "folder_id": fid,
            "epoch": epoch,
            "classic": {
                "return": summarize(classic_rets),
                "opp_return": summarize(classic_opp_rets),
                "seats": classic_seat,
            },
            "om": {
                "return": summarize(om_rets),
                "opp_return": summarize(om_opp_rets),
                "seats": om_seat,
                "kl": summarize(om_kl) if om_kl else None,
            },
        }
    )

    print(
        f"  classic: {np.mean(classic_rets):.3f} +- {np.std(classic_rets, ddof=1):.3f} "
        f"(opp {np.mean(classic_opp_rets):.3f}) | "
        f"seat A {classic_seat['seat_A']['mean']:.3f} / seat B {classic_seat['seat_B']['mean']:.3f}"
    )
    print(
        f"  om:     {np.mean(om_rets):.3f} +- {np.std(om_rets, ddof=1):.3f} "
        f"(opp {np.mean(om_opp_rets):.3f}) | "
        f"seat A {om_seat['seat_A']['mean']:.3f} / seat B {om_seat['seat_B']['mean']:.3f}"
    )

# ---- Aggregate across seeds ----
classic_means = [s["classic"]["return"]["mean"] for s in results["seeds"]]
om_means = [s["om"]["return"]["mean"] for s in results["seeds"]]
results["agents"] = {
    "classic": summarize(classic_means),
    "om": summarize(om_means),
    "classic_opp": summarize(
        [s["classic"]["opp_return"]["mean"] for s in results["seeds"]]
    ),
    "om_opp": summarize([s["om"]["opp_return"]["mean"] for s in results["seeds"]]),
}
if len(folder_ids) > 1:
    diffs = [o - c for o, c in zip(om_means, classic_means)]
    results["paired"] = {"om_minus_classic": summarize(diffs)}

print(
    f"\n=== AGGREGATE over {len(folder_ids)} seed(s) vs {args_parsed.opponent.upper()} "
    f"(swap_prob={args_parsed.swap_prob}, {args_parsed.episodes} eps each) ==="
)
print(
    f"classic: {np.mean(classic_means):.3f} +- {np.std(classic_means, ddof=1) if len(classic_means) > 1 else 0:.3f}"
)
print(
    f"om:     {np.mean(om_means):.3f} +- {np.std(om_means, ddof=1) if len(om_means) > 1 else 0:.3f}"
)
if results["paired"]:
    d = results["paired"]["om_minus_classic"]
    print(f"paired om-classic: {d['mean']:.3f} +- {d['std']:.3f}")

out_dir = "./eval_results"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(
    out_dir,
    f"map{args_parsed.map}_{args_parsed.opponent}_v{args_parsed.vision_radius}"
    f"_p{int(args_parsed.belief_map_prior)}_n{args_parsed.n_step}_sw{args_parsed.swap_prob}.json",
)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to {out_path}")
