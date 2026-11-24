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
from omexplore.agents.q_agent_classic_h import QLearningAgentClassicH
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
parser.add_argument(
    "--blind_om",
    action="store_true",
    default=False,
    help="Disable the opponent model (zero out the fused heatmap during eval)",
)
parser.add_argument(
    "--classic_h",
    action="store_true",
    default=False,
    help="Evaluate the history-matched Classic-H baseline instead of the plain "
    "Classic agent: expects classic_qnet_ep*.pth AND "
    "classic_h_encoder_ep*.pth in each folder. OM is evaluated as usual "
    "and the paired column is OM minus Classic-H.",
)
parser.add_argument(
    "--om_control",
    type=str,
    default="none",
    choices=["none", "blind", "shuffled", "uniform", "random_window"],
    help="Ablation control on the OM heatmap at eval time (no retraining): "
    "blind = zero heatmap; shuffled = permute H*W cells each step (same "
    "magnitudes, content spatially decoupled) — R3's content-vs-channel "
    "control; uniform = uniform free-cell distribution; "
    "random_window = heatmap from an unrelated trajectory. "
    "('--blind_om' is an alias for --om_control=blind.)",
)
args_parsed = parser.parse_args()
if args_parsed.blind_om and args_parsed.om_control == "none":
    args_parsed.om_control = "blind"

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
    blind_om=args_parsed.blind_om,
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


class _UnrelatedPool:
    """Heatmaps from an UNRELATED trajectory distribution (R3 random_window
    control). Stateless generator: for each draw it refills with a fresh random
    observation history forwarded through the loaded transformer so Q consumes
    a real, valid heatmap that has no correspondence with the live episode."""

    def __init__(self, agent_om, env, H, W, seed):
        self.om = agent_om.model.inference_model
        self.args = agent_om.args
        self.device = agent_om.device
        self.env = env
        self.H, self.W = H, W
        self.rng = np.random.default_rng(seed)
        self.be_library = []

    def fill(self, n=32):
        """Produce n unrelated heatmaps by advancing the real feature extractor
        through fresh env resets (valid learned representations, no live-episode
        correspondence)."""
        import torch.nn.functional as _F

        for _ in range(n):
            obs = self.env.reset()
            T = self.args.max_history_length
            rf = torch.zeros(1, T, self.args.d_model, device=self.device)
            mask = torch.zeros(1, T, dtype=torch.bool, device=self.device)
            L = self.rng.integers(3, T + 1)
            mask[0, -L:] = True
            prev = torch.zeros(1, *obs[0].shape, device=self.device)
            x = torch.from_numpy(obs[0]).float().unsqueeze(0).to(self.device)
            feat = self.om.get_features(x, prev)
            rf[0, -L:] = feat
            hist = {"state_features": rf, "mask": mask, "prev_obs": prev}
            with torch.no_grad():
                gl = self.om(x, hist)
                g = _F.softmax(gl.view(gl.shape[0], -1), dim=-1).view_as(gl)
            self.be_library.append(g.cpu())


folder_ids = [f.strip() for f in args_parsed.folder_ids.split(",") if f.strip()]
results = {"config": vars(args_parsed), "seeds": [], "agents": {}, "paired": {}}

for fid in folder_ids:
    folder = f"./models/{fid}"
    epoch = args_parsed.epoch or latest_epoch(folder, "classic_qnet")
    print(f"\n=== Seed folder {fid} (epoch {epoch}) ===")

    # ---- Classic or history-matched Classic-H ----
    if args_parsed.classic_h:
        _inf = SpatialOpponentModel(args=args).to(device)
        _op_model = OpponentModel(_inf, args=args)
        agent_classic = QLearningAgentClassicH(env, _op_model, args=args)
        agent_classic.encoder.load_state_dict(
            torch.load(
                f"{folder}/classic_h_encoder_ep{epoch}.pth",
                map_location=device,
                weights_only=True,
            )
        )
    else:
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

    # The new classic-only training runs don't contain OM artifacts. Keep
    # those evaluations usable without requiring a phantom Phase-2 model.
    _has_om = os.path.exists(f"{folder}/om_qnet_ep{epoch}.pth") and os.path.exists(
        f"{folder}/om_inference_ep{epoch}.pth"
    )
    if not _has_om:
        if not args_parsed.classic_h:
            raise FileNotFoundError(
                f"Missing OM checkpoints in {folder} at epoch {epoch}"
            )
        results["seeds"].append(
            {
                "folder_id": fid,
                "epoch": epoch,
                "classic": {
                    "return": summarize(classic_rets),
                    "opp_return": summarize(classic_opp_rets),
                    "seats": classic_seat,
                },
                "om": None,
            }
        )
        print("  No OM checkpoint present; evaluated Classic-H only.")
        continue

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

    # ---- OM ablation controls (R3: separate meaningful content from an
    # extra channel / input-distribution shift). Override the agent's
    # select_action so the heatmap is post-processed BEFORE Q consumes it.
    _mode = args_parsed.om_control
    if _mode != "none":
        import torch.nn.functional as _F
        from torch.distributions import Categorical as _Cat

        _rng = np.random.default_rng(args_parsed.seed + 7)
        if _mode == "random_window":
            _pool = _UnrelatedPool(agent_om, env, H, W, args_parsed.seed)
            _pool.fill(48)

        def _apply_control(gh, s_raw):
            """gh: (1, H, W) softmaxed. s_raw: raw obs (H, W, F). -> (1, H, W)"""
            if _mode == "blind":
                return torch.zeros_like(gh)
            if _mode == "uniform":
                # Uniform over ACCESSIBLE cells = complement of walls (channel 4).
                # Raw obs channel 0 is *visible empty* (fog-masked), which would
                # not match the Q-net's heatmap support: use 1 - wall instead.
                free = 1.0 - torch.from_numpy(s_raw[:, :, 4]).float().to(gh.device)
                free = free / free.sum().clamp(min=1e-8)
                return free.view_as(gh)
            if _mode == "shuffled":
                flat = gh.reshape(-1).detach().cpu().numpy().copy()
                _rng.shuffle(flat)
                return torch.from_numpy(flat).to(gh.device).view_as(gh)
            if _mode == "random_window":
                pick = _pool.be_library[_rng.integers(len(_pool.be_library))]
                return pick.to(gh.device).view_as(gh)
            return gh

        @torch.no_grad()
        def _om_select_controlled(s_t, s_aug, history, eval=False):
            a_dummy = None
            x = torch.from_numpy(s_t).float().unsqueeze(0).to(device)
            x_aug = torch.from_numpy(s_aug).float().unsqueeze(0).to(device)
            g_logits = agent_om.model(x, history)
            g_real = _F.softmax(g_logits.view(g_logits.shape[0], -1), dim=-1).view_as(
                g_logits
            )
            g_mod = _apply_control(g_real, s_t)

            qvals = agent_om.q(x_aug, g_mod)
            tau = 0.05 if eval else agent_om._tau()
            entropy = _Cat(logits=qvals / 0.05).entropy().item()
            a = agent_om.choose_action(qvals, tau, eval)
            return a, g_mod.squeeze(0), entropy

        agent_om.select_action = _om_select_controlled

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
        f"  {'classic-H' if args_parsed.classic_h else 'classic'}: {np.mean(classic_rets):.3f} +- {np.std(classic_rets, ddof=1):.3f} "
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
om_means = [s["om"]["return"]["mean"] for s in results["seeds"] if s["om"] is not None]
results["agents"] = {
    "classic": summarize(classic_means),
    "om": summarize(om_means) if om_means else None,
    "classic_opp": summarize(
        [s["classic"]["opp_return"]["mean"] for s in results["seeds"]]
    ),
    "om_opp": summarize(
        [s["om"]["opp_return"]["mean"] for s in results["seeds"] if s["om"] is not None]
    )
    if om_means
    else None,
}
if len(om_means) > 1:
    diffs = [
        s["om"]["return"]["mean"] - s["classic"]["return"]["mean"]
        for s in results["seeds"]
        if s["om"] is not None
    ]
    results["paired"] = {"om_minus_classic": summarize(diffs)}

print(
    f"\n=== AGGREGATE over {len(folder_ids)} seed(s) vs {args_parsed.opponent.upper()} "
    f"(swap_prob={args_parsed.swap_prob}, {args_parsed.episodes} eps each) ==="
)
print(
    f"{'classic-H' if args_parsed.classic_h else 'classic'}: {np.mean(classic_means):.3f} +- {np.std(classic_means, ddof=1) if len(classic_means) > 1 else 0:.3f}"
)
if om_means:
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
    f"_p{int(args_parsed.belief_map_prior)}_n{args_parsed.n_step}_sw{args_parsed.swap_prob}"
    f"_blind{'T' if args_parsed.om_control == 'blind' else 'F'}"
    f"{'_' + args_parsed.om_control if args_parsed.om_control not in ('none', 'blind') else ''}"
    f"{'_classicH' if args_parsed.classic_h else ''}.json",
)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved results to {out_path}")
