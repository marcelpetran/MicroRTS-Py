"""Diagnose whether the trained Q-net actually uses its belief channels.

Motivation: the belief prior fix put all 64 goal positions into belief
channel 0, yet the latest run is unchanged (eval_return ~3, uniform
policy). Hypothesis: the 3x(3x3)-conv backbone (7x7 receptive field) +
Flatten->Linear heads cannot turn distant goal cells into a routing
decision, so the net learned to ignore the channel.

Three probes on a trained qnet.pth (or random weights without --qnet):

  1. Ablation   Q(s, belief) vs Q(s, empty belief) on real env states.
                Small |dQ| / few argmax flips => channel ignored.
  2. Adjacency  Plant one believed goal on each free NEIGHBOUR of self.
                A healthy Q-net must prefer stepping onto it. This signal
                is locally representable (7x7 RF), so failure here means
                even local goal-seeking was never learned.
  3. Distance   Same but at distance 3 and 10 (still inside vision);
                Q(onto goal) should not depend much on where the goal is.

Run on the machine holding the checkpoint, e.g.:
  python scripts/probe_qnet_belief.py --qnet ./models/11609947/qnet.pth \
      --qnet_dim 512
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

import wandb  # QNet imports reference wandb at package import
from omexplore.envs.roadmap_foraging_env import TeamRoadmapEnv
from omexplore.models.beliefs import BeliefTracker
from omexplore.models.networks import QNet
from omexplore.utils.omg_args import OMGArgs

# action -> (dr, dc); 0-3 Up/Down/Left/Right, 4-7 diagonals (env order).
ACTION_DELTAS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
    4: (-1, -1),
    5: (-1, 1),
    6: (1, -1),
    7: (1, 1),
}


def build_qnet(a, env):
    args = OMGArgs(
        state_shape=(env.height, env.width, env.features),
        belief_channels=3,
        friendly_om=True,
        qnet_hidden=a.qnet_dim,
        cnn_hidden=a.cnn_hidden,
    )
    args.action_dim = 8
    q = QNet(args)
    if a.qnet:
        q.load_state_dict(torch.load(a.qnet, map_location="cpu"))
        print(f"Loaded checkpoint: {a.qnet}")
    else:
        print("No --qnet given: probing RANDOM weights (script smoke test)")
    q.eval()
    return q


def q_values(q, obs, belief):
    """Q(s, belief) with zeroed OM heatmaps; returns (8,) numpy."""
    s_aug = np.concatenate([obs.astype(np.float32), belief], axis=-1)
    x = torch.from_numpy(s_aug).float().unsqueeze(0)
    g0 = torch.zeros((1, obs.shape[0], obs.shape[1]))
    with torch.no_grad():
        out = q(x, g0, g0)
    return out.squeeze(0).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qnet", type=str, default=None, help="qnet.pth checkpoint")
    p.add_argument("--map", type=str, default="den312d")
    p.add_argument("--num_goals", type=int, default=64)
    p.add_argument("--vision_radius", type=int, default=10)
    p.add_argument("--team_sizes", type=str, default="2,2")
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--qnet_dim", type=int, default=512)
    p.add_argument("--cnn_hidden", type=int, default=64)
    p.add_argument("--n_states", type=int, default=32, help="env states to probe")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    wandb.init(mode="disabled")

    team_sizes = tuple(int(x) for x in a.team_sizes.split(","))
    env = TeamRoadmapEnv(
        map_name=a.map,
        max_steps=a.max_steps,
        vision_radius=a.vision_radius,
        num_goals=a.num_goals,
        team_sizes=team_sizes,
    )
    q = build_qnet(a, env)
    n_params = sum(pp.numel() for pp in q.parameters())
    print(f"QNet params: {n_params / 1e6:.1f}M")

    # ------------------------------------------------------------------ #
    # Probe 1: belief ablation on real states (full prior vs empty).
    # ------------------------------------------------------------------ #
    dq_ablate, flips, spreads_full, spreads_empty = [], 0, [], []
    for i in range(a.n_states):
        obs = env.reset()
        tr = BeliefTracker(
            env.height, env.width, horizon=env.max_steps, channels=(1, 4, 6)
        )
        tr.set_food_prior(env.food_positions)
        tr.set_opp_prior(
            [
                pos
                for aid, pos in env.get_agent_positions().items()
                if env.teams[aid] != 0
            ]
        )
        tr.reset(use_map_prior=True)
        tr.update(obs[0])
        full = tr.channels()
        empty = tr.channels()  # belief_food/belief_opp are empty here
        empty[:, :, :] = 0.0
        empty[:, :, 2] = full[:, :, 2]  # keep opp_age; zero the position sets

        qf = q_values(q, obs[0], full)
        qe = q_values(q, obs[0], empty)
        dq_ablate.append(np.abs(qf - qe).mean())
        if int(np.argmax(qf)) != int(np.argmax(qe)):
            flips += 1
        spreads_full.append(qf.max() - qf.min())
        spreads_empty.append(qe.max() - qe.min())

    print("\n=== Probe 1: belief ablation (real states, prior vs empty) ===")
    print(f"mean |dQ|          : {np.mean(dq_ablate):.4f}")
    print(f"argmax flips       : {flips}/{a.n_states}")
    print(f"spread w/ belief   : {np.mean(spreads_full):.4f}")
    print(f"spread w/o belief  : {np.mean(spreads_empty):.4f}")

    # ------------------------------------------------------------------ #
    # Probe 2/3: planted believed goal near the self agent. The expected
    # action is the FIRST step of the true BFS path (walls may bend it).
    # ------------------------------------------------------------------ #
    for dist in (1, 3, 10):
        correct, tested, spread = 0, 0, []
        for i in range(a.n_states):
            obs = env.reset()
            r0, c0 = env.agents[0]
            for act, (dr, dc) in ACTION_DELTAS.items():
                gr, gc = r0 + dr * dist, c0 + dc * dist
                if not (0 <= gr < env.height and 0 <= gc < env.width):
                    continue
                if (gr, gc) in env.walls:
                    continue
                path = env.find_path((r0, c0), (gr, gc))
                if not path:
                    continue  # unreachable
                first = path[0]  # true greedy action toward the planted goal
                belief = np.zeros((env.height, env.width, 3), dtype=np.float32)
                belief[gr, gc, 0] = 1.0  # one believed goal only
                qv = q_values(q, obs[0], belief)
                spread.append(qv.max() - qv.min())
                tested += 1
                if int(np.argmax(qv)) == first:
                    correct += 1
                break  # one direction per state is enough
        print(f"\n=== Probe 2: believed goal at distance {dist} ===")
        if tested == 0:
            print("no valid placement found (spawn next to walls?)")
            continue
        acc = correct / tested
        print(f"argmax = BFS-toward-goal : {correct}/{tested} ({acc:.0%})")
        print(f"mean Q spread           : {np.mean(spread):.4f}")
        print("(random policy would score ~1/8 = 12.5%)")

    print("\nInterpretation:")
    print("- Probe1 |dQ| ~ 0 AND Probe2 ~ 12%  => belief channel ignored;")
    print("  information reaches the input but the function class / training")
    print("  never used it (architecture bottleneck).")
    print("- Probe2 >> 12% at dist 1 but Probe1 |dQ| ~ 0 => net uses only very")
    print("  local goal info; distant believed goals (the prior) still unused.")


if __name__ == "__main__":
    main()
