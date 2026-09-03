"""Render a team-exploration episode to a GIF (+ key PNG frames) — showcase.

Two modes:

  Mode A (default, no checkpoints needed):
      two scripted TeamAgents (personas) compete; side panels show each
      team's TRUE claim map (the oracle intent heatmap).
        python scripts/render_team_episode.py --personas1 switch,stalker

  Mode B (with trained checkpoints):
      the learning team (QLearningAgent, one shared Q-net) plays against a
      scripted team; side panels additionally show the hostile/friendly OMs'
      PREDICTED claim maps next to the hostile oracle intent — the
      "is the OM actually reading the opponent?" money shot.
        python scripts/render_team_episode.py --mode rl \
            --qnet ./models/X/qnet.pth --hostile_om ./models/pretrained_oms/hostile_om.pth \
            --friendly_om ./models/pretrained_oms/friendly_om.pth

Architecture flags (d_model, nhead, ...) must match the run that produced
the checkpoints (defaults match pretrain_team_oms.py / train_team_exploration.py).

Output: <out> GIF plus <out>.frame{0,mid,last}.png key frames.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
import torch.nn.functional as F

import wandb  # QLearningAgent/OpponentModel reference wandb at import
from omexplore.agents.q_agent import QLearningAgent
from omexplore.agents.team_agents import TeamAgent
from omexplore.envs.roadmap_foraging_env import TeamRoadmapEnv
from omexplore.models.opponent_model import OpponentModel
from omexplore.models.transformers import SpatialOpponentModel
from omexplore.utils.omg_args import OMGArgs
from omexplore.utils.team_renderer import render_episode, snapshot


def parse_personas(s):
    return tuple(p.strip() for p in s.split(",") if p.strip())


def build_env(a):
    team_sizes = tuple(int(x) for x in a.team_sizes.split(","))
    return TeamRoadmapEnv(
        map_name=a.map,
        max_steps=a.max_steps,
        vision_radius=a.vision_radius,
        num_goals=a.num_goals,
        team_sizes=team_sizes,
    )


# ----------------------------------------------------------------------
# Mode A: scripted vs scripted (true intents only)
# ----------------------------------------------------------------------
def rollout_scripted(env, personas0, personas1):
    cache = {}
    team0 = TeamAgent(env, team_id=0, personas=personas0, field_cache=cache)
    team1 = TeamAgent(env, team_id=1, personas=personas1, field_cache=cache)
    obs = env.reset()
    team0.reset()
    team1.reset()

    frames = [
        snapshot(
            env,
            0,
            hostile_true=team1.get_team_heatmap(),
            friendly_true=team0.get_team_heatmap(),
        )
    ]
    done = False
    step = 0
    while not done and step < env.max_steps:
        step += 1
        actions = {**team0.select_actions(obs), **team1.select_actions(obs)}
        obs, _, done, _ = env.step(actions)
        frames.append(
            snapshot(
                env,
                step,
                hostile_true=team1.get_team_heatmap(),
                friendly_true=team0.get_team_heatmap(),
            )
        )
    return frames


# ----------------------------------------------------------------------
# Mode B: RL team (checkpoints) vs scripted team (predictions + oracle)
# ----------------------------------------------------------------------
def rollout_rl(env, args, a):
    hostile_om = OpponentModel(SpatialOpponentModel(args), args)
    friendly_om = OpponentModel(SpatialOpponentModel(args), args)
    agent = QLearningAgent(env, hostile_om, friendly_om, args=args)

    agent.q.load_state_dict(torch.load(a.qnet, map_location=args.device))
    agent.q.eval()
    if a.hostile_om:
        agent.model.inference_model.load_state_dict(
            torch.load(a.hostile_om, map_location=args.device)
        )
    if a.friendly_om:
        agent.team_model.inference_model.load_state_dict(
            torch.load(a.friendly_om, map_location=args.device)
        )
    agent.model.inference_model.eval()
    agent.team_model.inference_model.eval()
    print(f"Loaded qnet={a.qnet} hostile_om={a.hostile_om} friendly_om={a.friendly_om}")

    opp = TeamAgent(env, team_id=1, personas=parse_personas(a.personas1))
    obs = env.reset()
    opp.reset()
    agent.tracker.reset(use_map_prior=args.belief_map_prior)
    anchor = agent.learn_ids[0]
    agent.tracker.update(obs[anchor])

    history_len = args.max_history_length
    rolling_feats = torch.zeros((1, history_len, args.d_model), device=agent.device)
    rolling_mask = torch.zeros((1, history_len), dtype=torch.bool, device=agent.device)
    cur_len = 0
    prev_state = torch.zeros((1, *obs[anchor].shape), device=agent.device)

    def om_prediction(x_np, history):
        """Hostile + friendly OM claim maps for one member's obs."""
        x = (
            torch.from_numpy(np.asarray(x_np, dtype=np.float32))
            .unsqueeze(0)
            .to(agent.device)
        )
        with torch.no_grad():
            gh = agent.model(x, history)
            gh = F.softmax(gh.view(1, -1), dim=-1).view_as(gh).squeeze(0)
            gf = None
            if args.friendly_om:
                gf = agent.team_model(x, history)
                gf = F.softmax(gf.view(1, -1), dim=-1).view_as(gf).squeeze(0)
        return gh, gf

    frames = []
    done = False
    step = 0
    while not done and step < env.max_steps:
        step += 1
        history = {
            "state_features": rolling_feats,
            "mask": rolling_mask,
            "prev_obs": prev_state,
        }

        # Anchor first: capture both OM predictions for the side panels.
        gh, gf = om_prediction(obs[anchor], history)
        actions = {}
        for aid in agent.learn_ids:
            s_aug = agent.tracker.augment(obs[aid])
            act, _, _ = agent.select_action(obs[aid], s_aug, history, eval=True)
            actions[aid] = act

        opp_actions = opp.select_actions(obs)
        for aid in agent.hostile_ids:
            actions[aid] = opp_actions[aid]

        frames.append(
            snapshot(
                env,
                step,
                hostile_pred=gh,
                friendly_pred=gf,
                hostile_true=opp.get_team_heatmap(),
            )
        )

        next_obs, _, done, _ = env.step(actions)
        agent.tracker.update(next_obs[anchor])

        st = torch.from_numpy(obs[anchor]).float().unsqueeze(0).to(agent.device)
        with torch.no_grad():
            new_feat = agent.model.inference_model.get_features(st, prev_state)
        rolling_feats = torch.roll(rolling_feats, shifts=-1, dims=1)
        rolling_mask = torch.roll(rolling_mask, shifts=-1, dims=1)
        rolling_feats[:, -1, :] = new_feat
        if cur_len < history_len:
            cur_len += 1
        rolling_mask[:, -cur_len:] = True
        prev_state = st
        obs = next_obs
    return frames


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--mode", choices=["scripted", "rl"], default="scripted")
    p.add_argument("--map", type=str, default="den312d")
    p.add_argument("--num_goals", type=int, default=16)
    p.add_argument("--max_steps", type=int, default=400)
    p.add_argument("--vision_radius", type=int, default=5)
    p.add_argument("--team_sizes", type=str, default="2,2")
    p.add_argument("--personas0", type=str, default="greedy,greedy")
    p.add_argument("--personas1", type=str, default="greedy,greedy")
    p.add_argument("--out", type=str, default="./diagrams/team_episode.gif")
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--title", type=str, default="Team-based competitive exploration")
    p.add_argument("--seed", type=int, default=None)
    # Mode B: checkpoints + architecture (must match training).
    p.add_argument("--qnet", type=str, default=None)
    p.add_argument("--hostile_om", type=str, default=None)
    p.add_argument("--friendly_om", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max_history_length", type=int, default=8)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_encoder_layers", type=int, default=1)
    p.add_argument("--dim_feedforward", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--cnn_hidden", type=int, default=64)
    a = p.parse_args()

    if a.seed is not None:
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)

    wandb.init(mode="disabled")
    env = build_env(a)
    if a.mode == "scripted":
        frames = rollout_scripted(
            env, parse_personas(a.personas0), parse_personas(a.personas1)
        )
        title = a.title
    else:
        if not a.qnet:
            p.error("--mode rl requires --qnet <path.pth>")
        device = a.device
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        args = OMGArgs(
            device=device,
            state_shape=(env.height, env.width, env.features),
            H=env.height,
            W=env.width,
            max_steps=a.max_steps,
            max_history_length=a.max_history_length,
            d_model=a.d_model,
            nhead=a.nhead,
            num_encoder_layers=a.num_encoder_layers,
            dim_feedforward=a.dim_feedforward,
            dropout=a.dropout,
            cnn_hidden=a.cnn_hidden,
        )
        frames = rollout_rl(env, args, a)
        title = a.title

    gif, pngs = render_episode(frames, a.out, fps=a.fps, title=title)
    print(f"\nSaved GIF: {gif}")
    for png in pngs:
        print(f"Saved PNG: {png}")
    s0 = frames[-1].scores.get(0, 0.0)
    s1 = frames[-1].scores.get(1, 0.0)
    print(f"Final score: team0={s0:g} team1={s1:g} ({frames[-1].step} steps)")


if __name__ == "__main__":
    main()
