import os
import random
import time
from typing import Dict, List

import numpy as np
import torch

from omexplore.agents.team_agents import TeamAgent
from omexplore.envs.roadmap_foraging_env import TeamRoadmapEnv
from omexplore.envs.simple_foraging_env import (
    ChameleonAgent,
    GreedySwitchAgent,
    SimpleAgent,
    SimpleForagingEnv,
    StalkerAgent,
)
from omexplore.utils.labeling import compute_agent_goals, sparse_claims
from omexplore.utils.maps import *
from omexplore.utils.omg_args import OMGArgs
from omexplore.utils.packing import pack_obs
from omexplore.utils.renderer import RealtimeRenderer


def _apply_hindsight_relabeling(episode_transitions: List, H: int, W: int):
    """
    Applies Hindsight Experience Replay (HER) labeling to a trajectory.
    Modifies the transitions in-place to include 'true_goal_map'.
    """
    current_true_goal_pos = None

    if len(episode_transitions) > 0:
        final_t = episode_transitions[-1]

        if final_t["opp_reward"] == 0:
            opp_pos_arr = np.argwhere(final_t["global_state"][:, :, 3] == 1)
            if len(opp_pos_arr) > 0:
                current_true_goal_pos = tuple(opp_pos_arr[0])

    for t in reversed(episode_transitions):
        if t["opp_reward"] > 0:
            opp_pos_indices = np.argwhere(t["next_global_state"][:, :, 3] == 1)
            if len(opp_pos_indices) > 0:
                current_true_goal_pos = tuple(opp_pos_indices[0])

        true_map = np.zeros((H, W), dtype=np.float32)
        if current_true_goal_pos is not None:
            true_map[current_true_goal_pos[0], current_true_goal_pos[1]] = 1.0

        t["true_goal_map"] = true_map

        del t["opp_reward"]


def collect_offline_data(
    num_episodes: int = 1000,
    save_path: str = "./dataset/dataset.pt",
    map_layout: list[str] = MAP_1,
    om_args: OMGArgs = OMGArgs(),
):
    args = om_args
    env = SimpleForagingEnv(max_steps=args.max_steps, map_layout=map_layout)
    obs = env.reset()
    agent_0 = SimpleAgent(0, map_layout=map_layout)
    # precompute paths for other agents to use during data collection
    # Dummy action to trigger path precomputation in the environment
    _ = agent_0.select_action(obs[0])
    precomputed_paths = agent_0.precomputed_paths
    agent_1 = SimpleAgent(1, precomputed_paths=precomputed_paths, map_layout=map_layout)
    agent_2 = GreedySwitchAgent(
        0, precomputed_paths=precomputed_paths, map_layout=map_layout
    )
    agent_3 = GreedySwitchAgent(
        1, precomputed_paths=precomputed_paths, map_layout=map_layout
    )
    agent_4 = StalkerAgent(
        0, precomputed_paths=precomputed_paths, map_layout=map_layout
    )
    agent_5 = StalkerAgent(
        1, precomputed_paths=precomputed_paths, map_layout=map_layout
    )
    agent_combinations = [
        (agent_0, agent_1),
        (agent_0, agent_3),
        (agent_2, agent_1),
        (agent_2, agent_3),
    ]

    master_dataset = []

    print(f"Starting offline data collection for {num_episodes} episodes...")
    for agent0, agent1 in agent_combinations:
        print(
            f"Collecting data for agent combination: {agent0.__class__.__name__} vs {agent1.__class__.__name__}"
        )
        for ep in range(num_episodes):
            obs = env.reset()
            if random.random() < 0.3:
                obs = env.reset_random_spawn()
            elif random.random() < 0.5:
                # 50% of the time swap spawns to add more diversity
                obs = env.swap_agents()
            agent0.reset()
            agent1.reset()

            episode_transitions = []
            ep_states = []

            H, W, _ = obs[0].shape

            for step in range(args.max_steps):
                # Both agents act using pure heuristics
                a_0, _, _ = agent0.select_action(obs[0])
                a_1, _, true_opp_heatmap = agent1.select_action(obs[1])
                actions = {0: a_0, 1: a_1}

                global_state = env.get_global_state()
                next_obs, reward, done, info = env.step(actions)
                next_global_state = env.get_global_state()

                transition = {
                    "state": obs[0].copy(),
                    "global_state": global_state.copy(),
                    "reward": float(reward[0]),
                    "opp_reward": float(reward[1]),
                    "next_state": next_obs[0].copy(),
                    "next_global_state": next_global_state.copy(),
                    "done": bool(done),
                    "true_opp_heatmap": true_opp_heatmap.copy(),
                    "hist_len": len(ep_states) - 1,
                }
                ep_states.append(obs[0].copy())
                episode_transitions.append(transition)

                obs = next_obs
                if done:
                    break

            _apply_hindsight_relabeling(episode_transitions, H, W)

            states_arr = np.stack(ep_states)
            for t in episode_transitions:
                t["history"] = {"states": states_arr}
            master_dataset.extend(episode_transitions)

            if (ep + 1) % 100 == 0:
                print(
                    f"Collected {ep + 1} episodes... Total valid transitions: {len(master_dataset)}"
                )

    print(f"Saving dataset to {save_path}...")
    torch.save(master_dataset, save_path)
    print("Done!")


# To test heurisctic agents, no collection, only run episodes and render gameplay


def run_episode(agent0, agent1, env, args, render=False):
    obs = env.reset()
    agent0.reset()
    agent1.reset()
    total_reward_0 = 0
    total_reward_1 = 0
    renderer = RealtimeRenderer() if render else None

    for step in range(100):
        if render and renderer is not None:
            global_state = env.get_global_state()
            H, W, _ = global_state.shape
            dummy_om = np.zeros((H, W), dtype=np.float32)
            renderer.render(global_state, obs[0], obs[1], dummy_om)

        a_0, _, _ = agent0.select_action(obs[0])
        a_1, _, _ = agent1.select_action(obs[1])
        actions = {0: a_0, 1: a_1}

        next_obs, reward, done, info = env.step(actions)

        total_reward_0 += reward[0]
        total_reward_1 += reward[1]

        obs = next_obs
        if done:
            if render and renderer is not None:
                global_state = env.get_global_state()
                H, W, _ = global_state.shape
                dummy_om = np.zeros((H, W), dtype=np.float32)
                renderer.render(global_state, obs[0], obs[1], dummy_om)
            break


# ======================================================================#
# Team-based (2v2) offline collection for OM pretraining                #
# ======================================================================#

# (team-0 personas, team-1 personas): both teams scripted. The dataset
# records transitions from EVERY agent's perspective, so one episode
# trains both the hostile OM ("what is the other team claiming") and the
# friendly OM ("what are my teammates claiming") symmetrically.
DEFAULT_TEAM_COMBOS = [
    (("greedy", "greedy"), ("greedy", "greedy")),
    (("greedy", "greedy"), ("simple", "simple")),
    (("simple", "simple"), ("greedy", "greedy")),
    (("greedy", "greedy"), ("switch", "switch")),
    (("switch", "switch"), ("greedy", "greedy")),
    (("greedy", "greedy"), ("stalker", "greedy")),
    (("simple", "stalker"), ("greedy", "switch")),
    (("greedy", "simple"), ("switch", "greedy")),
]


def run_team_collection_episode(env: TeamRoadmapEnv, team_agents, records_out=None):
    """One scripted episode -> labeled transitions for every agent.

    Mirrors QLearningAgent.run_episode's recording order exactly:
    transitions are created at decision time (pre-step obs, hist_len =
    anchor states appended so far), the anchor obs stream is appended
    after the step's transitions, and hindsight claim labels are added at
    episode end via the shared labeling functions.

    records_out (optional dict) is filled with "step_records" and
    "final_positions" so tests can cross-check the labels against
    QLearningAgent._apply_hindsight_relabeling.

    Storage format (memory-lean for large maps):
      - state / history states are bit-packed (H, W, 1) uint8
        (unpack with omexplore.utils.packing.unpack_obs)
      - labels are sparse [(r, c, weight), ...] lists
      - "history"["states"] arrays are shared by reference per team.
    """
    num_teams = env.num_teams
    obs = env.reset()
    if random.random() < 0.3:
        obs = env.reset_random_spawn()
    elif random.random() < 0.5:
        obs = env.swap_agents()
    for ta in team_agents:
        ta.reset()

    all_agent_ids = list(env.agents.keys())
    teams = env.teams
    anchors = {t: env.get_team_members(t)[0] for t in range(num_teams)}
    anchor_states = {t: [] for t in range(num_teams)}

    transitions = []
    step_records = []

    done = False
    for step in range(env.max_steps):
        actions = {}
        for ta in team_agents:
            actions.update(ta.select_actions(obs))

        # Decision-time intent (each member's current target), sparse.
        intent = {ta.team_id: ta.get_team_heatmap_sparse() for ta in team_agents}

        for a in all_agent_ids:
            t = teams[a]
            hostile_cells = []
            for t2 in range(num_teams):
                if t2 != t:
                    hostile_cells.extend(intent[t2])
            own_ta = next(ta for ta in team_agents if ta.team_id == t)
            friendly_cells = own_ta.get_team_heatmap_sparse(exclude_id=a)
            transitions.append(
                {
                    "agent_id": a,
                    "team": t,
                    "step_idx": step,
                    "state": pack_obs(obs[a]),
                    "action": int(actions[a]),
                    "reward": 0.0,
                    "done": False,
                    "hist_len": len(anchor_states[t]),
                    "true_opp_heatmap_cells": hostile_cells,
                    "true_team_heatmap_cells": friendly_cells,
                }
            )

        next_obs, rewards, done, info = env.step(actions)
        for tr in transitions[-len(all_agent_ids) :]:
            tr["reward"] = float(rewards[tr["agent_id"]])
            tr["done"] = bool(done)
        step_records.append({"collectors": info.get("collectors", {})})
        for t in range(num_teams):
            anchor_states[t].append(pack_obs(obs[anchors[t]]))

        obs = next_obs
        if done:
            break

    final_positions = env.get_agent_positions()
    if records_out is not None:
        records_out["step_records"] = step_records
        records_out["final_positions"] = final_positions
    if step_records:
        agent_goals = compute_agent_goals(step_records, final_positions)
    else:
        agent_goals = []
    # Stack each team's anchor stream ONCE and share the array by reference
    # across that team's transitions (stacking inside the transition loop
    # would copy the full stream per transition and OOM on large runs;
    # torch.save/pickle preserves the sharing, so the file stays small too).
    team_histories = {t: np.stack(s) for t, s in anchor_states.items() if len(s) > 0}
    for tr in transitions:
        if agent_goals:
            goals = agent_goals[tr["step_idx"]]
            hostile_agents = [a for a in env.agents if teams[a] != tr["team"]]
            tr["true_goal_cells"] = sparse_claims(hostile_agents, goals)
            mates = [m for m in env.get_team_members(tr["team"]) if m != tr["agent_id"]]
            tr["true_team_cells"] = sparse_claims(mates, goals)
        else:
            tr["true_goal_cells"] = []
            tr["true_team_cells"] = []
        tr["history"] = {"states": team_histories[tr["team"]]}
    return transitions


def collect_team_offline_data(
    num_episodes: int = 25,
    save_path: str = "./dataset/team_dataset.pt",
    map_name: str = "den312d",
    num_goals: int = 16,
    vision_radius: int = 5,
    max_steps: int = 400,
    team_sizes: tuple = (2, 2),
    team_combos=None,
    seed: int | None = None,
):
    """Collect 2v2 scripted trajectories for OM pretraining.

    Default scale: 8 persona combos x num_episodes episodes. Every episode
    yields ~steps x agents transitions (all four perspectives).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    if team_combos is None:
        team_combos = DEFAULT_TEAM_COMBOS

    env = TeamRoadmapEnv(
        map_name=map_name,
        max_steps=max_steps,
        vision_radius=vision_radius,
        num_goals=num_goals,
        team_sizes=team_sizes,
    )
    field_cache = {}  # shared across teams: fields depend only on walls

    master_dataset = []
    for combo in team_combos:
        team_agents = [
            TeamAgent(env, team_id=t, personas=personas, field_cache=field_cache)
            for t, personas in enumerate(combo)
        ]
        print(f"\nCollecting combo: {combo[0]} vs {combo[1]}")
        for ep in range(num_episodes):
            t0 = time.time()
            transitions = run_team_collection_episode(env, team_agents)
            master_dataset.extend(transitions)
            scores = env.team_scores
            print(
                f"  ep {ep + 1}/{num_episodes}: {len(transitions)} transitions, "
                f"scores {dict(scores)}, {time.time() - t0:.1f}s, "
                f"total {len(master_dataset)}"
            )

    print(f"\nSaving {len(master_dataset)} transitions to {save_path}...")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(master_dataset, save_path)
    print("Done!")
    return master_dataset


def _demo_legacy_agents():
    map_layout = MAP_4
    args = OMGArgs()
    env = SimpleForagingEnv(max_steps=args.max_steps, map_layout=map_layout)
    obs = env.reset()
    agent_0 = SimpleAgent(0, map_layout=map_layout)
    _ = agent_0.select_action(obs[0])
    precomputed_paths = agent_0.precomputed_paths
    agent_1 = SimpleAgent(1, precomputed_paths=precomputed_paths, map_layout=map_layout)
    agent_2 = GreedySwitchAgent(
        0, precomputed_paths=precomputed_paths, map_layout=map_layout
    )
    agent_3 = GreedySwitchAgent(
        1, precomputed_paths=precomputed_paths, map_layout=map_layout
    )
    agent_4 = StalkerAgent(
        0, precomputed_paths=precomputed_paths, map_layout=map_layout
    )
    agent_5 = StalkerAgent(
        1, precomputed_paths=precomputed_paths, map_layout=map_layout
    )

    for ep in range(1):
        print(f"Episode {ep + 1}")
        run_episode(agent_2, agent_3, env, args, render=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Offline data collection")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_team = sub.add_parser("team", help="2v2 team collection for OM pretraining")
    p_team.add_argument("--episodes", type=int, default=25, help="episodes per combo")
    p_team.add_argument("--save_path", type=str, default="./dataset/team_dataset.pt")
    p_team.add_argument("--map", type=str, default="den312d")
    p_team.add_argument("--num_goals", type=int, default=16)
    p_team.add_argument("--vision_radius", type=int, default=5)
    p_team.add_argument("--max_steps", type=int, default=400)
    p_team.add_argument("--team_sizes", type=str, default="2,2")
    p_team.add_argument("--seed", type=int, default=None)

    p_legacy = sub.add_parser("legacy", help="render a 1v1 heuristic demo")

    cli = parser.parse_args()
    if cli.mode == "team":
        collect_team_offline_data(
            num_episodes=cli.episodes,
            save_path=cli.save_path,
            map_name=cli.map,
            num_goals=cli.num_goals,
            vision_radius=cli.vision_radius,
            max_steps=cli.max_steps,
            team_sizes=tuple(int(s) for s in cli.team_sizes.split(",")),
            seed=cli.seed,
        )
    else:
        _demo_legacy_agents()
