"""Sanity: get_coverage must stay in [0, 1] on den312d after the wall-mask fix."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from omexplore.agents.team_agents import TeamAgent
from omexplore.envs.roadmap_foraging_env import TeamRoadmapEnv


def main():
    np.random.seed(0)
    env = TeamRoadmapEnv(
        map_name="den312d",
        max_steps=400,
        vision_radius=10,
        num_goals=32,
        team_sizes=(2, 2),
    )
    print(
        f"map {env.height}x{env.width}: {len(env.walls)} walls, "
        f"{env.height * env.width - len(env.walls)} free cells, "
        f"old saturation ceiling "
        f"{env.height * env.width / (env.height * env.width - len(env.walls)):.3f}"
    )
    a = TeamAgent(env, team_id=0)
    b = TeamAgent(env, team_id=1)
    obs = env.reset()
    a.reset()
    b.reset()
    done = False
    steps = 0
    while not done:
        acts = {}
        acts.update(a.select_actions(obs))
        acts.update(b.select_actions(obs))
        obs, _, done, info = env.step(acts)
        steps += 1
        for t, c in info["coverage"].items():
            assert 0.0 <= c <= 1.0, (steps, t, c)
    print(
        f"{steps} steps, scores {dict(env.team_scores)}, "
        f"coverage {{0: {env.get_coverage(0):.3f}, 1: {env.get_coverage(1):.3f}}} "
        f"(both in [0, 1])"
    )
    print("coverage bounds OK")


if __name__ == "__main__":
    main()
