"""Reward-shaping tests for TeamRoadmapEnv.

Checks:
  1. shaping=False == shaping=True with zero weights (plumbing is a no-op:
     same rewards, observations, done, team_scores on identical layouts).
  2. Novelty term == beta * (own post-move visibility minus the team's
     pre-step coverage), per agent; scripted opponents (team 1) unshaped.
  3. Goal-approach term == alpha * (d_t - d_t1) w.r.t. the goal set visible
     BEFORE the move; one improving step pays exactly alpha.
  4. Telescoping (the potential-based shaping invariant): walking to a
     visible goal pays alpha * d0 in total approach reward, plus the goal
     share on the final step; team_scores are unaffected by shaping.
  5. End-to-end: scripted greedy episodes collect identical goals with and
     without shaping (shaping only changes rewards, never dynamics).

Run:  /opt/homebrew/anaconda3/envs/om/bin/python tests/test_shaping.py
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from omexplore.envs.roadmap_foraging_env import MOVES, TeamRoadmapEnv

FREEZE = 99  # not 0-7 -> _try_move no-op (agent stays put)


def _legal_target(env, pos, dr, dc):
    """Replicates _try_move's legality (bounds, walls, no corner cutting)."""
    nr, nc = pos[0] + dr, pos[1] + dc
    if not (0 <= nr < env.height and 0 <= nc < env.width):
        return None
    if (nr, nc) in env.walls:
        return None
    if (
        dr
        and dc
        and ((pos[0] + dr, pos[1]) in env.walls or (pos[0], pos[1] + dc) in env.walls)
    ):
        return None
    return (nr, nc)


def _legal_actions(env, agent_id):
    pos = env.agents[agent_id]
    out = []
    for dr, dc, a in MOVES:
        tgt = _legal_target(env, pos, dr, dc)
        if tgt is not None and tgt != pos:
            out.append((a, tgt))
    return out


def _improving_action(env, agent_id, goal):
    """Action that reduces the BFS distance to goal by exactly 1."""
    pos = env.agents[agent_id]
    field = env._dist_field(goal)
    d = field[pos]
    assert d > 0, "already at the goal"
    for dr, dc, a in MOVES:
        tgt = _legal_target(env, pos, dr, dc)
        if tgt is not None and field[tgt] == d - 1:
            return a
    raise AssertionError("no improving legal move (BFS guarantees one)")


def _pick_goal(env, agent_id=0, min_dist=3):
    """Free cell visible to the agent's team, BFS-distance d >= min_dist."""
    t = env.teams[agent_id]
    vis = env._team_vis[t]
    field = env._dist_field(env.agents[agent_id])
    occupied = set(env.agents.values())
    for want in range(min_dist, 12):  # prefer the nearest far-enough goal
        for r in range(env.height):
            for c in range(env.width):
                p = (r, c)
                if (
                    vis[r, c]
                    and p not in env.walls
                    and p not in occupied
                    and field[p] == want
                ):
                    return p
    raise AssertionError("no visible free cell at distance >= 3")


def test_zero_weights_are_noop():
    np.random.seed(7)
    plain = TeamRoadmapEnv(max_steps=20, num_goals=8, team_sizes=(2, 2))
    np.random.seed(7)
    off = TeamRoadmapEnv(
        max_steps=20,
        num_goals=8,
        team_sizes=(2, 2),
        shaping=True,
        shaping_alpha=0.0,
        shaping_beta=0.0,
    )
    acts = {a: a % 8 for a in plain.agents}
    for _ in range(3):
        obs_a, rew_a, done_a, info_a = plain.step(acts)
        obs_b, rew_b, done_b, info_b = off.step(acts)
        for a in rew_a:
            assert rew_a[a] == rew_b[a], (rew_a, rew_b)
            assert np.array_equal(obs_a[a], obs_b[a])
        assert done_a == done_b
        assert info_a["team_scores"] == info_b["team_scores"]
        assert info_b["team_shaping"] == {0: 0.0} or all(
            v == 0.0 for v in info_b["team_shaping"].values()
        )
    print("[ok] zero shaping weights are a no-op")


def test_novelty_exact():
    beta = 1e-3
    env = TeamRoadmapEnv(
        max_steps=30,
        vision_radius=10,
        num_goals=8,
        team_sizes=(2, 2),
        shaping=True,
        shaping_alpha=0.0,
        shaping_beta=beta,
    )
    env.reset()
    env.food_positions = set()  # isolate the novelty term (no goals)

    total_new = 0
    for step in range(3):
        # Move agent 0 away from its teammate to guarantee fresh vision.
        p0, p1 = env.agents[0], env.agents[1]
        best = max(
            _legal_actions(env, 0),
            key=lambda at: abs(at[1][0] - p1[0]) + abs(at[1][1] - p1[1]),
        )
        acts = {0: best[0], 1: FREEZE, 2: FREEZE, 3: FREEZE}
        pre_cov = env._coverage[0].copy()
        _, rewards, _, info = env.step(acts)

        vis_after = env.get_visibility_map(0).astype(bool)
        expected = int((vis_after & ~pre_cov).sum())
        assert abs(rewards[0] - beta * expected) < 1e-9, (
            step,
            rewards[0],
            beta * expected,
        )
        # Frozen teammate and the unshaped opponent team get nothing.
        assert rewards[1] == 0.0, rewards
        assert rewards[2] == 0.0 and rewards[3] == 0.0, rewards
        assert abs(info["team_shaping"][0] - beta * expected) < 1e-9
        total_new += expected
    assert total_new > 0, "walk never entered novel territory"
    print(f"[ok] novelty == beta * newly covered cells ({total_new} cells)")


def test_approach_single_step():
    alpha = 0.05
    env = TeamRoadmapEnv(
        max_steps=30,
        vision_radius=10,
        num_goals=8,
        team_sizes=(2, 2),
        shaping=True,
        shaping_alpha=alpha,
        shaping_beta=0.0,
    )
    env.reset()
    g = _pick_goal(env, agent_id=0)
    env.food_positions = {g}
    d0 = env._dist_field(g)[env.agents[0]]

    act = _improving_action(env, 0, g)
    acts = {0: act, 1: FREEZE, 2: FREEZE, 3: FREEZE}
    _, rewards, done, info = env.step(acts)
    assert not done
    assert abs(rewards[0] - alpha * 1.0) < 1e-9, rewards  # d0 -> d0-1
    assert rewards[1] == 0.0, rewards  # frozen teammate
    assert abs(info["team_shaping"][0] - alpha) < 1e-9
    assert info["team_scores"] == {0: 0.0, 1: 0.0}

    # Moving away is punished symmetrically (anti-loafing).
    p = env.agents[0]
    field = env._dist_field(g)
    away = [(a, tgt) for a, tgt in _legal_actions(env, 0) if field[tgt] == field[p] + 1]
    if away:
        _, rewards, _, _ = env.step({0: away[0][0], 1: FREEZE, 2: FREEZE, 3: FREEZE})
        assert abs(rewards[0] + alpha) < 1e-9, rewards
        print("[ok] approach: +alpha toward visible goal, -alpha away")
    else:
        print("[ok] approach: +alpha toward visible goal (no away-move here)")


def test_walk_telescoping():
    alpha = 0.02
    env = TeamRoadmapEnv(
        max_steps=60,
        vision_radius=10,
        num_goals=8,
        team_sizes=(2, 2),
        shaping=True,
        shaping_alpha=alpha,
        shaping_beta=0.0,
    )
    env.reset()
    g = _pick_goal(env, agent_id=0)
    env.food_positions = {g}
    d0 = env._dist_field(g)[env.agents[0]]

    approach_total = 0.0
    steps = 0
    done = False
    while not done:
        act = _improving_action(env, 0, g)
        _, rewards, done, info = env.step({0: act, 1: FREEZE, 2: FREEZE, 3: FREEZE})
        steps += 1
        assert steps <= d0 + 2, "walked past the goal"
        approach_total += rewards[0] - (1.0 if done else 0.0)

    assert steps == d0, (steps, d0)
    assert env.agents[0] == g
    # Potential-based telescoping: total approach reward == alpha * d0.
    assert abs(approach_total - alpha * d0) < 1e-9, (approach_total, alpha * d0)
    assert env.team_scores == {0: 1.0, 1: 0.0}
    print(
        f"[ok] walk telescoping: {steps} steps, approach {approach_total:.4f} == alpha*d0 ({alpha}*{d0})"
    )


def test_scripted_scores_unaffected():
    """Shaping changes rewards, never dynamics: scripted greedy episodes
    must collect identical goals with and without shaping."""
    from omexplore.agents.team_agents import TeamAgent

    def run(shaping):
        random.seed(3)
        np.random.seed(3)
        env = TeamRoadmapEnv(
            max_steps=40,
            vision_radius=5,
            num_goals=8,
            team_sizes=(2, 2),
            shaping=shaping,
            shaping_alpha=0.02,
            shaping_beta=2.5e-4,
        )
        env.reset()
        agents = [TeamAgent(env, team_id=t) for t in (0, 1)]
        done = False
        steps = 0
        shaped_reward = 0.0
        while not done and steps < 40:
            actions = {}
            for ta in agents:
                actions.update(ta.select_actions(env._get_ego_centric_obs()))
            _, rewards, done, _ = env.step(actions)
            shaped_reward += sum(rewards[a] for a in (0, 1))
            steps += 1
        return dict(env.team_scores), steps, shaped_reward

    scores_plain, steps_plain, _ = run(False)
    scores_shaped, steps_shaped, shaped_reward = run(True)
    assert scores_plain == scores_shaped, (scores_plain, scores_shaped)
    assert steps_plain == steps_shaped
    assert shaped_reward > 2 * scores_shaped[0], (
        "expected shaping mass on top of goal rewards",
        shaped_reward,
        scores_shaped,
    )
    print(
        f"[ok] scripted dynamics identical with/without shaping "
        f"(scores {scores_shaped}, shaped team return {shaped_reward:.2f})"
    )


def main():
    test_zero_weights_are_noop()
    test_novelty_exact()
    test_approach_single_step()
    test_walk_telescoping()
    test_scripted_scores_unaffected()
    print("\ntest_shaping: ALL PASSED")


if __name__ == "__main__":
    main()
